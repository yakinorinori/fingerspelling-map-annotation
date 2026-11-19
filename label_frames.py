"""
動画を再生しながら、1フレームずつ「指文字か、それ以外か」のラベルを付けるツール。

キーで「ブラシ」を選ぶと、再生やコマ送りで通過したフレームにそのラベルが塗られる。
1 を押せば以降のフレームが「指文字」、2 を押せば「それ以外」になり、もう一度押し直すまで続く。

  ラベル: 1 = 指文字, 2 = それ以外, 0 = 未定（まだ付けていない、判断できない）

フォルダ内の動画を番号順に開き、完了済みは飛ばし、途中のものは続きから再開する。
フレーム番号は先頭から順に読んだ番号（0 始まり）で、骨格抽出と同じ数え方。

使い方:
    python label_frames.py --videos Video/文章 --out annotations_frames --annotator yamada
    python label_frames.py --files Video/文章/3.mp4 --annotator yamada
    （--list / --pick も使える。annotate.py の説明を参照）

キー操作（動画のウィンドウを選択した状態で押す）:
    1       ブラシ「指文字」（現在のフレームから塗り始める）
    2       ブラシ「それ以外」
    3       ブラシ「消しゴム」（未定に戻す）
    0       ブラシを離す（塗らずに見返す）
    space   再生 / 一時停止
    d / a   1フレーム進む / 戻る（ブラシを持っていれば通過したフレームを塗る）
    D / A   10フレーム進む / 戻る（c / z でも可）
    ] / [   再生速度を上げる / 下げる（0.1〜1倍）
    u       直前の操作を取り消す
    g       次の未定のフレームへ移動
    n       この動画を完了にして次へ（未定のフレームが残っていると確認される）
    p       完了にせず次へ（保存はされる）
    q       保存して終了
    h       この説明を表示

ラベルの境界をフレーム単位で正確に付けたい場合は、境界の手前で一時停止し、
コマ送り（a / d）で境界のフレームを確認してからブラシを切り替える。
人の反応は再生より遅れるので、通常速度のまま押すと境界が数フレーム遅れる。
"""

import argparse
import csv
import datetime
import os
import sys

import cv2
import numpy as np

from annotate import (SPEEDS, add_video_args, list_videos, load_frames, natural_key, resolve_videos,
                      select_todo)

PROGRESS_FILE = '_progress.csv'
PROGRESS_FIELDS = ['video', 'status', 'n_frames', 'n_fs', 'n_other', 'n_unlabeled', 'fps', 'annotator', 'updated']
COLORS = {0: (90, 90, 90), 1: (0, 180, 0), 2: (0, 140, 255)}  # BGR: 灰=未定, 緑=指文字, 橙=それ以外
BRUSH_NAMES = {None: 'OFF (view)', 1: '1 FINGERSPELLING', 2: '2 OTHER', 0: '3 ERASE'}


# ---------------------------------------------------------------- 保存・読み込み

def labels_path(out_dir, video):
    return os.path.join(out_dir, os.path.splitext(video)[0] + '.csv')


def load_labels(out_dir, video, n):
    labels = np.zeros(n, dtype=np.int8)
    path = labels_path(out_dir, video)
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                i = int(r['frame'])
                if 0 <= i < n:
                    labels[i] = int(r['label'])
    return labels


def save_labels(out_dir, video, labels, annotator):
    path = labels_path(out_dir, video)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'label', 'annotator'])
        for i, lb in enumerate(labels):
            w.writerow([i, int(lb), annotator])
    os.replace(tmp, path)


def load_progress(out_dir):
    path = os.path.join(out_dir, PROGRESS_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        return {r['video']: r for r in csv.DictReader(f)}


def save_progress(out_dir, progress):
    path = os.path.join(out_dir, PROGRESS_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        w.writeheader()
        for v in sorted(progress, key=natural_key):
            w.writerow({k: progress[v].get(k, '') for k in PROGRESS_FIELDS})
    os.replace(tmp, path)


# ---------------------------------------------------------------- ラベル付けの状態と操作

class FrameLabeler:
    """1本の動画のラベル付け状態。画面表示とは分けてあるので、キー操作を模擬して試験できる。"""

    def __init__(self, video, labels, fps, ask_yes_no, on_change):
        self.video = video
        self.labels = labels
        self.n = len(labels)
        self.fps = fps
        self.ask_yes_no = ask_yes_no
        self.on_change = on_change
        self.pos = 0
        self.brush = None          # None=塗らない, 1, 2, 0(消しゴム)
        self.playing = False
        self.speed_idx = SPEEDS.index(0.5)
        self.undo_stack = []       # (開始, 変更前のラベル列)
        self.result = None
        self.dirty = False

    def paint(self, frames):
        """ブラシを持っていれば、指定のフレームを塗る（取り消し用に元の値を記録）。"""
        if self.brush is None or len(frames) == 0:
            return
        frames = np.asarray(frames)
        before = self.labels[frames].copy()
        if np.all(before == self.brush):
            return
        self.undo_stack.append((frames, before))
        self.labels[frames] = self.brush
        self.dirty = True

    def move(self, delta):
        """delta フレーム移動し、通過したフレーム（移動先を含む）を塗る。"""
        new = int(np.clip(self.pos + delta, 0, self.n - 1))
        if new != self.pos:
            step = 1 if new > self.pos else -1
            self.paint(np.arange(self.pos + step, new + step, step))
        self.pos = new

    def set_brush(self, brush):
        self.brush = brush
        self.paint([self.pos])  # 押した時点のフレームから塗る

    def undo(self):
        if self.undo_stack:
            frames, before = self.undo_stack.pop()
            self.labels[frames] = before
            self.dirty = True

    def next_unlabeled(self):
        idx = np.flatnonzero(self.labels == 0)
        if len(idx):
            later = idx[idx > self.pos]
            self.pos = int(later[0] if len(later) else idx[0])
            self.playing = False

    def handle_key(self, key):
        """キーを1つ処理する。動画を終える操作なら True を返す。"""
        if key < 0:
            return False
        ch = chr(key) if key < 256 else ''
        if ch == '1':
            self.set_brush(1)
        elif ch == '2':
            self.set_brush(2)
        elif ch == '3':
            self.set_brush(0)
        elif ch == '0':
            self.brush = None
        elif ch == ' ':
            self.playing = not self.playing
        elif ch == 'd':
            self.playing = False; self.move(1)
        elif ch == 'a':
            self.playing = False; self.move(-1)
        elif ch in ('D', 'c'):
            self.playing = False; self.move(10)
        elif ch in ('A', 'z'):
            self.playing = False; self.move(-10)
        elif ch == ']':
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif ch == '[':
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif ch == 'u':
            self.undo()
        elif ch == 'g':
            self.next_unlabeled()
        elif ch == 'n':
            self.playing = False
            n0 = int((self.labels == 0).sum())
            q = (f'{self.video}: 未定のフレームが {n0} 個残っています。このまま完了にしますか？'
                 if n0 else f'{self.video}: この動画を完了にしますか？')
            if self.ask_yes_no(q):
                self.result = 'done'
                return True
        elif ch == 'p':
            self.result = 'skip'
            return True
        elif ch == 'q':
            self.result = 'quit'
            return True
        elif ch == 'h':
            print(__doc__)
        return False

    def step_playback(self):
        if self.playing:
            if self.pos >= self.n - 1:
                self.playing = False
            else:
                self.move(1)

    def flush(self):
        """変更があれば保存する。"""
        if self.dirty:
            self.on_change(self.labels)
            self.dirty = False

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]


# ---------------------------------------------------------------- 画面表示

def render(st, frame_jpg):
    img = cv2.imdecode(frame_jpg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    bar_h = 30
    canvas = np.zeros((h + bar_h + 60, w, 3), np.uint8)
    canvas[:h] = img

    # タイムライン: 各フレームのラベルを色で表示
    y0 = h + 5
    cols = (np.arange(w) * st.n // w).clip(0, st.n - 1)
    strip = np.array([COLORS[int(l)] for l in st.labels[cols]], dtype=np.uint8)
    canvas[y0:y0 + bar_h, :] = strip[None, :, :]
    x = int(st.pos / max(st.n - 1, 1) * (w - 1))
    cv2.line(canvas, (x, y0 - 3), (x, y0 + bar_h + 3), (0, 0, 255), 2)

    # 現在のフレームのラベルを枠の色で表示
    cur = int(st.labels[st.pos])
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), COLORS[cur], 6)
    brush_color = COLORS[st.brush] if st.brush is not None else (200, 200, 200)
    cv2.putText(canvas, f'BRUSH: {BRUSH_NAMES[st.brush]}', (8, h + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, brush_color, 2)
    info = (f'frame {st.pos}/{st.n - 1}  t={st.pos / st.fps:6.2f}s  x{st.speed}  '
            f'{"PLAY" if st.playing else "PAUSE"}  label={cur}  '
            f'unlabeled={int((st.labels == 0).sum())}')
    cv2.putText(canvas, info, (8, h + bar_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def ask_yes_no_terminal(question):
    print(f'\n{question} [y/N]: ', end='', flush=True)
    return sys.stdin.readline().strip().lower() in ('y', 'yes')


# ---------------------------------------------------------------- 1本の動画

def label_video(video_dir, video, out_dir, annotator, progress):
    print(f'\n=== {video} ===')
    frames, fps = load_frames(os.path.join(video_dir, video))
    if not frames:
        print('  フレームがありません。スキップします。')
        return 'skip'
    labels = load_labels(out_dir, video, len(frames))

    def on_change(lb, status=None):
        save_labels(out_dir, video, lb, annotator)
        e = progress.get(video, {})
        e.update(video=video, n_frames=len(lb), n_fs=int((lb == 1).sum()), n_other=int((lb == 2).sum()),
                 n_unlabeled=int((lb == 0).sum()), fps=f'{fps:.3f}', annotator=annotator,
                 updated=datetime.datetime.now().isoformat(timespec='seconds'),
                 status=status or 'in_progress')  # 変更したら完了の確認をやり直す
        progress[video] = e
        save_progress(out_dir, progress)

    st = FrameLabeler(video, labels, fps, ask_yes_no_terminal, on_change)
    if (labels != 0).any():
        st.next_unlabeled()  # 再開時は最初の未定のフレームから
    win = 'label frames (h: help)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_save = 0
    while True:
        cv2.imshow(win, render(st, frames[st.pos]))
        delay = max(1, int(1000 / (fps * st.speed))) if st.playing else 30
        key = cv2.waitKey(delay)
        if st.handle_key(key):
            break
        st.step_playback()
        last_save += 1
        if not st.playing or last_save >= 60:  # 再生中は時々、止めたらすぐ保存
            st.flush(); last_save = 0
    cv2.destroyWindow(win)

    on_change(st.labels, status='done' if st.result == 'done' else None)
    print(f'  指文字 {int((st.labels == 1).sum())}, それ以外 {int((st.labels == 2).sum())}, '
          f'未定 {int((st.labels == 0).sum())} フレーム' + ('（完了）' if st.result == 'done' else ''))
    return st.result


def main():
    ap = argparse.ArgumentParser(description='1/2 キーによるフレーム単位のラベル付け',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_video_args(ap)
    ap.add_argument('--out', default='annotations_frames')
    ap.add_argument('--annotator', required=True)
    ap.add_argument('--redo', action='store_true', help='--videos のフォルダ指定でも完了済みの動画を開く')
    args = ap.parse_args()

    items, explicit = resolve_videos(args)
    os.makedirs(args.out, exist_ok=True)
    progress = load_progress(args.out)
    todo = select_todo(items, explicit, progress, args.redo)
    print(f'動画 {len(items)} 本のうち、{len(todo)} 本を順に開きます。')
    for video_dir, v in todo:
        if label_video(video_dir, v, args.out, args.annotator, progress) == 'quit':
            print('保存して終了しました。次回は続きから再開します。')
            break
    else:
        print('すべての動画を処理しました。')


if __name__ == '__main__':
    main()

"""
手話文章動画の指文字区間アノテーションツール。

フォルダ内の動画をファイル名の番号順（1, 2, ..., 10）に1本ずつ開き、
指文字区間の開始・終了フレームと、その区間の文字列（ひらがな）を付けて CSV に保存する。

- フレーム番号は「動画を先頭から順に読んだときの番号（0始まり）」。
  骨格抽出（extract_sequence_features.py）と同じ数え方なので、そのまま対応が取れる。
- 動画は最初に1回だけ先頭から読み込み、縮小した画像をメモリに保持する。
  コマ送り・巻き戻しが正確（mp4 のシークによるフレームずれが起きない）。
- 変更するたびに自動保存する。途中で終了しても、次回は続きから再開できる。
- 1本の動画の指文字を「すべて」付け終えたら n キーで完了にする。
  完了していない動画は、印のないフレームを「指文字ではない」とみなせないため評価に使わない。

使い方:
    python annotate.py --videos Video/文章 --out annotations --annotator yamada

キー操作（動画のウィンドウを選択した状態で押す）:
    space   再生 / 一時停止
    d / a   1フレーム進む / 戻る
    D / A   10フレーム進む / 戻る（c / z でも可）
    ] / [   再生速度を上げる / 下げる（0.1〜1倍。速い指文字はスロー再生で確認）
    s       開始フレームを現在位置にする
    e       終了フレームを現在位置にする
    Enter   区間を確定し、ターミナルで文字列を入力する（k でも可）
    x       現在位置を含む区間を削除する
    u       最後に追加した区間を取り消す
    g       区間の先頭へ移動（押すたびに次の区間へ）
    n       この動画を完了にして次の動画へ（すべての指文字を付けたか確認される）
    p       完了にせず次の動画へ（保存はされる）
    q       保存して終了
    h       この説明を表示
"""

import argparse
import csv
import datetime
import os
import re
import sys

import cv2
import numpy as np

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.m4v', '.mkv')
SPEEDS = [0.1, 0.25, 0.5, 1.0]
PROGRESS_FILE = '_progress.csv'
PROGRESS_FIELDS = ['video', 'status', 'n_intervals', 'n_frames', 'fps', 'annotator', 'updated']
ANNOT_FIELDS = ['video', 'start_frame', 'end_frame', 'text', 'annotator']


def natural_key(name):
    """'2.mp4' < '10.mp4' となるように数字を数値として比較するキー。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def list_videos(folder):
    names = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS)]
    return sorted(names, key=natural_key)


# ---------------------------------------------------------------- 保存・読み込み

def annot_path(out_dir, video):
    return os.path.join(out_dir, os.path.splitext(video)[0] + '.csv')


def load_intervals(out_dir, video):
    path = annot_path(out_dir, video)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        return [{'start': int(r['start_frame']), 'end': int(r['end_frame']), 'text': r['text']}
                for r in csv.DictReader(f)]


def save_intervals(out_dir, video, intervals, annotator):
    path = annot_path(out_dir, video)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ANNOT_FIELDS)
        w.writeheader()
        for iv in sorted(intervals, key=lambda v: v['start']):
            w.writerow({'video': video, 'start_frame': iv['start'], 'end_frame': iv['end'],
                        'text': iv['text'], 'annotator': annotator})
    os.replace(tmp, path)  # 書き込み途中で落ちてもファイルが壊れないように


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
        for video in sorted(progress, key=natural_key):
            w.writerow({k: progress[video].get(k, '') for k in PROGRESS_FIELDS})
    os.replace(tmp, path)


# ---------------------------------------------------------------- 動画の読み込み

def load_frames(path, max_width=960, jpeg_quality=85):
    """動画を先頭から順に読み、縮小した JPEG のリストとして保持する。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'動画を開けません: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1])
        if len(frames) % 300 == 0:
            print(f'\r  読み込み中: {len(frames)} フレーム', end='', flush=True)
    cap.release()
    print(f'\r  読み込み完了: {len(frames)} フレーム, {fps:.2f} fps')
    return frames, fps


# ---------------------------------------------------------------- アノテーションの状態と操作

class Annotator:
    """1本の動画のアノテーション状態。画面表示とは分けてあるので、キー操作を模擬して試験できる。"""

    def __init__(self, video, n_frames, fps, intervals, ask_text, ask_yes_no, on_change):
        self.video = video
        self.n = n_frames
        self.fps = fps
        self.intervals = list(intervals)
        self.ask_text = ask_text        # 文字列を入力させる関数
        self.ask_yes_no = ask_yes_no    # はい/いいえを確認する関数
        self.on_change = on_change      # 変更時に呼ぶ関数（自動保存）
        self.pos = 0
        self.mark_start = None
        self.mark_end = None
        self.playing = False
        self.speed_idx = SPEEDS.index(0.5)
        self.history = []               # 追加した区間（取り消し用）
        self.result = None              # 'done' / 'skip' / 'quit'
        self.message = ''

    # 移動
    def seek(self, delta):
        self.pos = int(np.clip(self.pos + delta, 0, self.n - 1))

    def interval_at(self, frame):
        for i, iv in enumerate(self.intervals):
            if iv['start'] <= frame <= iv['end']:
                return i
        return None

    def confirm_interval(self):
        if self.mark_start is None or self.mark_end is None:
            self.message = '開始(s)と終了(e)の両方を指定してください'
            return
        start, end = sorted((self.mark_start, self.mark_end))
        for iv in self.intervals:
            if not (end < iv['start'] or start > iv['end']):
                self.message = f'既存の区間 {iv["start"]}-{iv["end"]} と重なっています'
                return
        text = self.ask_text(start, end)
        if text is None:
            self.message = '入力を取り消しました'
            return
        iv = {'start': start, 'end': end, 'text': text}
        self.intervals.append(iv)
        self.intervals.sort(key=lambda v: v['start'])
        self.history.append(iv)
        self.mark_start = self.mark_end = None
        self.message = f'追加: {start}-{end} 「{text}」'
        self.on_change(self.intervals)

    def delete_current(self):
        i = self.interval_at(self.pos)
        if i is None:
            self.message = '現在位置に区間がありません'
            return
        iv = self.intervals.pop(i)
        if iv in self.history:
            self.history.remove(iv)
        self.message = f'削除: {iv["start"]}-{iv["end"]} 「{iv["text"]}」'
        self.on_change(self.intervals)

    def undo(self):
        if not self.history:
            self.message = '取り消せる区間がありません'
            return
        iv = self.history.pop()
        self.intervals.remove(iv)
        self.message = f'取り消し: {iv["start"]}-{iv["end"]} 「{iv["text"]}」'
        self.on_change(self.intervals)

    def jump_next_interval(self):
        if not self.intervals:
            self.message = '区間がありません'
            return
        later = [iv['start'] for iv in self.intervals if iv['start'] > self.pos]
        self.pos = later[0] if later else self.intervals[0]['start']
        self.playing = False

    def handle_key(self, key):
        """キーを1つ処理する。動画を終える操作なら True を返す。"""
        if key < 0:
            return False
        ch = chr(key) if key < 256 else ''
        if ch == ' ':
            self.playing = not self.playing
        elif ch == 'd':
            self.playing = False; self.seek(1)
        elif ch == 'a':
            self.playing = False; self.seek(-1)
        elif ch in ('D', 'c'):
            self.playing = False; self.seek(10)
        elif ch in ('A', 'z'):
            self.playing = False; self.seek(-10)
        elif ch == ']':
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif ch == '[':
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif ch == 's':
            self.mark_start = self.pos; self.message = f'開始: {self.pos}'
        elif ch == 'e':
            self.mark_end = self.pos; self.message = f'終了: {self.pos}'
        elif key in (10, 13) or ch == 'k':
            self.playing = False; self.confirm_interval()
        elif ch == 'x':
            self.delete_current()
        elif ch == 'u':
            self.undo()
        elif ch == 'g':
            self.jump_next_interval()
        elif ch == 'n':
            self.playing = False
            if self.ask_yes_no(f'{self.video}: この動画の指文字区間をすべて付けましたか？（{len(self.intervals)} 区間）'):
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
        """再生中なら1フレーム進める。最後まで行ったら止める。"""
        if self.playing:
            if self.pos >= self.n - 1:
                self.playing = False
            else:
                self.pos += 1

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]


# ---------------------------------------------------------------- 画面表示

def render(state, frame_jpg):
    img = cv2.imdecode(frame_jpg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    bar_h = 40
    canvas = np.zeros((h + bar_h + 50, w, 3), np.uint8)
    canvas[:h] = img

    # タイムライン（緑: 確定した区間、黄: 指定中の範囲、赤: 現在位置）
    y0 = h + 5
    cv2.rectangle(canvas, (0, y0), (w - 1, y0 + bar_h), (60, 60, 60), -1)
    def x_of(f):
        return int(f / max(state.n - 1, 1) * (w - 1))
    for i, iv in enumerate(state.intervals):
        cv2.rectangle(canvas, (x_of(iv['start']), y0), (max(x_of(iv['end']), x_of(iv['start']) + 1), y0 + bar_h),
                      (0, 180, 0), -1)
        cv2.putText(canvas, str(i + 1), (x_of(iv['start']) + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    if state.mark_start is not None:
        a = state.mark_start
        b = state.mark_end if state.mark_end is not None else state.pos
        lo, hi = sorted((a, b))
        cv2.rectangle(canvas, (x_of(lo), y0 + bar_h - 10), (max(x_of(hi), x_of(lo) + 1), y0 + bar_h),
                      (0, 220, 255), -1)
    cv2.line(canvas, (x_of(state.pos), y0), (x_of(state.pos), y0 + bar_h), (0, 0, 255), 2)

    # 情報表示（日本語は表示できないので、文字列はターミナルに出す）
    i = state.interval_at(state.pos)
    info = (f'frame {state.pos}/{state.n - 1}  t={state.pos / state.fps:6.2f}s  '
            f'speed x{state.speed}  {"PLAY" if state.playing else "PAUSE"}  '
            f'start={state.mark_start} end={state.mark_end}  '
            f'{"in interval #" + str(i + 1) if i is not None else ""}')
    cv2.putText(canvas, info, (8, h + bar_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if i is not None:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 180, 0), 4)
    return canvas


# ---------------------------------------------------------------- 入力（ターミナル）

def ask_text_terminal(start, end):
    print(f'\n区間 {start}-{end} の文字列をひらがなで入力（空欄で取り消し）: ', end='', flush=True)
    text = sys.stdin.readline().strip()
    return text or None


def ask_yes_no_terminal(question):
    print(f'\n{question} [y/N]: ', end='', flush=True)
    return sys.stdin.readline().strip().lower() in ('y', 'yes')


# ---------------------------------------------------------------- 1本の動画のアノテーション

def annotate_video(video_dir, video, out_dir, annotator, progress):
    print(f'\n=== {video} ===')
    frames, fps = load_frames(os.path.join(video_dir, video))
    if not frames:
        print('  フレームがありません。スキップします。')
        return 'skip'

    def on_change(intervals):
        save_intervals(out_dir, video, intervals, annotator)
        entry = progress.get(video, {})
        entry.update(video=video, status=entry.get('status') or 'in_progress',
                     n_intervals=len(intervals), n_frames=len(frames), fps=f'{fps:.3f}',
                     annotator=annotator, updated=datetime.datetime.now().isoformat(timespec='seconds'))
        if entry['status'] == 'done':
            entry['status'] = 'in_progress'  # 完了後に修正したら、もう一度完了の確認をしてもらう
        progress[video] = entry
        save_progress(out_dir, progress)

    state = Annotator(video, len(frames), fps, load_intervals(out_dir, video),
                      ask_text_terminal, ask_yes_no_terminal, on_change)
    for i, iv in enumerate(state.intervals):
        print(f'  既存の区間 #{i + 1}: {iv["start"]}-{iv["end"]} 「{iv["text"]}」')

    win = 'annotate (h: help)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_msg = ''
    while True:
        cv2.imshow(win, render(state, frames[state.pos]))
        if state.message and state.message != last_msg:
            print('  ' + state.message)
            last_msg = state.message
        delay = max(1, int(1000 / (fps * state.speed))) if state.playing else 30
        key = cv2.waitKey(delay)
        if state.handle_key(key):
            break
        state.step_playback()
    cv2.destroyWindow(win)

    # 状態を保存（区間が0本でも、完了なら「指文字なし」として記録する）
    on_change(state.intervals)
    if state.result == 'done':
        progress[video]['status'] = 'done'
        save_progress(out_dir, progress)
        print(f'  完了: {len(state.intervals)} 区間')
    return state.result


def main():
    ap = argparse.ArgumentParser(description='指文字区間のアノテーションツール',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--videos', required=True, help='動画のフォルダ')
    ap.add_argument('--out', default='annotations', help='CSV の保存先フォルダ')
    ap.add_argument('--annotator', required=True, help='アノテーションした人の名前（一致度の評価に使う）')
    ap.add_argument('--redo', action='store_true', help='完了済みの動画も開く')
    ap.add_argument('--only', help='この名前の動画だけを開く')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    progress = load_progress(args.out)
    videos = list_videos(args.videos)
    if args.only:
        videos = [v for v in videos if v == args.only]
    todo = [v for v in videos if args.redo or progress.get(v, {}).get('status') != 'done']
    print(f'動画 {len(videos)} 本のうち、未完了 {len(todo)} 本を順に開きます。')

    for video in todo:
        result = annotate_video(args.videos, video, args.out, args.annotator, progress)
        if result == 'quit':
            print('保存して終了しました。次回は続きから再開します。')
            break
    else:
        print('すべての動画を処理しました。python validate.py で検査してください。')


if __name__ == '__main__':
    main()

"""
アノテーション CSV を検査し、完了した動画の区間を1つの CSV にまとめる。

検査する内容:
- 開始 <= 終了、フレーム番号が動画の範囲内か
- 区間どうしが重なっていないか
- 文字列がひらがな（小書き文字・長音記号を含む）だけか
- 未完了（in_progress）の動画がないか

使い方:
    python validate.py --out annotations
    python validate.py --out annotations --export annotations_all.csv
"""

import argparse
import csv
import os
import re
import sys

from annotate import PROGRESS_FILE, load_intervals, load_progress, natural_key

HIRAGANA = re.compile(r'^[ぁ-ゖー]+$')  # ぁ〜ゖ と ー


def check_video(video, intervals, n_frames):
    problems = []
    for iv in intervals:
        s, e, t = iv['start'], iv['end'], iv['text']
        if s > e:
            problems.append(f'開始 > 終了: {s}-{e}')
        if s < 0 or (n_frames is not None and e >= n_frames):
            problems.append(f'範囲外のフレーム: {s}-{e}（動画は {n_frames} フレーム）')
        if not HIRAGANA.match(t or ''):
            problems.append(f'ひらがな以外を含む文字列: 「{t}」({s}-{e})')
    ivs = sorted(intervals, key=lambda v: v['start'])
    for a, b in zip(ivs, ivs[1:]):
        if b['start'] <= a['end']:
            problems.append(f'区間が重なっている: {a["start"]}-{a["end"]} と {b["start"]}-{b["end"]}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='annotations', help='アノテーション CSV のフォルダ')
    ap.add_argument('--export', help='完了した動画の区間をまとめて書き出す CSV')
    args = ap.parse_args()

    progress = load_progress(args.out)
    csv_videos = {f for f in os.listdir(args.out) if f.endswith('.csv') and f != PROGRESS_FILE}
    if not progress and not csv_videos:
        print('アノテーションがありません。')
        return 0

    n_error = 0
    rows = []
    stats = {'done': 0, 'in_progress': 0}
    for video in sorted(progress, key=natural_key):
        p = progress[video]
        status = p.get('status', '')
        stats[status] = stats.get(status, 0) + 1
        n_frames = int(p['n_frames']) if p.get('n_frames') else None
        intervals = load_intervals(args.out, video)
        problems = check_video(video, intervals, n_frames)
        if problems:
            n_error += len(problems)
            print(f'[NG] {video}')
            for msg in problems:
                print(f'     {msg}')
        if status != 'done':
            print(f'[未完了] {video}（{len(intervals)} 区間）: 評価には使えません')
            continue
        fps = float(p['fps']) if p.get('fps') else None
        for iv in intervals:
            rows.append({'video': video, 'start_frame': iv['start'], 'end_frame': iv['end'],
                         'text': iv['text'], 'n_chars': len(iv['text']),
                         'duration_s': f'{(iv["end"] - iv["start"] + 1) / fps:.3f}' if fps else '',
                         'chars_per_s': f'{len(iv["text"]) * fps / (iv["end"] - iv["start"] + 1):.2f}' if fps else '',
                         'fps': p.get('fps', ''), 'annotator': p.get('annotator', '')})

    print(f'\n完了 {stats.get("done", 0)} 本 / 未完了 {stats.get("in_progress", 0)} 本、'
          f'完了動画の指文字区間 {len(rows)} 個、問題 {n_error} 件')
    if rows:
        speeds = sorted(float(r['chars_per_s']) for r in rows if r['chars_per_s'])
        if speeds:
            print(f'表現速度（文字/秒）: 最小 {speeds[0]:.2f}, 中央値 {speeds[len(speeds) // 2]:.2f}, '
                  f'最大 {speeds[-1]:.2f}')

    if args.export:
        with open(args.export, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['video'])
            w.writeheader()
            w.writerows(rows)
        print(f'書き出し: {args.export}')
    return 1 if n_error else 0


if __name__ == '__main__':
    sys.exit(main())

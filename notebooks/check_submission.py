"""Check that a Kaggle csv has the expected format before uploading it.

    python notebooks/check_submission.py out/submissions/ensemble_yolo_frcnn.csv --test test_images.csv

Format: image_id, PredictionString = "class score x y w h ..." with class in 0..7, score in (0, 1],
w and h > 0, at most 500 boxes per image, one row for every test image.
"""
import argparse
import sys

import pandas as pd

NUM_CLASSES = 8
MAX_DET = 500


def check(sub, test_ids=None):
    """Return a list of problems (empty if the file is fine)."""
    problems = []
    if list(sub.columns) != ['image_id', 'PredictionString']:
        return [f'columns are {list(sub.columns)}, expected image_id, PredictionString']
    if not sub.image_id.is_unique:
        problems.append(f'{sub.image_id.duplicated().sum()} duplicated image_id')
    if test_ids is not None:
        missing, extra = set(test_ids) - set(sub.image_id), set(sub.image_id) - set(test_ids)
        if missing:
            problems.append(f'{len(missing)} test images missing, e.g. {sorted(missing)[:3]}')
        if extra:
            problems.append(f'{len(extra)} unknown image_id, e.g. {sorted(extra)[:3]}')
    for r in sub.itertuples():
        parts = str(r.PredictionString).split()
        if len(parts) % 6:
            problems.append(f'{r.image_id}: {len(parts)} values, not a multiple of 6'); continue
        if len(parts) // 6 > MAX_DET:
            problems.append(f'{r.image_id}: {len(parts) // 6} boxes, limit {MAX_DET}')
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            problems.append(f'{r.image_id}: non-numeric value'); continue
        for i in range(0, len(vals), 6):
            c, s, _, _, w, h = vals[i:i + 6]
            if c != int(c) or not 0 <= c < NUM_CLASSES:
                problems.append(f'{r.image_id}: class {c}'); break
            if not 0 < s <= 1:
                problems.append(f'{r.image_id}: score {s}'); break
            if w <= 0 or h <= 0:
                problems.append(f'{r.image_id}: box with w={w}, h={h}'); break
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--test', help='test_images.csv, to check that every test image has a row')
    args = ap.parse_args()
    test_ids = pd.read_csv(args.test).image_id.tolist() if args.test else None
    bad = 0
    for path in args.csv:
        sub = pd.read_csv(path, keep_default_na=False)
        problems = check(sub, test_ids)
        boxes = sub.PredictionString.astype(str).str.split().map(len).sum() // 6 if 'PredictionString' in sub else 0
        print(f'{path}: {"OK" if not problems else "FAILED"}, {len(sub)} rows, {boxes / max(len(sub), 1):.1f} boxes per image')
        for p in problems[:20]:
            print('   ', p)
        bad += bool(problems)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

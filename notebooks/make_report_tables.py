"""Build the markdown results tables for report.md from the saved run results.

Run from the directory that contains out/:

    python notebooks/make_report_tables.py
    python notebooks/make_report_tables.py --kaggle kaggle_scores.csv

out/results_*.csv are written by the notebook (one row per run). The optional Kaggle file is filled in by hand
from the leaderboard and has the columns: file, public, private (private may stay empty until it is revealed).
Runs whose csv is not in the Kaggle file get "—", runs missing from the results get "not saved".
"""
import argparse
import glob
import os

import pandas as pd

FAMILY_ORDER = ['two-stage', 'one-stage anchor', 'anchor-free', 'query', 'pipeline', 'inference', 'ensemble', 'cv']
METRICS = ['mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']


def load_results(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, 'results_*.csv')))
    if not files:
        raise SystemExit(f'no results_*.csv in {out_dir}')
    # same rule as load_results() in the notebook: a later file overrides an earlier one
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates('name', keep='last')


def csv_name(run):
    """Name of the leaderboard csv that the notebook writes for a run ('final_yolo+tiles_soft' -> 'final_yolo_tiles_soft.csv')."""
    return run.replace('+full', '').replace('+', '_') + '.csv'


def fmt(v, digits=4):
    return '' if pd.isna(v) else f'{v:.{digits}f}'


def to_markdown(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='out', help='directory with results_*.csv')
    ap.add_argument('--kaggle', help='csv with columns file, public, private')
    ap.add_argument('--save', help='also write the tables to this markdown file')
    args = ap.parse_args()

    res = load_results(args.out)
    main_runs = res[res.family != 'cv'].copy()
    main_runs['order'] = main_runs.family.map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(len(FAMILY_ORDER))
    main_runs = main_runs.sort_values(['order', 'mAP'], ascending=[True, False])
    main_runs['file'] = main_runs.name.map(csv_name)

    if args.kaggle:
        kg = pd.read_csv(args.kaggle)
        main_runs = main_runs.merge(kg, on='file', how='left')
    for c in ('public', 'private'):
        if c not in main_runs:
            main_runs[c] = float('nan')

    table = main_runs.assign(**{m: main_runs[m].map(fmt) for m in METRICS},
                             fps=main_runs.fps.map(lambda v: fmt(v, 1)),
                             public=main_runs.public.map(lambda v: '—' if pd.isna(v) else fmt(v, 5)),
                             private=main_runs.private.map(lambda v: '' if pd.isna(v) else fmt(v, 5)),
                             run=main_runs.name.map(lambda n: f'`{n}`'))
    parts = ['### Validation and leaderboard', '',
             to_markdown(table, ['run', 'family', 'mAP', 'AP50', 'AP_small', 'fps', 'public', 'private', 'note'])]

    cv = res[res.family == 'cv']
    if len(cv):
        parts += ['', '### Sequence-level cross-validation (YOLO11s)', '',
                  to_markdown(cv.assign(run=cv.name.map(lambda n: f'`{n}`'), mAP=cv.mAP.map(fmt), AP_small=cv.AP_small.map(fmt)),
                              ['run', 'mAP', 'AP_small']),
                  '', f'mean mAP {cv.mAP.mean():.4f} ± {cv.mAP.std():.4f} over {len(cv)} folds']

    text = '\n'.join(parts)
    print(text)
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()

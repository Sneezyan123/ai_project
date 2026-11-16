"""Recover available artifacts, without inventing validation or training metrics.

Run from any directory: python ai_project/notebooks/recover_outputs.py
The default data directory is the parent of the repository, containing top5_csv/.
Recovered result filenames deliberately differ from training result filenames.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ['mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'fps']
FAMILIES = {
    'frcnn_base': ('rcnn', 'two-stage'),
    'final_yolo': ('yolo', 'pipeline'),
    'final_yolo_tiles_nms': ('yolo', 'inference'),
    'final_yolo_tiles_soft': ('yolo', 'inference'),
    'ensemble_yolo_frcnn': ('ensemble', 'ensemble'),
}


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inspect_submission(path, expected):
    counts = Counter()
    ids = []
    boxes = invalid = above_one = 0
    max_score = None
    max_boxes = 0
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_id', 'PredictionString']:
            raise ValueError(f'Unexpected submission columns in {path}')
        for row in reader:
            ids.append(row['image_id'])
            values = row['PredictionString'].split()
            if len(values) % 6:
                raise ValueError(f'Malformed predictions: {path}, {row["image_id"]}')
            n = len(values) // 6
            boxes += n
            max_boxes = max(max_boxes, n)
            for i in range(0, len(values), 6):
                c, score, x, y, w, h = map(float, values[i:i + 6])
                finite = all(np.isfinite(v) for v in (c, score, x, y, w, h))
                valid_class = np.isfinite(c) and c.is_integer() and 0 <= c < 8
                invalid += int(not (finite and valid_class and 0 < score <= 1 and w > 0 and h > 0))
                above_one += int(score > 1)
                max_score = score if max_score is None else max(max_score, score)
                if valid_class:
                    counts[int(c)] += 1
    return dict(file=path.name, frames=len(ids), boxes=boxes,
                boxes_per_frame=boxes / max(1, len(ids)), max_boxes=max_boxes,
                missing_ids=len(set(expected) - set(ids)), extra_ids=len(set(ids) - set(expected)),
                duplicate_ids=len(ids) - len(set(ids)), ids_in_expected_order=ids == expected,
                invalid_predictions=invalid, scores_above_one=above_one, max_score=max_score,
                **{f'class_{c}_boxes': counts[c] for c in range(8)})


def recover(data, out, scores_path):
    out.mkdir(parents=True, exist_ok=True)
    submissions = out / 'submissions'
    submissions.mkdir(exist_ok=True)
    with (data / 'test_images.csv').open(newline='', encoding='utf-8-sig') as f:
        expected = [row['image_id'] for row in csv.DictReader(f)]
    with scores_path.open(newline='', encoding='utf-8-sig') as f:
        scores = {row['file']: row for row in csv.DictReader(f)}
    grouped = defaultdict(list)
    checks, sources = [], []
    for name, (part, family) in FAMILIES.items():
        src = data / 'top5_csv' / f'{name}.csv'
        if not src.exists():
            continue
        check = inspect_submission(src, expected)
        target = submissions / src.name
        if target.exists() and digest(src) != digest(target):
            raise FileExistsError(f'Refusing to replace a different submission: {target}')
        if not target.exists():
            shutil.copy2(src, target)
        sha = digest(src)
        assert sha == digest(target)
        sources.append(dict(path=str(src.resolve()), sha256=sha))
        checks.append(check)
        score = scores.get(src.name, {})
        grouped[part].append(dict(
            name=name.replace('_tiles_', '+tiles_'), family=family,
            **{metric: '' for metric in METRICS}, gpu='',
            note='Recovered test submission only; validation and training logs unavailable.',
            file=src.name, public=score.get('public', ''), private=score.get('private', ''),
            selected=score.get('selected', ''), validation_status='not_recovered',
            leaderboard_status='manually_recorded_not_independently_verified',
            leaderboard_source=str(scores_path.resolve()), artifact_sha256=sha))
    fields = ['name', 'family', *METRICS, 'gpu', 'note', 'file', 'public', 'private',
              'selected', 'validation_status', 'leaderboard_status', 'leaderboard_source', 'artifact_sha256']
    for part, rows in grouped.items():
        write_csv(out / f'results_{part}_recovered.csv', rows, fields)
    if checks:
        write_csv(out / 'submission_checks.csv', checks, list(checks[0]))

    summaries, classes, sequences = [], [], []
    split_videos = {}
    for split in ('train', 'val'):
        path = data / split / 'annotations_coco.json'
        with path.open(encoding='utf-8') as f:
            annotations = json.load(f)
        images = {im['id']: im for im in annotations['images']}
        split_videos[split] = {im['video'] for im in images.values()}
        per_image = Counter()
        class_counts = Counter()
        sizes = []
        ignored = 0
        for a in annotations['annotations']:
            if a.get('iscrowd', 0):
                ignored += 1
                continue
            im = images[a['image_id']]
            _, _, w, h = a['bbox']
            per_image[im['id']] += 1
            class_counts[a['category_id']] += 1
            scale = min(800 / min(im['width'], im['height']), 1333 / max(im['width'], im['height']))
            if w * scale >= 1 and h * scale >= 1:
                sizes.append((w * h) ** .5 * scale)
        summaries.append(dict(split=split, frames=len(images), sequences=len(split_videos[split]),
                              target_boxes=sum(class_counts.values()), ignored_annotations=ignored,
                              size_eligible_boxes=len(sizes), median_sqrt_area_resized_px=float(np.median(sizes)),
                              fraction_sqrt_area_below_32_resized=float(np.mean(np.array(sizes) < 32)),
                              resize_short_side=800, resize_max_long_side=1333, source=str(path.resolve())))
        for c in annotations['categories']:
            classes.append(dict(split=split, category_id=c['id'], name=c['name'], boxes=class_counts[c['id']]))
        by_video = defaultdict(list)
        for im in images.values():
            by_video[im['video']].append(per_image[im['id']])
        for video, counts in sorted(by_video.items()):
            sequences.append(dict(split=split, video=video, frames=len(counts), boxes=sum(counts),
                                  objects_per_frame=float(np.mean(counts)), max_objects_per_frame=max(counts)))
        del annotations
    for filename, rows in [('data_summary.csv', summaries), ('class_counts.csv', classes), ('sequence_summary.csv', sequences)]:
        write_csv(out / filename, rows, list(rows[0]))
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Recovery of existing test submissions and recomputation of dataset statistics; not a training run.',
        validation_metrics_recovered=False, training_performed=False,
        missing=['model checkpoints', 'validation predictions', 'per-epoch logs', 'fold metrics', 'inference timings'],
        leaderboard_source=dict(path=str(scores_path.resolve()), sha256=digest(scores_path), independently_verified=False),
        train_val_overlapping_video_ids=sorted(split_videos['train'] & split_videos['val']),
        submission_sources=sources,
        warning='Do not treat recovered CSVs as evidence that training or validation experiments were completed.')
    (out / 'recovery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Recovered {len(checks)} submissions into {out.resolve()}')
    print('Validation metrics and fps are blank. Leaderboard scores retain their manual-source status.')
    for row in checks:
        print(f'{row["file"]}: {row["frames"]} frames, {row["boxes"]} boxes, {row["invalid_predictions"]} invalid predictions')


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=repo.parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    recover(args.data, args.out or args.data / 'out', repo / 'notebooks' / 'kaggle_scores.csv')

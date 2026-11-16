"""Weighted boxes fusion of two submissions (best YOLO + Faster R-CNN).

Run from the directory that contains out/, val/ and test_images.csv:

    python notebooks/ensemble.py                    # val check + test csv
    python notebooks/ensemble.py --skip-val         # only the test csv
    python notebooks/ensemble.py --weights 3 1 --iou 0.6

WBF idea: Solovyev et al., "Weighted boxes fusion", 2021 (arXiv:1910.13302).
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

MAX_DET = 500
EMPTY = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))


def iou_one(b, F):
    """IoU of one xyxy box with every row of F."""
    lt = np.maximum(b[:2], F[:, :2])
    rb = np.minimum(b[2:], F[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(b) + area(F) - inter + 1e-9)


def wbf_class(boxes, scores, weights, iou_thr, total_weight):
    """Fuse boxes of one class. Each box joins the fused cluster it overlaps most (IoU > iou_thr)
    or starts a new one; cluster box = score*weight weighted mean of its members."""
    order = np.argsort(-scores)
    n = len(order)
    fused, box_sum = np.zeros((n, 4)), np.zeros((n, 4))
    w_sum = np.zeros(n)
    k = 0
    for i in order:
        c = scores[i] * weights[i]
        j = -1
        if k:
            ious = iou_one(boxes[i], fused[:k])
            m = int(ious.argmax())
            if ious[m] > iou_thr:
                j = m
        if j < 0:
            j, k = k, k + 1
        box_sum[j] += boxes[i] * c
        w_sum[j] += c
        fused[j] = box_sum[j] / w_sum[j]
    # a box found by only one model keeps a share of its score proportional to that model's weight
    return fused[:k], w_sum[:k] / total_weight


def merge(models, weights, iou_thr, max_det=MAX_DET):
    """models: list of (boxes xyxy, scores, labels), one per model, in the same order as weights."""
    if len(models) != len(weights):
        raise ValueError('one weight per model is needed')
    boxes = np.concatenate([m[0] for m in models]).reshape(-1, 4)
    scores = np.concatenate([m[1] for m in models])
    labels = np.concatenate([m[2] for m in models]).astype(int)
    per_box_w = np.concatenate([np.full(len(m[1]), w, dtype=float) for m, w in zip(models, weights)])
    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        ids = labels == c
        b, s = wbf_class(boxes[ids], scores[ids], per_box_w[ids], iou_thr, float(sum(weights)))
        out_b.append(b); out_s.append(s); out_l.append(np.full(len(s), c))
    if not out_b:
        return EMPTY
    b, s, l = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)
    top = np.argsort(-s)[:max_det]
    return b[top], s[top], l[top]


def load_preds(path):
    """COCO-style val predictions -> {image_id: (boxes xyxy, scores, labels 0-based)}."""
    per = {}
    with open(path) as f:
        for p in json.load(f):
            x, y, w, h = p['bbox']
            per.setdefault(p['image_id'], []).append([x, y, x + w, y + h, p['score'], p['category_id'] - 1])
    return {k: (np.array(v)[:, :4], np.array(v)[:, 4], np.array(v)[:, 5].astype(int)) for k, v in per.items()}


def read_submission(path):
    """Kaggle csv (class score x y w h ...) -> {image_id: (boxes xyxy, scores, labels)}."""
    d = pd.read_csv(path, keep_default_na=False)
    out = {}
    for r in d.itertuples():
        v = np.array(r.PredictionString.split(), dtype=float).reshape(-1, 6) if r.PredictionString else np.zeros((0, 6))
        b = v[:, 2:6].copy()
        b[:, 2:] += b[:, :2]
        out[r.image_id] = (b, v[:, 1], v[:, 0].astype(int))
    return out


def val_check(args):
    """Score the fusion on val with the same YOLO variant as in the test csv when its val predictions exist."""
    from faster_coco_eval import COCO, COCOeval_faster  # only needed for this check

    if not os.path.exists(args.val_a):
        # runs made before the notebook saved tiled val predictions only have the full-frame file
        print(f'{args.val_a} not found, falling back to {args.val_a_fallback}: val score is for full-frame YOLO, not the test variant')
        args.val_a = args.val_a_fallback

    with open(os.path.join(args.data, 'val', 'annotations_coco.json')) as f:
        val = json.load(f)
    off = 10_000_000  # same id offset as in the notebook
    imgs = [dict(i, id=i['id'] + off) for i in val['images'] if (i['frame_id'] - 1) % args.val_step == 0]
    ids = {i['id'] for i in imgs}
    anns = [dict(a, id=a['id'] + off, image_id=a['image_id'] + off) for a in val['annotations'] if a['image_id'] + off in ids]
    gt = {'images': imgs, 'categories': val['categories'], 'annotations': anns}
    A, B = load_preds(args.val_a), load_preds(args.val_b)
    preds = []
    for i in sorted(ids):
        b, s, l = merge([A.get(i, EMPTY), B.get(i, EMPTY)], args.weights, args.iou)
        for (x1, y1, x2, y2), sc, c in zip(b.tolist(), s.tolist(), l.tolist()):
            preds.append({'image_id': i, 'category_id': int(c) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})
    coco = COCO(gt)
    ev = COCOeval_faster(coco, coco.loadRes(preds), 'bbox')
    ev.params.maxDets = [1, 100, MAX_DET]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print('ensemble val mAP %.4f AP50 %.4f AP_small %.4f' % (ev.stats[0], ev.stats[1], ev.stats[3]))
    # same columns as the notebook's results files, so make_report_tables.py picks the ensemble up
    s = ev.stats
    row = dict(name=os.path.splitext(os.path.basename(args.out))[0], family='ensemble',
               mAP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AP_large=s[5], fps=float('nan'), gpu='',
               note=f'WBF weights {args.weights[0]:g}:{args.weights[1]:g}, iou {args.iou:g}, val A={os.path.basename(args.val_a)}')
    pd.DataFrame([row]).round(4).to_csv(os.path.join(args.results_dir, 'results_ensemble.csv'), index=False)


def write_test(args):
    X, Y = read_submission(args.test_a), read_submission(args.test_b)
    test = pd.read_csv(os.path.join(args.data, 'test_images.csv'))
    assert set(test.image_id) == set(X) == set(Y), 'both submissions must cover exactly the test images'
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'PredictionString'])
        for image_id in test.image_id:
            b, s, l = merge([X[image_id], Y[image_id]], args.weights, args.iou)
            parts = [f'{c} {sc:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}'
                     for (x1, y1, x2, y2), sc, c in zip(b.tolist(), s.tolist(), l.tolist()) if x2 - x1 >= .5 and y2 - y1 >= .5]
            w.writerow([image_id, ' '.join(parts)])

    check = pd.read_csv(args.out, keep_default_na=False)
    fields = check.PredictionString.str.split()
    assert len(check) == len(test) and check.image_id.is_unique
    assert (fields.map(len) % 6 == 0).all() and fields.map(len).max() <= 6 * MAX_DET
    assert all(int(p[i]) in range(8) and float(p[i + 4]) > 0 and float(p[i + 5]) > 0 for p in fields for i in range(0, len(p), 6))
    print('saved', args.out, len(check), 'rows, avg boxes', round(float(fields.map(len).mean() / 6), 1))


def main():
    subs = 'out/submissions/'
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='.', help='directory with val/ and test_images.csv')
    ap.add_argument('--weights', type=float, nargs=2, default=(2.0, 1.0), help='weights of model A and model B')
    ap.add_argument('--iou', type=float, default=0.55, help='IoU threshold for joining a cluster')
    ap.add_argument('--val-step', type=int, default=5, help='every n-th val frame, as VAL_STEP in the notebook')
    ap.add_argument('--val-a', default='out/runs/final_yolo/val_preds_tiles_soft.json')
    ap.add_argument('--val-a-fallback', default='out/runs/final_yolo/val_preds.json')
    ap.add_argument('--val-b', default='out/runs/frcnn_base/val_preds.json')
    ap.add_argument('--test-a', default=subs + 'final_yolo_tiles_soft.csv')
    ap.add_argument('--test-b', default=subs + 'frcnn_base.csv')
    ap.add_argument('--out', default=subs + 'ensemble_yolo_frcnn.csv')
    ap.add_argument('--results-dir', default='out', help='where results_ensemble.csv with the val score is written')
    ap.add_argument('--skip-val', action='store_true', help='do not run the val check')
    args = ap.parse_args()
    if not args.skip_val:
        val_check(args)
    write_test(args)


if __name__ == '__main__':
    main()

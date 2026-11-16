"""Error analysis of saved validation predictions, in the spirit of TIDE (Bolya et al., ECCV 2020).

Runs on CPU from the files the notebook already saved, no retraining. From the directory with out/ and val/:

    python notebooks/error_analysis.py                                  # frcnn_base and final_yolo
    python notebooks/error_analysis.py --model frcnn=out/runs/frcnn_base/val_preds.json --draw 3

Every prediction is given one type, checked in this order (fg = 0.5, bg = 0.1 IoU):
    TP    IoU >= fg with a not yet matched ground truth box of the same class
    Dupl  IoU >= fg with a same-class box that an earlier, higher-scored prediction already took
    Cls   IoU >= fg with a box of another class
    Loc   same class, bg <= IoU < fg
    Both  another class, bg <= IoU < fg
    Ign   centre inside an ignored (iscrowd) region, not an error
    Bkg   IoU < bg with every box
Ground truth boxes that no prediction took are Missed.

Counts and the tables use predictions with score >= --conf (as frame_errors in the notebook). dAP uses all
predictions: each error type is fixed by an oracle on its own (Cls -> relabel, Loc -> take the ground truth box,
Dupl/Bkg/Both -> drop, Missed -> add the box with the top score) and AP50 is measured again. Our fixes are a
simplified version of the TIDE ones, so the numbers are close but not identical to the official tool.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

CLASSES = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
TYPES = ['TP', 'Dupl', 'Cls', 'Loc', 'Both', 'Ign', 'Bkg']
ERRORS = ['Cls', 'Loc', 'Both', 'Dupl', 'Bkg', 'Missed']
COLORS = {'TP': 'lime', 'Dupl': 'yellow', 'Cls': 'orange', 'Loc': 'cyan', 'Both': 'magenta', 'Bkg': 'red', 'Missed': 'blue'}
VAL_OFFSET = 10_000_000  # val ids get this offset in the notebook


def iou_matrix(a, b):
    """IoU between every xyxy box of a and every xyxy box of b."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def md(df, index=True):
    """Markdown table without the tabulate dependency."""
    d = df.reset_index() if index else df
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, r)) + ' |' for r in d.itertuples(index=False)]
    return '\n'.join([head, '|' + '---|' * len(d.columns)] + rows)


def size_name(area):
    return 'small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large'


def classify_image(p_boxes, p_labels, g_boxes, g_labels, crowd, fg=.5, bg=.1):
    """Types for the predictions of one frame (must be sorted by score, highest first).
    Returns (types, gt index each prediction refers to or -1, matched flag of every ground truth box)."""
    ious = iou_matrix(p_boxes, g_boxes)
    matched = np.zeros(len(g_boxes), bool)
    types, refs = [], []
    for i in range(len(p_boxes)):
        row = ious[i] if len(g_boxes) else np.zeros(0)
        same = g_labels == p_labels[i]
        free = row * (same & ~matched)
        taken = row * (same & matched)
        other = row * ~same
        if len(row) and free.max() >= fg:
            j = int(free.argmax()); matched[j] = True; t = 'TP'
        elif len(row) and taken.max() >= fg:
            j = int(taken.argmax()); t = 'Dupl'
        elif len(row) and other.max() >= fg:
            j = int(other.argmax()); t = 'Cls'
        elif len(row) and (row * same).max() >= bg:
            j = int((row * same).argmax()); t = 'Loc'
        elif len(row) and other.max() >= bg:
            j = int(other.argmax()); t = 'Both'
        else:
            cx, cy = (p_boxes[i, 0] + p_boxes[i, 2]) / 2, (p_boxes[i, 1] + p_boxes[i, 3]) / 2
            inside = any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in crowd)
            j, t = -1, 'Ign' if inside else 'Bkg'
        types.append(t); refs.append(j)
    return types, np.array(refs, dtype=int), matched


def load_gt(data, val_step):
    with open(os.path.join(data, 'val', 'annotations_coco.json')) as f:
        val = json.load(f)
    imgs = [dict(i, id=i['id'] + VAL_OFFSET) for i in val['images'] if (i['frame_id'] - 1) % val_step == 0]
    ids = {i['id'] for i in imgs}
    anns = [dict(a, id=a['id'] + VAL_OFFSET, image_id=a['image_id'] + VAL_OFFSET) for a in val['annotations']
            if a['image_id'] + VAL_OFFSET in ids]
    return {'images': imgs, 'categories': val['categories'], 'annotations': anns}


def xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def analyse(preds, gt, conf):
    """One row per prediction and one per missed ground truth box, for every score >= 0 (filter by conf later)."""
    by_img = {}
    for a in gt['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    p_by_img = {}
    for p in preds:
        p_by_img.setdefault(p['image_id'], []).append(p)
    video = {im['id']: im.get('video', '') for im in gt['images']}
    rows = []
    for im in gt['images']:
        items = [a for a in by_img.get(im['id'], []) if not a.get('iscrowd')]
        crowd = [xyxy(a['bbox']) for a in by_img.get(im['id'], []) if a.get('iscrowd')]
        g_boxes = np.array([xyxy(a['bbox']) for a in items]).reshape(-1, 4)
        g_labels = np.array([a['category_id'] for a in items], dtype=int)
        ps = sorted(p_by_img.get(im['id'], []), key=lambda p: -p['score'])
        p_boxes = np.array([xyxy(p['bbox']) for p in ps]).reshape(-1, 4)
        p_labels = np.array([p['category_id'] for p in ps], dtype=int)
        # matching at the conf threshold decides what counts as missed in the tables
        keep = np.array([p['score'] >= conf for p in ps], dtype=bool)
        types, refs, matched_all = classify_image(p_boxes, p_labels, g_boxes, g_labels, crowd)
        _, _, matched_conf = classify_image(p_boxes[keep], p_labels[keep], g_boxes, g_labels, crowd)
        for p, b, t, j in zip(ps, p_boxes, types, refs):
            rows.append(dict(image_id=im['id'], video=video[im['id']], kind=t, score=p['score'], cat=p['category_id'],
                             gt_cat=int(g_labels[j]) if j >= 0 else 0, gt=int(j),
                             size=size_name((b[2] - b[0]) * (b[3] - b[1])), box=b.tolist()))
        for j in range(len(items)):
            if not matched_conf[j] or not matched_all[j]:
                rows.append(dict(image_id=im['id'], video=video[im['id']], kind='Missed', score=np.nan,
                                 cat=int(g_labels[j]), gt_cat=int(g_labels[j]), gt=j, size=size_name(items[j]['area']),
                                 box=g_boxes[j].tolist(), missed_conf=not matched_conf[j], missed_all=not matched_all[j]))
    df = pd.DataFrame(rows)
    for c in ('missed_conf', 'missed_all'):
        if c not in df:
            df[c] = False
        df[c] = df[c].eq(True)
    return df


def apply_fix(df, kind, gt):
    """Predictions (COCO json list) with one error type fixed by an oracle."""
    boxes = {}
    for a in gt['annotations']:
        boxes.setdefault(a['image_id'], []).append(a)
    p = df[df.kind != 'Missed']
    out = []
    for r in p.itertuples():
        box, cat = r.box, r.cat
        if r.kind == kind and kind in ('Dupl', 'Bkg', 'Both'):
            continue
        if r.kind == kind == 'Cls':
            cat = r.gt_cat
        if r.kind == kind == 'Loc':
            items = [a for a in boxes[r.image_id] if not a.get('iscrowd')]
            box = xyxy(items[r.gt]['bbox'])
        out.append({'image_id': r.image_id, 'category_id': int(cat), 'score': float(r.score),
                    'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]]})
    if kind == 'Missed':
        top = float(p.score.max()) if len(p) else 1.0
        for r in df[(df.kind == 'Missed') & df.missed_all].itertuples():
            b = r.box
            out.append({'image_id': r.image_id, 'category_id': int(r.cat), 'score': top, 'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    return out


def coco_ap(preds, gt):
    from faster_coco_eval import COCO, COCOeval_faster  # only needed for dAP
    import contextlib, io
    coco = COCO(gt)
    if not preds:
        return 0.0, 0.0
    ev = COCOeval_faster(coco, coco.loadRes(preds), 'bbox')
    ev.params.maxDets = [1, 100, 500]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1]


def delta_ap(df, gt):
    base = apply_fix(df, None, gt)
    m0, a0 = coco_ap(base, gt)
    rows = [dict(fix='none', mAP=m0, AP50=a0, dAP50=0.0)]
    for k in ERRORS:
        m, a = coco_ap(apply_fix(df, k, gt), gt)
        rows.append(dict(fix=k, mAP=m, AP50=a, dAP50=a - a0))
    return pd.DataFrame(rows).round(4)


def tables(df, conf):
    d = df[(df.kind != 'Missed') & (df.score >= conf) | (df.kind == 'Missed') & df.missed_conf]
    d = d[d.kind != 'Ign']
    by_size = d.groupby(['kind', 'size']).size().unstack(fill_value=0).reindex(['TP'] + ERRORS, fill_value=0)
    by_size.index.name = 'type'
    by_size = by_size.reindex(columns=['small', 'medium', 'large'], fill_value=0)
    by_class = d.groupby(['cat', 'kind']).size().unstack(fill_value=0).reindex(columns=['TP'] + ERRORS, fill_value=0)
    by_class.index = pd.Index([CLASSES[c - 1] for c in by_class.index], name='class')
    fp = by_class[['Cls', 'Loc', 'Both', 'Dupl', 'Bkg']].sum(axis=1)
    by_class['precision'] = (by_class.TP / (by_class.TP + fp).clip(lower=1)).round(3)
    by_class['recall'] = (by_class.TP / (by_class.TP + by_class.Missed).clip(lower=1)).round(3)
    frames = df.groupby('video').image_id.nunique()
    errs = d[d.kind != 'TP'].groupby('video').size().reindex(frames.index, fill_value=0)
    by_video = pd.DataFrame({'frames': frames, 'errors_per_frame': (errs / frames).round(1)}).sort_values('errors_per_frame', ascending=False)
    confused = d[d.kind == 'Cls'].groupby(['gt_cat', 'cat']).size().sort_values(ascending=False).head(5)
    confused.index = pd.Index([f'{CLASSES[g - 1]} -> {CLASSES[p - 1]}' for g, p in confused.index], name='true -> predicted')
    confused.name = 'boxes'
    return by_size, by_class, by_video, confused


def draw_worst(df, gt, data, conf, n, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    d = df[((df.kind != 'Missed') & (df.score >= conf)) | ((df.kind == 'Missed') & df.missed_conf)]
    err = d[~d.kind.isin(['TP', 'Ign'])].groupby(['video', 'image_id']).size().sort_values(ascending=False)
    worst = err.groupby(level=0).head(1).head(n)  # one frame per clip
    files = {im['id']: im['file_name'] for im in gt['images']}
    fig, axes = plt.subplots(1, len(worst), figsize=(7 * len(worst), 5), squeeze=False)
    for ax, ((clip, img_id), cnt) in zip(axes[0], worst.items()):
        ax.imshow(Image.open(os.path.join(data, 'val', files[img_id])))
        for r in d[d.image_id == img_id].itertuples():
            if r.kind == 'Ign':
                continue
            x1, y1, x2, y2 = r.box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=COLORS[r.kind], lw=.8))
        ax.set_title(f'{clip}: {cnt} errors', fontsize=9); ax.axis('off')
    handles = [Rectangle((0, 0), 1, 1, fill=False, color=c, label=k) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(COLORS), fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig(path, dpi=110); plt.close(fig)
    return list(worst.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='.', help='directory with val/')
    ap.add_argument('--model', action='append', help='name=path to val_preds.json (can repeat)')
    ap.add_argument('--conf', type=float, default=.3, help='score threshold for the count tables and pictures')
    ap.add_argument('--val-step', type=int, default=5, help='every n-th val frame, as VAL_STEP in the notebook')
    ap.add_argument('--no-dap', action='store_true', help='skip the dAP part (needs faster-coco-eval)')
    ap.add_argument('--draw', type=int, default=0, help='save the n worst frames (one per clip) to --figs')
    ap.add_argument('--figs', default='images', help='folder for the pictures')
    ap.add_argument('--save', default='out/error_analysis', help='folder for the csv tables')
    args = ap.parse_args()
    models = args.model or ['frcnn_base=out/runs/frcnn_base/val_preds.json', 'final_yolo=out/runs/final_yolo/val_preds.json']

    gt = load_gt(args.data, args.val_step)
    os.makedirs(args.save, exist_ok=True)
    for spec in models:
        name, path = spec.split('=', 1)
        if not os.path.exists(path):
            print(f'skip {name}: {path} not found'); continue
        with open(path) as f:
            df = analyse(json.load(f), gt, args.conf)
        by_size, by_class, by_video, confused = tables(df, args.conf)
        print(f'\n## {name} (score >= {args.conf})\n')
        print('errors by object size\n', md(by_size), '\n')
        print('per class\n', md(by_class), '\n')
        print('most confused classes\n', md(confused), '\n')
        print('clips with most errors per frame\n', md(by_video.head(5)), '\n')
        by_size.to_csv(f'{args.save}/{name}_by_size.csv'); by_class.to_csv(f'{args.save}/{name}_by_class.csv')
        by_video.to_csv(f'{args.save}/{name}_by_video.csv')
        if not args.no_dap:
            dap = delta_ap(df, gt)
            print('AP50 gained by fixing each error type (all scores)\n', md(dap, index=False))
            dap.to_csv(f'{args.save}/{name}_dap.csv', index=False)
        if args.draw:
            os.makedirs(args.figs, exist_ok=True)
            out = os.path.join(args.figs, f'errors_{name}.png')
            frames = draw_worst(df, gt, args.data, args.conf, args.draw, out, f'{name}: worst frames, score >= {args.conf}')
            print('saved', out, frames)


if __name__ == '__main__':
    main()

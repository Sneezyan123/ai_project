import csv, json
import numpy as np
import pandas as pd
from faster_coco_eval import COCO, COCOeval_faster

WEIGHTS = (2.0, 1.0)  # YOLO has the better val score, so it counts more
IOU = 0.55
SUBS = 'out/submissions/'


def iou_one(b, F):
    lt = np.maximum(b[:2], F[:, :2])
    rb = np.minimum(b[2:], F[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(b) + area(F) - inter + 1e-9)


def wbf_class(boxes, scores, weights):
    order = np.argsort(-scores)
    fused = np.zeros((len(order), 4))
    box_sum, w_sum, score_sum = np.zeros((len(order), 4)), np.zeros(len(order)), np.zeros(len(order))
    k = 0
    for i in order:
        c = scores[i] * weights[i]
        j = -1
        if k:
            ious = iou_one(boxes[i], fused[:k])
            m = int(ious.argmax())
            if ious[m] > IOU:
                j = m
        if j < 0:
            j = k
            k += 1
        box_sum[j] += boxes[i] * c
        w_sum[j] += c
        score_sum[j] += c
        fused[j] = box_sum[j] / w_sum[j]
    return fused[:k], score_sum[:k] / sum(WEIGHTS)


def merge(models):
    # models: list of (boxes xyxy, scores, labels), one per model
    boxes = np.concatenate([m[0] for m in models])
    scores = np.concatenate([m[1] for m in models])
    labels = np.concatenate([m[2] for m in models])
    weights = np.concatenate([np.full(len(m[1]), w) for m, w in zip(models, WEIGHTS)])
    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        ids = labels == c
        b, s = wbf_class(boxes[ids], scores[ids], weights[ids])
        out_b.append(b); out_s.append(s); out_l.append(np.full(len(s), c))
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b, s, l = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)
    top = np.argsort(-s)[:500]
    return b[top], s[top], l[top]


# val check: tiled predictions were not saved for val, so full-image YOLO is used here
def load_preds(path):
    per = {}
    for p in json.load(open(path)):
        x, y, w, h = p['bbox']
        per.setdefault(p['image_id'], []).append([x, y, x + w, y + h, p['score'], p['category_id'] - 1])
    return {k: np.array(v) for k, v in per.items()}


def as_model(row):
    if row is None:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return row[:, :4], row[:, 4], row[:, 5].astype(int)


val = json.load(open('val/annotations_coco.json'))
imgs = [dict(i, id=i['id'] + 10_000_000) for i in val['images'] if (i['frame_id'] - 1) % 5 == 0]
ids = {i['id'] for i in imgs}
anns = [dict(a, id=a['id'] + 10_000_000, image_id=a['image_id'] + 10_000_000) for a in val['annotations'] if a['image_id'] + 10_000_000 in ids]
gt = {'images': imgs, 'categories': val['categories'], 'annotations': anns}
A, B = load_preds('out/runs/final_yolo/val_preds.json'), load_preds('out/runs/frcnn_base/val_preds.json')
preds = []
for i in sorted(ids):
    b, s, l = merge([as_model(A.get(i)), as_model(B.get(i))])
    for (x1, y1, x2, y2), sc, c in zip(b.tolist(), s.tolist(), l.tolist()):
        preds.append({'image_id': i, 'category_id': int(c) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})
coco = COCO(gt)
ev = COCOeval_faster(coco, coco.loadRes(preds), 'bbox')
ev.params.maxDets = [1, 100, 500]
ev.evaluate(); ev.accumulate(); ev.summarize()
print('ensemble val mAP %.4f AP50 %.4f AP_small %.4f' % (ev.stats[0], ev.stats[1], ev.stats[3]))


# test csv: best YOLO variant plus Faster R-CNN
def read(path):
    d = pd.read_csv(path, keep_default_na=False)
    out = {}
    for r in d.itertuples():
        v = np.array(r.PredictionString.split(), dtype=float).reshape(-1, 6) if r.PredictionString else np.zeros((0, 6))
        b = v[:, 2:6].copy(); b[:, 2:] += b[:, :2]
        out[r.image_id] = (b, v[:, 1], v[:, 0].astype(int))
    return out


X, Y = read(SUBS + 'final_yolo_tiles_soft.csv'), read(SUBS + 'frcnn_base.csv')
target = SUBS + 'ensemble_yolo_frcnn.csv'
test = pd.read_csv('test_images.csv')
assert set(test.image_id) == set(X) == set(Y)
with open(target, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image_id', 'PredictionString'])
    for image_id in test.image_id:
        b, s, l = merge([X[image_id], Y[image_id]])
        parts = [f'{c} {sc:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}' for (x1, y1, x2, y2), sc, c in zip(b.tolist(), s.tolist(), l.tolist()) if x2 - x1 >= .5 and y2 - y1 >= .5]
        w.writerow([image_id, ' '.join(parts)])

check = pd.read_csv(target, keep_default_na=False)
fields = check.PredictionString.str.split()
assert len(check) == len(test) and check.image_id.is_unique and (fields.map(len) % 6 == 0).all() and fields.map(len).max() <= 3000
assert all(int(p[i]) in range(8) and float(p[i + 4]) > 0 and float(p[i + 5]) > 0 for p in fields for i in range(0, len(p), 6))
print('saved', target, len(check), 'rows, avg boxes', round(float(fields.map(len).mean() / 6), 1))

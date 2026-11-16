"""CPU tests for the box utilities, the data conversion and the post-processing scripts.

    python -m pytest tests -q

Functions from the notebook are taken from its source (by name), so the tests check the code that actually ran.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))

import check_submission  # noqa: E402
import ensemble  # noqa: E402
import error_analysis  # noqa: E402


def notebook_defs(*names, **namespace):
    """Exec the top-level functions/classes with these names from hw2_detection.ipynb into a namespace."""
    nb = json.loads((ROOT / 'notebooks' / 'hw2_detection.ipynb').read_text(encoding='utf-8'))
    found = {}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        try:
            tree = ast.parse(src)
        except SyntaxError:  # cells with notebook-only syntax
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                found[node.name] = ast.get_source_segment(src, node)
    assert set(found) == set(names), f'not found in notebook: {set(names) - set(found)}'
    ns = {'np': np, **namespace}
    for name in names:
        exec(found[name], ns)
    return ns


# ---------- IoU and Soft-NMS (notebook) ----------

def test_box_iou_known_values():
    ns = notebook_defs('box_iou_np')
    a = np.array([[0, 0, 10, 10.]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30.]])
    assert np.allclose(ns['box_iou_np'](a, b)[0], [1, 50 / 150, 0], atol=1e-6)


def test_soft_nms_decays_overlaps_and_keeps_separate_boxes():
    ns = notebook_defs('box_iou_np', 'soft_nms')
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60.]])
    scores = np.array([.9, .8, .7])
    keep, new = ns['soft_nms'](boxes, scores)
    assert keep[0] == 0 and set(keep) == {0, 1, 2}           # nothing is removed outright
    assert new[1] < .8 and np.isclose(new[2], .7) and np.isclose(new[0], .9)
    keep, _ = ns['soft_nms'](boxes, scores, min_score=.5)     # with a higher floor the duplicate goes
    assert set(keep) == {0, 2}


# ---------- COCO -> YOLO labels -> torchvision targets (notebook) ----------

def test_make_set_and_frameset_round_trip(tmp_path):
    from PIL import Image, ImageDraw
    from tqdm.auto import tqdm
    torch = pytest.importorskip('torch')
    pil_to_tensor = pytest.importorskip('torchvision.transforms.functional').pil_to_tensor

    img_path = tmp_path / 'f1.jpg'
    Image.new('RGB', (200, 100), (255, 255, 255)).save(img_path)
    im = {'id': 1, 'video': 'v1', 'frame_id': 1, 'width': 200, 'height': 100, 'path': str(img_path)}
    anns = [{'image_id': 1, 'category_id': 3, 'bbox': [10, 20, 30, 40], 'iscrowd': 0},
            {'image_id': 1, 'category_id': 1, 'bbox': [150, 50, 40, 40], 'iscrowd': 1}]
    mask = (123, 117, 104)
    ns = notebook_defs('make_set', 'FrameSet', images=[im], by_image={1: anns}, SETS=tmp_path / 'sets', TRAIN_STEP=10,
                       VAL_STEP=5, MASK_COLOR=mask, CLASSES=['c%d' % i for i in range(8)], tqdm=tqdm, Image=Image,
                       ImageDraw=ImageDraw, Dataset=object, torch=torch, pil_to_tensor=pil_to_tensor)
    root = ns['make_set']('t', ['v1'], [])
    label = (root / 'labels' / 'train' / 'v1_0000001.txt').read_text().split()
    assert label[0] == '2' and np.allclose([float(v) for v in label[1:]], [25 / 200, 40 / 100, 30 / 200, 40 / 100])
    pic = np.array(Image.open(root / 'images' / 'train' / 'v1_0000001.jpg'))
    assert np.abs(pic[70, 170].astype(int) - mask).max() < 10  # ignored region is painted
    torch.manual_seed(0)
    seen = set()
    for _ in range(20):  # random flip: both variants must give the right box
        _, t = ns['FrameSet'](root)[0]
        assert t['labels'].tolist() == [3]
        box = tuple(np.round(t['boxes'][0].numpy(), 3))
        assert box in {(10, 20, 40, 60), (160, 20, 190, 60)}
        seen.add(box)
    assert len(seen) == 2


# ---------- WBF ensemble ----------

def test_wbf_same_box_keeps_box_and_score():
    bx = np.array([[10, 10, 50, 50.]])
    b, s, l = ensemble.merge([(bx, np.array([.9]), np.array([1])), (bx, np.array([.9]), np.array([1]))], (2, 1), .55)
    assert np.allclose(b, bx) and np.allclose(s, .9) and l.tolist() == [1]


def test_wbf_single_model_box_is_down_weighted_and_classes_not_mixed():
    a = (np.array([[0, 0, 10, 10.]]), np.array([.9]), np.array([0]))
    b = (np.array([[0, 0, 10, 10.]]), np.array([.9]), np.array([5]))  # same place, other class
    boxes, scores, labels = ensemble.merge([a, b], (2, 1), .55)
    assert sorted(labels.tolist()) == [0, 5]
    assert np.allclose(sorted(scores), [.9 / 3, .9 * 2 / 3])


def test_wbf_averages_by_confidence():
    a = (np.array([[0, 0, 10, 10.]]), np.array([.8]), np.array([2]))
    b = (np.array([[2, 0, 12, 10.]]), np.array([.4]), np.array([2]))
    boxes, _, _ = ensemble.merge([a, b], (1, 1), .55)
    assert np.allclose(boxes[0], [2 * .4 / 1.2, 0, 10 + 2 * .4 / 1.2, 10])


def test_wbf_respects_max_det():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (700, 2))
    m = (np.c_[xy, xy + 5], rng.uniform(.1, 1, 700), np.zeros(700, int))
    assert len(ensemble.merge([m, ensemble.EMPTY], (1, 1), .55)[1]) == 500


# ---------- submission check ----------

def test_check_submission():
    good = pd.DataFrame({'image_id': ['a', 'b'], 'PredictionString': ['0 0.9 1 2 3 4', '']})
    assert check_submission.check(good, ['a', 'b']) == []
    bad = pd.DataFrame({'image_id': ['a', 'a'], 'PredictionString': ['8 0.9 1 2 3 4', '0 0.9 1 2 0 4 1']})
    problems = ' '.join(check_submission.check(bad, ['a', 'c']))
    for text in ('duplicated', 'missing', 'class 8', 'multiple of 6'):
        assert text in problems


# ---------- error types ----------

def test_error_types():
    g = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210.]])
    gl = np.array([1, 1, 2])
    p = np.array([[0, 0, 10, 10],        # TP
                  [0, 0, 10, 10],        # Dupl of the first
                  [200, 200, 210, 210],  # Cls: right place, class 1 instead of 2
                  [103, 103, 113, 113],  # Loc: IoU about 0.32 with a class-1 box
                  [500, 500, 510, 510],  # Ign: inside the crowd region
                  [800, 800, 810, 810.]])  # Bkg
    pl = np.array([1, 1, 1, 1, 1, 1])
    types, refs, matched = error_analysis.classify_image(p, pl, g, gl, crowd=[[450, 450, 600, 600]])
    assert types == ['TP', 'Dupl', 'Cls', 'Loc', 'Ign', 'Bkg']
    assert refs[2] == 2 and refs[3] == 1
    assert matched.tolist() == [True, False, False]  # boxes 1 and 2 are missed


def test_error_analysis_tables_and_fixes():
    gt = {'images': [{'id': 1, 'video': 'v', 'file_name': 'x.jpg', 'frame_id': 1}],
          'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(error_analysis.CLASSES)],
          'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0},
                          {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [50, 50, 10, 10], 'area': 100, 'iscrowd': 0}]}
    preds = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .9},
             {'image_id': 1, 'category_id': 1, 'bbox': [300, 300, 10, 10], 'score': .8}]
    df = error_analysis.analyse(preds, gt, conf=.3)
    assert sorted(df.kind) == ['Bkg', 'Missed', 'TP']
    by_size, by_class, _, _ = error_analysis.tables(df, .3)
    assert by_size.loc['TP', 'small'] == 1 and by_size.loc['Missed', 'small'] == 1
    assert by_class.loc['person', 'precision'] == .5 and by_class.loc['bicycle', 'recall'] == 0
    assert len(error_analysis.apply_fix(df, 'Bkg', gt)) == 1      # background box removed
    assert len(error_analysis.apply_fix(df, 'Missed', gt)) == 3   # missed box added


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))

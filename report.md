---
title: "Homework 2 — Multiple Object Detection in VisDrone Videos"
author: "Artem Onyshchuk, Ivan Polianskyi"
course: Artificial Intelligence · UCU
date: 2026-11-09
---

## Task and approach

During this project Me (Ivan) and Artem focuses on detecting objects in VisDrone video frames. We detect 8 classes (person, bicycle, car, van, truck, tricycle, bus, motor) Every frame is handled on its own, we do not use neighbouring frames. The hard part is size: half of the objects are smaller than 32 px, and some frames have hundreds of them. All code in `notebooks/`.

## Data preparation

We looked at class shares, box sizes and objects per frame for each clip. The number that shaped the rest of the work: after the torchvision resize (short side 800, long side up to 1333) the median box side is 31.7 px, and 50.9% of boxes are below 32 px. The default RetinaNet anchors start at 32 px, so we added a run with 16 px anchors and a YOLO run at 1280 instead of 960.

To reduce training time and repeated examples, we keep every 10-th training frame and every fifth validation frame.

COCO annotations are converted to YOLO format: class index and normalised box centre, width, and height. For torchvision, boxes are converted back to corner coordinates, with class zerro reserved for background. Regions marked `iscrowd` are excluded from target boxes and painted over with RGB `(123, 117, 104)` in training images. This avoids treating unlabelled objects in these regions as background

For torchvision, images are converted to RGB tensors in [0, 1], followed by the models’ built-in resizing and normalisation

## Validation

The main comparison uses the provided train/validation split by video sequence. Randomly splitting individual frames could inflate the scores because neighbouring frames are very similar.

All models are evaluated through the same `coco_eval` function on the same original, unpainted validation frames. The recorded metrics are mAP over IoU thresholds 0.50–0.95, AP50, AP75, and AP for small, medium, and large objects. Evaluation allows up to 500 detections per frame and passes ignored regions through `iscrowd`.

Also implements three-fold `GroupKFold` validation by sequence for the YOLO baseline. It runs when `RUN_CV=True`, which defaults to a GPU-memory-based setting. This checks one model’s stability. cross-validation for every method and a pooled out-of-fold score are still missing

## Models and experiments

All models start from COCO-pretrained weights and train for 12 epochs with seed 42. We had one rented A40, so 12 epochs was what fit into the budget for 13 runs. Batch size depends on GPU memory.

**Faster R-CNN ResNet-50 FPN v2.** The RPN proposes candidate regions, and the second stage classifies them and refines their boxes. We test more RPN proposals to retain potential objects in crowded scenes, plus a smaller Smooth L1 beta for box regression together with label smoothing in the classification loss. Training uses SGD, momentum 0.9, weight decay 0.0001, a 300-iteration warmup, and tenfold learning-rate reductions after epochs 8 and 11.

**RetinaNet ResNet-50 FPN v2.** This model predicts directly from anchors without a separate proposal stage. Focal loss reduces the contribution of easy examples, addressing the large number of background locations. We test smaller anchors and different focal loss settings separately.

**YOLO11s.** We picked it because Ultralytics trains it fast and it needs no anchor tuning. Its task-aligned assigner chooses positive locations by class score and IoU together, so even tiny boxes get some positives. We tried a larger input (1280 instead of 960) and higher box/DFL loss weights

**RT-DETR-L.** Each ground-truth box is matched to exactly one query (Hungarian matching), so the model learns not to output duplicates and needs no NMS. We set up runs with AdamW at lr 2e-4 and with more weight on GIoU.

| Run | Configuration or change from baseline |
|---|---|
| `frcnn_base` | Faster R-CNN baseline |
| `frcnn_rpn` | RPN proposals: 4000 train / 2000 test instead of 2000/1000 |
| `frcnn_loss` | Smooth L1 beta=0.05; label smoothing=0.1 |
| `retina_base` | RetinaNet with default anchors and focal loss |
| `retina_anchors` | Base anchor sizes: 16, 32, 64, 128, 256 |
| `retina_focal` | Focal alpha=0.35, gamma=1.5 instead of 0.25/2 |
| `yolo_base` | YOLO11s, input size 960 |
| `yolo_1280` | Input size 1280 (batch 8 instead of 16 on a larger GPU, to fit in memory) |
| `yolo_box` | Loss weights: box=10, cls=0.7, dfl=2 |
| `rtdetr_base` | RT-DETR-L, input size 640 |
| `rtdetr_lr` | AdamW, lr=0.0002, lrf=0.1 |
| `rtdetr_loss` | Loss weights: bbox=2, giou=5 instead of 5/2 |
| `final_yolo` | YOLO11m/1280 on a larger GPU; YOLO11s/960 on a smaller GPU |
| `+full`, `+tiles_nms`, `+tiles_soft` | Full frame; tiles with NMS; tiles with Soft-NMS |
| `cv_yolo_0…2` | Three sequence-level folds for YOLO11s |
| `ensemble_yolo_frcnn` | Weighted boxes fusion of `final_yolo_tiles_soft.csv` and `frcnn_base.csv` (weights 2:1, IoU 0.55), `notebooks/ensemble.py` |

### Results

| Run | val mAP | val AP50 | val AP_small | Public | Private |
|---|---|---|---|---|---|
| `frcnn_base` | 0.2197 | 0.4421 | 0.126 | 0.1981 | 0.1863 |
| `frcnn_rpn`, `frcnn_loss` | not finished | | | — | — |
| `retina_base`, `retina_anchors`, `retina_focal` | not finished | | | — | — |
| `yolo_base`, `yolo_1280`, `yolo_box` | not saved | | | — | — |
| `rtdetr_base`, `rtdetr_lr`, `rtdetr_loss` | not finished | | | — | — |
| `final_yolo` (full frame) ✔ | not saved | | | 0.2555 | 0.2222 |
| `final_yolo+tiles_nms` | not saved | | | 0.2384 | 0.2108 |
| `final_yolo+tiles_soft` | not saved | | | 0.2429 | 0.2145 |
| `ensemble_yolo_frcnn` ✔ (val: full-frame YOLO + `frcnn_base`) | 0.2654 | 0.5099 | 0.1507 | **0.2599** | **0.2314** |

✔ marks the two submissions selected for the final score. Public and private scores were read from the Kaggle submissions page. Run `python notebooks/make_report_tables.py --kaggle notebooks/kaggle_scores.csv` to rebuild this table from `out/results_*.csv`.

**Tiling did not help.** Both tiled variants scored lower than full-frame inference: −0.017 public and −0.011 private with NMS, −0.013 and −0.008 with Soft-NMS. Possible causes: at `imgsz=1280` a 60% crop is upscaled, so the model sees objects larger than during training. Boxes cut at crop borders survive merging as extra false positives. Soft-NMS lowers the scores of duplicates instead of removing them, so the 500-box limit fills with low-scoring boxes. We could not confirm this on validation, because tiled validation predictions were not saved for that run. The notebook now saves them.

**The ensemble helped despite the weaker YOLO input.** It fused `tiles_soft` (0.2429 public) with Faster R-CNN (0.1981) and still beat full-frame YOLO, at +0.004 public and +0.009 private. This matches the validation gain of the full-frame ensemble over `frcnn_base`. The two models make different errors, and WBF averages box coordinates rather than discarding boxes. Fusing full-frame YOLO with Faster R-CNN is the obvious next step.

**Final selection.** We chose one safe and one riskier submission: `final_yolo.csv` (best single model) and `ensemble_yolo_frcnn.csv` (the ensemble, which also gained on validation). They turned out to be the top two on the private leaderboard as well!

## Final pipeline and error analysis

For the YOLO model with the highest validation mAP, the notebook compares full-frame inference with four overlapping crops, each covering 60% of the frame’s width and height. Crop predictions are mapped back to frame coordinates and combined with full-frame predictions. The merging options are NMS at IoU 0.6 and Gaussian Soft-NMS with sigma 0.5

`notebooks/error_analysis.py` breaks the errors of the saved validation predictions into TIDE-style types: wrong class, poor localisation, duplicate, background and missed. It counts them by object size, class and clip, measures how much AP50 an oracle fix of each type would recover, and draws the worst frames to `images/`.

## Reproduction

The project requires Python, CUDA-compatible PyTorch/torchvision, Ultralytics, faster-coco-eval, NumPy, pandas, Pillow, matplotlib, scikit-learn, tqdm, and Jupyter. `requirements.txt` lists the local environment. The runs completed so far used a Runpod pod with an NVIDIA A40 and the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8) with Ultralytics 8.4.208.

1. Place `train/`, `val/`, `test/`, and `test_images.csv` in the data directory. Set `HW2_DATA` for a local run.
2. Open `notebooks/hw2_detection.ipynb`, set the environment variable `HW2_PART=all`, and execute the cells in order. Alternatively, run `yolo`, `rcnn`, and `detr` separately, then `merge` to build the joint tables. With many CPU cores, set `OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=8`, or OpenBLAS can crash the kernel. `HW2_BIG_GPU=0/1` forces the batch sizes, and `HW2_WORKERS` sets the dataloader workers.
3. For the ensemble submission, run `python notebooks/ensemble.py` from the directory that contains `out/`, `val/`, and `test_images.csv`.
4. Check `out/results_*.csv`, checkpoints and training logs in `out/runs/`, and prediction files in `out/submissions/`.
5. Seed 42 alone does not guarantee identical outputs across GPUs.

`ensemble.py` takes the model pair, weights and IoU as arguments (see `--help`). It validates the same YOLO variant that goes into the test file when `val_preds_tiles_soft.json` exists, otherwise it falls back to full-frame YOLO. The validation score is written to `out/results_ensemble.csv`. `check_submission.py` validates a CSV before upload, and `python -m pytest tests` runs CPU tests of the box utilities, the label conversion, WBF and the error types.

## References

Code was written with help from AI assistants. Ideas and libraries used:

0. O. Basystyi. *Multiple Object Video Detection, HW2, UCU AI 2026.* Kaggle, 2026. https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026
1. P. Zhu et al. *Detection and Tracking Meet Drones Challenge.* IEEE TPAMI, 2021. VisDrone dataset.
2. S. Ren, K. He, R. Girshick, J. Sun. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS, 2015.
3. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for Dense Object Detection.* ICCV, 2017. RetinaNet.
4. Y. Li, S. Xie, X. Chen, P. Dollár, K. He, R. Girshick. *Benchmarking Detection Transfer Learning with Vision Transformers.* 2021. Source of the torchvision `v2` Faster R-CNN and RetinaNet weights.
5. Y. Zhao et al. *DETRs Beat YOLOs on Real-time Object Detection.* CVPR, 2024. RT-DETR.
6. G. Jocher, J. Qiu. *Ultralytics YOLO11*, 2024. https://github.com/ultralytics/ultralytics
7. N. Bodla, B. Singh, R. Chellappa, L. S. Davis. *Soft-NMS — Improving Object Detection With One Line of Code.* ICCV, 2017.
8. R. Solovyev, W. Wang, T. Gabruseva. *Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models.* Image and Vision Computing, 2021. Our `ensemble.py` is a simplified re-implementation.
9. F. C. Akyon, S. O. Altinuc, A. Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* ICIP, 2022. Idea behind the overlapping-crop inference.
10. torchvision detection models and references: https://github.com/pytorch/vision
11. faster-coco-eval: https://github.com/MiXaiLL76/faster_coco_eval

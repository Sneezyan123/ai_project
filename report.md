---
title: "Homework 2 — Multiple Object Detection in VisDrone Videos"
author: "Artem Onyshchuk, Ivan Polianskyi"
course: Artificial Intelligence · UCU
date: 2026-11-09
---

## Task and approach

This project focuses on detecting objects in VisDrone video frames. For each frame, the model predicts a class and a bounding box for every object. The prepared dataset contains eight classes: person, bicycle, car, van, truck, tricycle, bus, and motor. Frames are processed independently, without using predictions from neighbouring frames.

The main challenges are small objects and crowded scenes. After resizing, a person or car may occupy only a few pixels, while nearby objects can be confused or merged. We therefore prepared a comparison of four detector families, followed by experiments with resolution, loss functions, and inference strategies.

The methods are implemented in `notebooks/hw2_detection.ipynb`.

## Data preparation

The exploration covers class frequencies, bounding box sizes, and objects per frame across sequences. We also compare image resolutions between train, validation, and test. These checks help choose settings that fit the dataset.

The saved training statistics show that, after resizing to a short side of 800 pixels and a maximum long side of 1333, the median square root of box area is 31.7 pixels. For 50.9% of boxes, this value is below 32 pixels. This motivates trying smaller anchors and higher input resolution.

To reduce training time and repeated examples, we keep every tenth training frame and every fifth validation frame. All baseline models use the same sampling. One drawback is that brief object appearances may be missed, so the sampling interval would also be worth testing separately.

COCO annotations are converted to YOLO format: class index and normalised box centre, width, and height. For torchvision, boxes are converted back to corner coordinates, with class zero reserved for background. Regions marked `iscrowd` are excluded from target boxes and painted over with RGB `(123, 117, 104)` in training images. This avoids treating unlabelled objects in these regions as background. Painting also changes image appearance, so it is a compromise. Ground-truth overlays are included to check box coordinates visually.

For torchvision, images are converted to RGB tensors in [0, 1], followed by the models’ built-in resizing and normalisation. Horizontal flipping is applied with probability 0.5. YOLO uses Ultralytics augmentation, with mosaic disabled for the final two epochs. The augmentation differences between frameworks need to be considered when comparing results.

## Validation

The main comparison uses the provided train/validation split by video sequence. Randomly splitting individual frames could inflate the scores because neighbouring frames are very similar.

All models are evaluated through the same `coco_eval` function on the same original, unpainted validation frames. The recorded metrics are mAP over IoU thresholds 0.50–0.95, AP50, AP75, and AP for small, medium, and large objects. Evaluation allows up to 500 detections per frame and passes ignored regions through `iscrowd`. FPS includes image loading and prediction, so speed should be compared on the same hardware.

The notebook also implements three-fold `GroupKFold` validation by sequence for the YOLO baseline. It runs when `RUN_CV=True`, which defaults to a GPU-memory-based setting. This checks one model’s stability; cross-validation for every method and a pooled out-of-fold score are still missing. Any statistics used to tune anchors should be computed only from the training portion of the relevant fold.

## Models and experiments

We use pretrained weights to spend the available GPU time adapting models to VisDrone. The main notebook uses 12 epochs and seed 42 for all training runs. Equal epoch counts do not imply equal compute budgets, so runtime also needs to be recorded. Batch size depends on GPU memory.

**Faster R-CNN ResNet-50 FPN v2.** The RPN proposes candidate regions, and the second stage classifies them and refines their boxes. We test more RPN proposals to retain potential objects in crowded scenes, plus a smaller Smooth L1 beta for box regression together with label smoothing in the classification loss. Training uses SGD, momentum 0.9, weight decay 0.0001, a 300-iteration warmup, and tenfold learning-rate reductions after epochs 8 and 11.

**RetinaNet ResNet-50 FPN v2.** This model predicts directly from anchors without a separate proposal stage. Focal loss reduces the contribution of easy examples, addressing the large number of background locations. We test smaller anchors and different focal loss settings separately. Smaller anchors may suit tiny objects better, but could also increase false positives.

**YOLO11s.** We use this anchor-free detector as a practical baseline. It does not require predefined anchor shapes; training locations are assigned to objects using classification scores and box overlap. The experiments increase input resolution from 960 to 1280 and adjust the box, classification, and DFL loss weights. Higher resolution should preserve more detail, at the cost of memory and inference time.

**RT-DETR-L.** This query-based model produces a set of predictions matched one-to-one with ground truth during training. The matching discourages duplicates and removes the need for conventional NMS. We compare a different learning rate and adjusted bbox/GIoU loss weights. Training curves are needed to check whether 12 epochs are enough for convergence before drawing conclusions about this architecture.

The table lists the notebook’s main configurations.

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

| Run | val mAP | val AP50 | val AP_small | Kaggle submission |
|---|---|---|---|---|
| `frcnn_base` | 0.2197 | 0.4421 | 0.126 | `frcnn_base.csv`, submission error |
| `frcnn_rpn`, `frcnn_loss` | not finished | | | — |
| `retina_base`, `retina_anchors`, `retina_focal` | not finished | | | — |
| `yolo_base`, `yolo_1280`, `yolo_box` | not saved | | | — |
| `rtdetr_base`, `rtdetr_lr`, `rtdetr_loss` | not finished | | | — |
| `final_yolo` (full frame) | not saved | | | `final_yolo.csv` |
| `final_yolo+tiles_nms` | not saved | | | `final_yolo_tiles_nms.csv` |
| `final_yolo+tiles_soft` | not saved | | | `final_yolo_tiles_soft.csv` |
| `ensemble_yolo_frcnn` (val: full-frame YOLO + `frcnn_base`) | 0.2654 | 0.5099 | 0.1507 | `ensemble_yolo_frcnn.csv` |

## Final pipeline and error analysis

For the YOLO model with the highest validation mAP, the notebook compares full-frame inference with four overlapping crops, each covering 60% of the frame’s width and height. Crop predictions are mapped back to frame coordinates and combined with full-frame predictions. The merging options are NMS at IoU 0.6 and Gaussian Soft-NMS with sigma 0.5. This isolates the effect of inference changes without retraining.

The notebook includes prediction matching and visualisation of frames with the most errors.

Selection considers validation mAP, AP_small, stability, and speed.

## Reproduction

The project requires Python, CUDA-compatible PyTorch/torchvision, Ultralytics, faster-coco-eval, NumPy, pandas, Pillow, matplotlib, scikit-learn, tqdm, and Jupyter. Local versions are recorded in `requirements.txt`. The runs completed so far used a Runpod pod with an NVIDIA A40 and the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8) with Ultralytics 8.4.208.

1. Place `train/`, `val/`, `test/`, and `test_images.csv` in the data directory. Set `HW2_DATA` for a local run.
2. Open `notebooks/hw2_detection.ipynb`, set the environment variable `HW2_PART=all`, and execute the cells in order. Alternatively, run `yolo`, `rcnn`, and `detr` separately, then `merge` to build the joint tables. On machines that report many CPU cores, set `OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=8`; otherwise OpenBLAS can crash the kernel.
3. For the ensemble submission, run `python notebooks/ensemble.py` from the directory that contains `out/`, `val/`, and `test_images.csv`.
4. Check `out/results_*.csv`, checkpoints and training logs in `out/runs/`, and prediction files in `out/submissions/`. The notebook checks row counts, class indices, and box formatting.
5. Seed 42 alone does not guarantee identical outputs across GPUs.

# model/yolov8n_640_int8.tflite

Stock COCO YOLOv8n, full integer quantized, NHWC — ready for larod on ARTPEC-8.
(The filename says int8; the I/O is actually uint8. Kept for now to avoid churning
the build, but rename it when the model is next re-exported.)

## Tensor contract

Read straight off the shipped file by `tools/parameter_finder.py`:

| | shape | dtype | scale | zero point |
|---|---|---|---|---|
| input | 1 x 640 x 640 x 3 | uint8 | 0.00392157 (=1/255) | 0 |
| output boxes | 1 x 4 x 8400 | uint8 | 2.5758814811706543 | 0 |
| output scores | 1 x 80 x 8400 | uint8 | 0.00390625 (=1/256) | 0 |

Output order is not guaranteed, so the ACAP tells the two apart by byte size.

Input is RGB, 0..255 straight through the scale, so larod's `convert` preprocessor output
feeds the model with no conversion at all.

Boxes are `cx, cy, w, h` in **pixels of the model input**. Scores are already sigmoid'd, per
class, with no objectness term. NMS is not in the graph — it runs on the CPU in the ACAP.

## Why the graph is cut before the final Concat

Ultralytics' single `output0` (1 x 84 x 8400) concatenates box coordinates (range 0..~650) with
class scores (0..1). Per-tensor quantization then picks one scale for both: **2.5758815**. Every
score below 1.29 rounds to zero, so a whole-graph export detects nothing.

Measured on `datasets/coco8/images/val/000000000009.jpg`:

| export | max class score |
|---|---|
| float32 reference | 0.665 |
| single-output | **0.000** (1 unique value across the whole score tensor) |
| split-output (this model) | 0.625 |

Split top detections on the same image: person 0.54 / 0.50 / 0.46, bowl 0.38, banana 0.38 —
consistent with the float reference.

## Verified on hardware (2026-09-11)

Running on the Q1656 at 192.168.1.156. larod accepted the **whole graph** on
`axis-a8-dlpu-tflite` — 54 ms/frame inference, so the ops once flagged as suspects (TRANSPOSE,
STRIDED_SLICE, RESIZE_NEAREST_NEIGHBOR, SOFTMAX in the DFL head) are not a problem in practice.
Cutting the head out of the graph is unnecessary.

Box geometry checked against a live snapshot: logged coordinates `tv [0.76, 0.00, 0.99, 0.44]`
and `toilet [0.00, 0.91, 0.11, 0.99]` land exactly on the drawn overlay rectangles. The
pixels-to-frame-normalized conversion is correct.

## Known weaknesses of this export

- **Square input on a 16:9 sensor.** VDO delivers the frame at the native aspect ratio and larod
  scales it to 640x640, so the model sees the scene squashed to about 56 % of its width — not
  what the COCO weights were trained on. `tools/export_yolov8.sh` now defaults to a rectangular
  `384x640`, which is close to 16:9 and 40 % fewer input pixels; re-export to pick it up. The
  ACAP reads width and height separately, so nothing in the C needs to change.
- **Calibrated on 8 images.** The original export used the `coco8` sample set, which is far too
  small to set the box tensor's scale and every intermediate activation range. The export script
  now wants a few hundred images and mixes in whatever is in `verify/`.

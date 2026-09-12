# model/yolov8n_384x640_uint8.tflite

Stock COCO YOLOv8n, full integer quantized, NHWC, **rectangular 384x640 input** — for larod on
ARTPEC-8.

## Tensor contract

Read straight off the shipped file by `tools/parameter_finder.py`:

| | shape | dtype | scale | zero point |
|---|---|---|---|---|
| input | 1 x 384 x 640 x 3 | uint8 | 0.00392157 (=1/255) | 0 |
| output boxes | 1 x 4 x 5040 | uint8 | 2.5369834899902344 | 0 |
| output scores | 1 x 80 x 5040 | uint8 | 0.00390625 (=1/256) | 0 |

sha256 `0357c7808b3d20a4fecb0506dc2accec510c4c57e4f79a28fdb16f81790aaf7e`, recorded in
`acap/app/model_params.h` so `acap/build.sh` can refuse a mismatched pair.

Output order is not guaranteed, so the ACAP tells the two apart by byte size (20160 vs 403200).

Input is RGB, 0..255 straight through the scale, so larod's `convert` preprocessor output feeds
the model with no conversion at all. Boxes are `cx, cy, w, h` in **pixels of the model input**;
the ACAP divides by width and height separately, so the rectangular input needs no special
handling. Scores are already sigmoid'd, per class, no objectness. NMS runs on the CPU.

## Why rectangular

VDO delivers the frame at the sensor's native aspect and larod scales it to the model input with
no letterbox. A square model therefore sees a 16:9 scene squashed to about 56 % of its width —
not what the COCO weights were trained on. 384x640 is 1.67:1, close enough to 16:9 that the
distortion nearly disappears, and it is 40 % fewer input pixels (5040 anchors instead of 8400).

Measured on 28 coco128 images held out of both models' calibration sets, decoded exactly as the
ACAP decodes (conf 0.25, IoU 0.45, area <= 0.90), scored against the float32 PyTorch model:

**Frames cropped to 16:9, i.e. what the Q1656 delivers** — 113 reference detections:

| model | input | matched | recall | false positives |
|---|---|---|---|---|
| old, square | 640x640 | 65 | 57.5 % | 19 |
| **this one** | 384x640 | **99** | **87.6 %** | **11** |

**The same images uncropped, i.e. ~4:3 photos** — 124 reference detections:

| model | input | matched | recall | false positives |
|---|---|---|---|---|
| old, square | 640x640 | 81 | 65.3 % | 16 |
| this one | 384x640 | 64 | 51.6 % | 9 |

The two tables are the same finding: whichever model's aspect ratio matches the input wins, and
by a lot. This model is the right one for a 16:9 camera and the wrong one for 4:3 stills. If the
camera is ever reconfigured to a 4:3 view area, re-export square.

## Calibration

100 coco128 images plus the camera snapshot in `verify/`, resized to 640x384 the same way
inference resizes — no letterbox, so calibration sees what the DLPU sees. The previous export
used 8 images, which is not enough to fix the box tensor's scale or the intermediate activation
ranges. The remaining 28 coco128 images were held out for the comparison above.

## Why the graph is cut before the final Concat

Ultralytics' single `output0` (1 x 84 x anchors) concatenates box coordinates (range 0..~650)
with class scores (0..1). Per-tensor quantization then picks one scale for both, around 2.54.
Every score below ~1.27 rounds to zero, so a whole-graph export detects nothing. `-onimc` cuts
the graph at `/model.22/Mul_2_output_0` and `/model.22/Sigmoid_output_0` so each tensor gets its
own scale.

## Verified on hardware (2026-09-11, square model)

larod accepted the **whole graph** on `axis-a8-dlpu-tflite` — 54 ms/frame inference at 640x640,
so the ops once flagged as suspects (TRANSPOSE, STRIDED_SLICE, RESIZE_NEAREST_NEIGHBOR, SOFTMAX
in the DFL head) are not a problem in practice. Box geometry checked against a live snapshot and
found correct.

The 384x640 model has the same op set and 40 % fewer input pixels, so it should load the same way
and run faster — **but that has not been measured on the camera yet.**

## Box geometry (2026-09-12, 384x640 model)

Verified two ways. Off camera: the shipped TFLite, decoded exactly as the ACAP does (÷640, ÷384),
matched against COCO ground truth on 90 people in coco128 cropped to 16:9 — mean centre offset
+0.005 / −0.018 of the frame, i.e. no bias. On camera: `live.json` compared against a JPEG
snapshot carrying the overlay, rotation 180, zoom 1.00, on the full view and on view area 2 —
raw decoded coordinates land on the subject in both axes.

Two things that looked like decode bugs and were not:

- **`bbox_coordinates_frame_normalized` needs no rotation compensation.** At rotation 180 the
  frame VDO delivers and the frame bbox draws on are the same frame. Flipping y (or both axes)
  "fixed" a centred subject and broke an off-centre one; only an off-centre subject can tell a
  correct box from a mirrored one.
- **Digital zoom misaligns the live view by design.** The detector sees the whole channel; a
  zoomed display stream is a crop of it. Boxes then appear shrunk toward one corner. The zoom
  readout in the camera UI is the tell — it must read 1.00 before judging alignment.

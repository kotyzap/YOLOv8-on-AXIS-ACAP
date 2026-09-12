#!/usr/bin/env bash
# YOLOv8n (stock COCO) -> NHWC full-uint8 TFLite for the ARTPEC-8 DLPU.
# Run off-camera, from the repo root. See model/MODEL.md for why each flag matters.
#
#   pip install ultralytics tensorflow-cpu tf_keras onnx onnx2tf onnxslim \
#               onnxruntime onnx_graphsurgeon sng4onnx sne4onnx ai-edge-litert
set -euo pipefail

# Rectangular input, height x width, both multiples of 32.
#
# 384x640 is 1.67:1, near enough to the 16:9 the camera actually delivers. The
# square 640x640 model got the full frame squashed to ~56 % of its width by the
# larod convert step, which is not what the COCO weights were trained on, and it
# spent 40 % more pixels to do it. Set IMGSZ=640,640 to go back to square.
IMGSZ=${IMGSZ:-384,640}
CALIB_GLOB=${CALIB_GLOB:-datasets/coco/images/val2017/*.jpg}
CALIB_N=${CALIB_N:-300}

IFS=, read -r CH CW <<<"$IMGSZ"

# 1. PyTorch -> ONNX (no NMS, no head fusion)
yolo export model=yolov8n.pt format=onnx opset=13 imgsz="$CH,$CW" nms=False simplify=True

# 2. Calibration array: N x H x W x 3, float32, 0..1, RGB.
#
#    Eight images (the old coco8 set) is not a calibration set: it fixes the box
#    tensor's scale and every intermediate activation range from almost no data.
#    A few hundred is cheap and much steadier. Resize the same way inference
#    does -- straight to HxW, no letterbox -- so calibration sees what the DLPU
#    will see. Mix in camera snapshots if you have them; verify/ is a good start.
CALIB_GLOB="$CALIB_GLOB" CALIB_N="$CALIB_N" CH="$CH" CW="$CW" python - <<'PY'
import glob, os, numpy as np, cv2
fs = sorted(glob.glob(os.environ["CALIB_GLOB"]))[: int(os.environ["CALIB_N"])]
fs += sorted(glob.glob("verify/*.jpg"))
if len(fs) < 50:
    raise SystemExit(f"only {len(fs)} calibration images matched; set CALIB_GLOB")
h, w = int(os.environ["CH"]), int(os.environ["CW"])
a = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (w, h)).astype(np.float32) / 255.0
     for f in fs]
np.save("calib.npy", np.stack(a))
print(f"calibration set: {len(a)} images at {w}x{h}")
PY

# 3. ONNX -> NHWC uint8 TFLite, CUT BEFORE THE FINAL CONCAT (two separate outputs)
#    -onimc keeps boxes and scores as separate tensors so each gets its own
#    quantization scale. Without it the shared per-tensor scale (~2.58) zeroes
#    every class score.
#    -iqd/-oqd uint8 matches what larod's convert preprocessor hands over, so the
#    ACAP does no conversion at all.
onnx2tf -i yolov8n.onnx -oiqt -qt per-tensor -iqd uint8 -oqd uint8 \
  -onimc /model.22/Mul_2_output_0 /model.22/Sigmoid_output_0 \
  -cind images calib.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]" \
  -o tf_split

cp tf_split/yolov8n_full_integer_quant.tflite model/yolov8n_640_int8.tflite

# 4. Regenerate the committed quantization header (and its model hash).
python tools/parameter_finder.py model/yolov8n_640_int8.tflite acap/app/model_params.h

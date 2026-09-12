#!/usr/bin/env bash
# YOLOv8n (stock COCO) -> NHWC full-uint8 TFLite for the ARTPEC-8 DLPU.
# Run off-camera, from the repo root. See model/MODEL.md for why each flag matters.
#
#   pip install ultralytics tensorflow-cpu tf_keras onnx onnx2tf onnxslim \
#               onnxruntime onnx_graphsurgeon sng4onnx sne4onnx ai-edge-litert
set -euo pipefail

# Rectangular input, height x width, both multiples of 32.
#
# 384x640 is 1.67:1, near enough to the 16:9 the camera delivers. A square model
# gets the full frame squashed to ~56 % of its width by the larod convert step,
# which is not what the COCO weights were trained on: on 16:9 frames that cost
# 30 points of recall (model/MODEL.md has the numbers). It also spends 40 % more
# input pixels to do it. Set IMGSZ=640,640 only if the camera is reconfigured to
# a 4:3 view area, where square wins instead.
IMGSZ=${IMGSZ:-384,640}

# Calibration images. Use datasets/coco/images/val2017 if you have COCO locally.
# Otherwise coco128 is one download and is what the shipped model used --
# images.cocodataset.org is unreachable from some networks:
#   curl -sSL -o coco128.zip \
#     https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
#   unzip -q coco128.zip
CALIB_GLOB=${CALIB_GLOB:-coco128/images/train2017/*.jpg}
CALIB_N=${CALIB_N:-100}

IFS=, read -r CH CW <<<"$IMGSZ"

# 1. PyTorch -> ONNX (no NMS, no head fusion)
yolo export model=yolov8n.pt format=onnx opset=13 imgsz="$CH,$CW" nms=False simplify=True

# 2. Calibration array: N x H x W x 3, float32, 0..1, RGB.
#
#    Eight images (the old coco8 set) is not a calibration set: it fixes the box
#    tensor's scale and every intermediate activation range from almost no data.
#    A hundred or more is cheap and much steadier. Resize the same way inference
#    does -- straight to HxW, no letterbox -- so calibration sees what the DLPU
#    will see. Snapshots in verify/ are mixed in automatically.
#
#    If you want to A/B this export against the previous model, hold the images
#    beyond CALIB_N back: calibrating and scoring on the same images flatters it.
CALIB_GLOB="$CALIB_GLOB" CALIB_N="$CALIB_N" CH="$CH" CW="$CW" python - <<'PYEOF'
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
PYEOF

# 2b. onnx2tf downloads a sample array for its own internal accuracy check, and that
#     release asset 404s on some versions, which aborts the export before it starts.
#     Provide it locally instead. It does not affect quantization -- -cind below is
#     what calibrates the model.
CALIB_GLOB="$CALIB_GLOB" python - <<'PYEOF'
import glob, os, numpy as np, cv2
f = "calibration_image_sample_data_20x128x128x3_float32.npy"
if not os.path.isfile(f):
    fs = sorted(glob.glob(os.environ["CALIB_GLOB"]))[:20]
    np.save(f, np.stack([cv2.resize(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), (128, 128))
                         .astype(np.float32) / 255.0 for x in fs]))
    print("wrote a stand-in for onnx2tf's sample data")
PYEOF

# 3. ONNX -> NHWC uint8 TFLite, CUT BEFORE THE FINAL CONCAT (two separate outputs)
#    -onimc keeps boxes and scores as separate tensors so each gets its own
#    quantization scale. Without it the shared per-tensor scale (~2.54) zeroes
#    every class score.
#    -iqd/-oqd uint8 matches what larod's convert preprocessor hands over, so the
#    ACAP does no conversion at all.
onnx2tf -i yolov8n.onnx -oiqt -qt per-tensor -iqd uint8 -oqd uint8 \
  -onimc /model.22/Mul_2_output_0 /model.22/Sigmoid_output_0 \
  -cind images calib.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]" \
  -o tf_split

cp tf_split/yolov8n_full_integer_quant.tflite model/yolov8n_384x640_uint8.tflite

# 4. Regenerate the committed quantization header (and its model hash), which
#    acap/build.sh checks before every build.
python tools/parameter_finder.py model/yolov8n_384x640_uint8.tflite acap/app/model_params.h

#!/bin/sh
# Build the .eap with Docker. Run from the repo root: sh acap/build.sh
# (exec bit does not stick on exFAT volumes, so use `sh`.)
set -e
cd "$(dirname "$0")/.."

# acap/app/model_params.h is committed rather than regenerated on every build.
# That is only safe while it still describes the committed model.
want=$(sed -n 's/.*MODEL_SHA256 \([0-9a-f]\{64\}\).*/\1/p' acap/app/model_params.h | head -1)
have=$(shasum -a 256 model/yolov8n_384x640_uint8.tflite 2>/dev/null || sha256sum model/yolov8n_384x640_uint8.tflite)
have=${have%% *}
if [ "$want" != "$have" ]; then
    echo "model_params.h describes a different model than model/yolov8n_384x640_uint8.tflite." >&2
    echo "  header: $want" >&2
    echo "  model:  $have" >&2
    echo "Re-run: python tools/parameter_finder.py model/yolov8n_384x640_uint8.tflite" >&2
    exit 1
fi

# Clear out earlier builds first. Leaving them here means acap/ shows a mix of
# versions and the newest file is not obviously the current one.
rm -rf acap/build-arm64
rm -f acap/YOLOv8_Detector_*_aarch64.eap

v=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' acap/app/manifest.json | head -1)
echo "Building version $v from acap/app/manifest.json"

docker build -f acap/Dockerfile --build-arg ARCH=aarch64 -t yolov8_detector:arm64 .
docker cp "$(docker create yolov8_detector:arm64)":/opt/app/. acap/build-arm64
find acap/build-arm64 -maxdepth 1 -name '*.eap' -exec cp {} acap/ \; -print
echo "Done."

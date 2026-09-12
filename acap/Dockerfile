# YOLOv8 detector ACAP for ARTPEC-8 (AXIS OS 12/13), built with the ACAP Native SDK.
#   docker build --build-arg ARCH=aarch64 -t yolov8_detector:arm64 .
#
# Both the model and its quantization header are committed:
#   model/yolov8n_640_int8.tflite  <- tools/export_yolov8.sh
#   acap/app/model_params.h        <- tools/parameter_finder.py, run by that script
# acap/build.sh verifies they still match before calling docker.

ARG ARCH=aarch64
ARG VERSION=12.11.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}
WORKDIR /opt/app
COPY acap/app .
COPY model/yolov8n_640_int8.tflite model/model.tflite
COPY model/labels.txt label/labels.txt

# GOTCHA: environment-setup exports its own ARCH -- capture the build arg first.
ARG ARCH
RUN TARGET_ARCH="$ARCH" && \
    . /opt/axis/acapsdk/environment-setup* && \
    echo "Build TARGET_ARCH=[$TARGET_ARCH]" && \
    acap-build . -a 'label/labels.txt' -a 'model/model.tflite' && \
    ls -la *.eap

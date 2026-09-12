# YOLOv8 on Axis (ARTPEC-8)

Stock COCO YOLOv8n running on an AXIS Q1656 (ARTPEC-8, AXIS OS 12.11, ACAP Native SDK 12.11.0).
The DLPU runs the whole graph; the detection head is decoded on the CPU and drawn as bounding
boxes on the stream. Detections are also published as a camera event, so any VMS that already
consumes Axis events can subscribe without anything extra deployed.

Test camera: 192.168.1.156

## How it works

VDO hands over frames at the native aspect ratio; larod's `convert` preprocessor scales them to
the model input and the model runs on `axis-a8-dlpu-tflite`. The graph is cut before YOLOv8's
final Concat, so boxes and scores arrive as two separate uint8 tensors with their own
quantization scales — `model/MODEL.md` explains why that matters. The ACAP identifies the two by
byte size rather than index, picks the best class per anchor by comparing raw quantized bytes,
drops anchors below the confidence threshold and anchors covering more of the frame than
`MaxAreaPercent`, and runs greedy per-class NMS over what survives.

Decode runs on a worker thread: frame N is decoded while frame N+1 is on the DLPU, so the loop
is inference-bound. The cost is one frame of lag between the video and the overlay.

Every setting is an axparameter that the app watches, so saving in the settings page takes
effect on the next frame — no restart, no model reload.

## Settings

| parameter | default | what it does |
|---|---|---|
| `ConfThresholdPercent` | 25 | minimum class score, 1..100 |
| `IouThresholdPercent` | 45 | NMS overlap threshold |
| `MaxAreaPercent` | 90 | reject boxes covering more of the frame than this |
| `Classes` | `person` | comma-separated label names; blank means all 80 |
| `EventsEnabled` | yes | publish detections as a camera event |
| `EventMinDurationMs` | 1000 | how long a class must be present before it is reported |
| `EventCooldownMs` | 30000 | minimum gap between two events for the same class |
| `LiveView` | no | set by the settings page while it is open; gates `live.json` |

`ConfThresholdPercent` has a floor of 1 % and NMS never compares more than 300 boxes, because at
a zero threshold every one of the 8400 anchors survives and the O(n²) suppression would stall
the app.

The event is stateless, on
`tnsaxis:CameraApplicationPlatform/tnsaxis:YOLOv8Detector/tnsaxis:Detection`, carrying `class`
(string) and `confidence` (double).

## Build and deploy

```sh
sh acap/build.sh                     # -> acap/YOLOv8_Detector_0_9_2_aarch64.eap
curl --digest -u root:PASS -F "packfil=@acap/YOLOv8_Detector_0_9_2_aarch64.eap" \
  "http://CAMERA/axis-cgi/applications/upload.cgi"
curl --digest -u root:PASS "http://CAMERA/axis-cgi/applications/control.cgi?action=start&package=yolov8_detector"
```

`runMode` is `never`, so the app is started explicitly — including after a camera reboot. If you
want it to come back on its own, change that to `respawn` in `acap/app/manifest.json`.

## Layout

- `tools/export_yolov8.sh` — model export, off-camera; also regenerates `model_params.h`
- `tools/parameter_finder.py` — reads the model's quantization parameters into that header
- `model/` — `.tflite`, `labels.txt`, `MODEL.md` (tensor contract + the quantization trap)
- `acap/` — native ACAP sources, Dockerfile, build script
- `verify/` — snapshots used to check box geometry
- `improvements.md` — review notes and what is left

## Things that will bite

- **AXIS Object Analytics holds the DLPU.** The manifest declares
  `deepLearningProcessor.required`, so AOA must be stopped before this app starts.
- First start takes ~60 s while larod compiles the model for the DLPU. It is not hung.
- Per-frame timings and per-object detections log at `LOG_DEBUG`. At `LOG_INFO` the app only
  reports state changes; turn debug on when tuning, not in production.
- `acap/app/model_params.h` is committed rather than generated on every build. `acap/build.sh`
  refuses to build if it no longer matches the model's sha256 — re-run the export script.
- The Docker base image is amd64; on Apple Silicon the build runs under emulation and warns
  about the platform mismatch. That is expected and harmless.

## Attribution

Derived from Axis's `object-detection-yolov5` example (Apache 2.0). The decode path is rewritten
for YOLOv8's channel-major, objectness-free, two-tensor output.

Pavel Kotyza <kotyza@gmail.com>

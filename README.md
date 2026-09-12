# YOLOv8 on AXIS — ACAP

![status: prototype](https://img.shields.io/badge/status-PROTOTYPE-orange)
![licence: AGPL-3.0](https://img.shields.io/badge/licence-AGPL--3.0-blue)
![hardware: ARTPEC-8](https://img.shields.io/badge/hardware-ARTPEC--8-informational)
![AXIS OS 12/13](https://img.shields.io/badge/AXIS%20OS-12%20%2F%2013-informational)

**Stock COCO YOLOv8n running on the camera's own DLPU. 80 classes, no server, no cloud.**

A functional edge prototype: an ACAP that loads a quantized YOLOv8n into larod on an AXIS Q1656,
decodes the detection head on the camera's CPU, draws bounding boxes on the live stream, and
publishes detections as a native Axis camera event that any VMS can subscribe to.

<img width="1600" height="893" alt="hero" src="https://github.com/user-attachments/assets/dd1ca48e-d12b-454c-88ce-4ffa7397803d" />


This is a weekend project — me teaching an Axis camera a new trick, and pushing the envelope to
find where it breaks. It is not a product and it is not supported. See
[**What this is not**](#what-this-is-not) before you deploy it anywhere that matters.

**[Read the write-up &rarr;](https://kotyzap.github.io/YOLOv8-on-AXIS-ACAP/)** — the measurements, the aspect-ratio finding and
the quantization trap, with the tables.

---

## What actually got measured

Everything below is a number from this repo, not an estimate.

| | |
|---|---|
| Inference, 640x640 model, `axis-a8-dlpu-tflite` | **54 ms/frame** |
| Whole graph on the DLPU | yes — including the DFL head |
| CPU decode + NMS, 8400 anchors x 80 classes | 14 ms |
| End-to-end, before pipelining | ~80 ms → ~12.5 fps |

### The aspect-ratio finding

The interesting result. larod's `convert` preprocessor scales the frame to the model input
**without letterboxing**, so a square model sees a 16:9 scene squashed to about 56 % of its
width — nothing like the photos COCO was trained on.

Re-exported at 384x640 and scored against the float32 PyTorch model, on 28 images held out of
both models' calibration sets, decoded exactly as the ACAP decodes:

**Frames cropped to 16:9 — what the camera actually delivers:**

| model | input | recall | false positives |
|---|---|---|---|
| square | 640x640 | 57.5 % | 19 |
| **rectangular** | **384x640** | **87.6 %** | **11** |

**The same images uncropped — ordinary ~4:3 photos:**

| model | input | recall | false positives |
|---|---|---|---|
| **square** | **640x640** | **65.3 %** | 16 |
| rectangular | 384x640 | 51.6 % | 9 |

Thirty points of recall, in both directions, from nothing but aspect ratio. The lesson is not
"rectangular is better" — it is that **the model's aspect ratio has to match the sensor's**, and
that getting it wrong costs more than most of the tuning anyone bothers with. The rectangular
model also uses 40 % fewer input pixels.

### The quantization trap

Worth knowing if you ever quantize a YOLOv8 yourself. Ultralytics' single `output0`
(1 x 84 x anchors) concatenates box coordinates, which range 0..~650, with class scores, which
range 0..1. Per-tensor int8 quantization picks **one scale for both**, around 2.54. Every class
score below ~1.27 rounds to zero, and the exported model detects nothing at all — silently, with
no error anywhere.

The fix is to cut the graph before the final `Concat` so boxes and scores become two tensors with
their own scales:

| export | max class score on the same image |
|---|---|
| float32 reference | 0.665 |
| single-output int8 | **0.000** |
| split-output int8 | 0.625 |

`model/MODEL.md` has the full tensor contract and the export recipe.

---

## Which cameras

The DLPU model format is not the same across Axis SoCs, so this does not travel as far as it
looks.

| SoC | DLPU model format | this repo |
|---|---|---|
| **ARTPEC-8** | TFLite int8 | **verified** — AXIS Q1656, AXIS OS 12.11 |
| ARTPEC-9 | TFLite int8 | should work, untested — see below |
| ARTPEC-7 | TFLite int8 | untested; a much weaker TPU/GPU, YOLOv8n at 384x640 is probably too heavy |
| CV25 | proprietary Ambarella CVflow (`.bin`) | **no** — different artefact, different toolchain |
| CV75 | proprietary, from ONNX | **no** |

`runOptions` names `axis-a8-dlpu-tflite`. If that device is not present, the app now enumerates
what larod does offer, picks a device whose name contains `dlpu`, logs the substitution and
carries on — so an ARTPEC-9 product should load the same `.tflite` without a rebuild. That path
has never run on real ARTPEC-9 hardware; if you have one, I would like to hear how it goes.

CV25 and CV75 are not a configuration problem. They take a proprietary format converted through
Ambarella's toolchain, so supporting them means a second model artefact and re-verifying that the
split-output trick and the channel-major decode survive that conversion. Axis ships separate
`object-detection-cv25` and `tensorflow-to-larod-cv25` examples for exactly this reason.

## How it works

VDO delivers frames at the sensor's native aspect ratio. larod's `convert` preprocessor scales
them to the model input, and the model runs on `axis-a8-dlpu-tflite`. The graph is cut before
YOLOv8's final Concat, so boxes and scores arrive as two separate uint8 tensors — the ACAP tells
them apart by byte size rather than index, because larod does not guarantee output order.

Decoding runs on a worker thread: frame N is decoded while frame N+1 is on the DLPU, so the loop
is inference-bound rather than inference-plus-decode. The cost is one frame of lag between the
video and the overlay, which is not visible at these frame rates. The decode picks the best class
per anchor by comparing raw quantized bytes, drops anchors below the confidence threshold and
anchors covering more of the frame than `MaxAreaPercent`, and runs greedy per-class NMS over what
survives.

Every setting is an axparameter the app watches, so saving in the settings page takes effect on
the next frame — no restart, no model reload.

## Settings

| parameter | default | what it does |
|---|---|---|
| `ConfThresholdPercent` | 25 | minimum class score, 1..100 |
| `IouThresholdPercent` | 45 | NMS overlap threshold |
| `MaxAreaPercent` | 90 | reject boxes covering more of the frame than this |
| `Classes` | `person` | comma-separated COCO label names; blank means all 80 |
| `EventsEnabled` | yes | publish detections as a camera event |
| `EventMinDurationMs` | 1000 | how long a class must be present before it is reported |
| `EventCooldownMs` | 30000 | minimum gap between two events for the same class |
| `LiveView` | 0 | nonce written by the settings page while it is open |

`ConfThresholdPercent` has a floor of 1 % and NMS never compares more than 300 boxes: at a zero
threshold every one of the anchors survives and the O(n²) suppression stalls the app.

The camera event is stateless, on
`tnsaxis:CameraApplicationPlatform/tnsaxis:YOLOv8Detector/tnsaxis:Detection`, carrying `class`
(string) and `confidence` (double). Any VMS that already consumes Axis events — AXIS Camera
Station, Genetec, Milestone — can subscribe with nothing extra deployed.

## Build and deploy

Needs Docker and the ACAP Native SDK image.

```sh
git clone https://github.com/kotyzap/YOLOv8-on-AXIS-ACAP
cd YOLOv8-on-AXIS-ACAP
sh acap/build.sh                     # -> acap/YOLOv8_Detector_0_9_5_aarch64.eap

curl --digest -u root:PASS -F "packfil=@acap/YOLOv8_Detector_0_9_5_aarch64.eap" \
  "http://CAMERA/axis-cgi/applications/upload.cgi"
curl --digest -u root:PASS \
  "http://CAMERA/axis-cgi/applications/control.cgi?action=start&package=yolov8_detector"
```

`runMode` is `respawn`, so it starts on its own and survives a reboot.

To re-export the model — different input size, different calibration set, your own weights:

```sh
sh tools/export_yolov8.sh            # regenerates the .tflite and acap/app/model_params.h
```

`acap/build.sh` refuses to build if `model_params.h` and the model file have drifted apart.

## Layout

- `acap/` — native ACAP sources, Dockerfile, build script
- `model/` — the `.tflite`, labels, and `MODEL.md` (tensor contract, quantization trap, measurements)
- `tools/` — model export and the quantization-parameter extractor
- `verify/` — snapshots and an on-camera check script
- `improvements.md` — a full code review of this repo and what came of it

## What this is not

Read this part.

- **It takes the DLPU exclusively.** The manifest declares `deepLearningProcessor.required`, so
  **AXIS Object Analytics must be stopped** before this app will start. You are trading AOA for
  this, not adding it.
- **COCO is a photo dataset, not a surveillance dataset.** It was trained on hand-held pictures,
  not on a camera mounted at 4 m looking down a car park in the rain at 2 a.m. AOA is trained for
  that and will beat this comfortably on the classes it covers. The aspect-ratio numbers above
  are exactly how much a domain mismatch can cost — and scene domain is a bigger mismatch than
  aspect ratio.
- **Most of the 80 classes are useless on a camera.** COCO includes toaster, hair drier and
  broccoli. Perhaps 15 of the 80 will ever fire meaningfully in a surveillance scene.
- **No tracking, no scenarios, no counting, no calibration.** Detections are per-frame. There is
  no line crossing, no time-in-area, no object ID across frames.
- **Not performance-tuned on hardware for the current model.** The 384x640 model's inference time
  on the camera has not been measured yet; only the 640x640 one has.
- **Tested on exactly one camera**, an AXIS Q1656 on AXIS OS 12.11.

The point of the project is the pipeline, not the COCO class list: an arbitrary detector,
quantized, on the camera itself, with a settings UI and VMS events. Swapping in a model trained
on something you actually care about is the interesting direction.

## Licence

**AGPL-3.0.** Not a stylistic choice: this repo ships a model derived from Ultralytics YOLOv8,
which is AGPL-3.0, and Ultralytics reads that as covering the whole derivative work. If you want
to build something closed on top of this, you need either an Ultralytics Enterprise licence or a
different model — several good detectors are Apache-2.0.

I am not a lawyer and this paragraph is not legal advice.

### Third-party

- The ACAP skeleton is derived from Axis's `object-detection-yolov5` example, Apache-2.0. Files
  that are still substantially Axis's keep their original Apache headers unchanged; files I
  rewrote or added carry mine. Apache-2.0 is one-way compatible into AGPL-3.0, which is why the
  combined work can be AGPL-3.0.
- `acap/app/LICENSE` is the licence that ships inside the `.eap` and is shown in the camera's
  Apps page. It leads with AGPL-3.0 and this application's copyright, then reproduces the
  Apache-2.0 and Ultralytics notices. It previously still carried Axis's Apache header verbatim,
  which made the app look like an Axis product on the camera — thanks to Christoph Acs for
  catching that.
- The decode path, the threaded pipeline, the event sender, the settings page and the export
  toolchain are mine.
- `model/yolov8n_384x640_uint8.tflite` is stock COCO YOLOv8n, re-exported. AGPL-3.0, Ultralytics.

## Credits

Built by [Pavel Kotyza](https://www.4xs.dev) — weekend prototyping on Axis cameras.
Claude Code wrote most of the lines; the domain judgement, the hardware and the debugging are mine.

---

**My personal after-hours vibe coded [4XS.dev](https://www.4xs.dev) experiments are not products,
and not affiliated with or endorsed by Axis Communications.**

*Not affiliated with, endorsed by or supported by CamStreamer s.r.o. either. AXIS, ARTPEC and
AXIS Object Analytics are trademarks of Axis Communications AB; CamStreamer, CamOverlay,
CamSwitcher and CamScripter are trademarks of CamStreamer s.r.o. Used here only to identify the
hardware and software this project runs on. No warranty. Use at your own risk.*

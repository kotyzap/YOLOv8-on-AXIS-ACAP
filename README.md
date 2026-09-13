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

<img width="2816" height="1536" alt="Gemini_Generated_Image_wy4ahfwy4ahfwy4a" src="https://github.com/user-attachments/assets/1f49b578-2c43-40d6-9c37-8d64f204647b" />


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
| ARTPEC-9 | TFLite int8 | **runs on ARTPEC-9 — first trial**; slower, caveats below |
| ARTPEC-7 | TFLite int8 | untested; a much weaker TPU/GPU, YOLOv8n at 384x640 is probably too heavy |
| CV25 | proprietary Ambarella CVflow (`.bin`) | **no** — different artefact, different toolchain |
| CV75 | proprietary, from ONNX | **no** |

`runOptions` names `axis-a8-dlpu-tflite`. If that device is not present, the app enumerates what
larod does offer and picks a DLPU device **whose name also contains `tflite`**, logs the
substitution and carries on.

That last detail matters: an ARTPEC-9 product offers both `a9-dlpu-native` and `a9-dlpu-tflite`,
and an earlier version took the first name containing `dlpu` — the native one, which rejected the
model with `Incorrect model format` and, under `runMode: respawn`, restarted every five seconds
forever.

### What an ARTPEC-9 run actually looked like

One evening on an AXIS Q6358. Not a verification pass — a first trial, reported as such.

- **It loads and detects.** Same `.tflite`, no rebuild, whole graph on `a9-dlpu-tflite`.
- **Compiling the model takes ~90 s**, against ~60 s on the Q1656.
- **The first load after every start fails** with `Could not run warmup job: Failure when invoking
  interpreter`, and the next attempt succeeds. Observed four times in a row. The app now retries
  in place rather than relying on `respawn` to paper over it.
- **It is slower.** Mean analysis time 80–113 ms, so the framerate governor settled at 5–10 fps,
  against ~12.5 fps on ARTPEC-8. Part of that is a larger stream (1024x576 vs 800x450); the rest
  is unmeasured.
- **VDO numbers channels differently.** The overview is channel 1, not 0, and view areas follow
  from 2. `ViewArea` is passed straight through to VDO, so a view area chosen on the settings page
  may not be the one you meant on this generation. Untested, and the reason ARTPEC-9 is "runs"
  rather than "verified".
- **Detection quality on a distant night scene was poor at a low threshold** — at confidence 7
  with all 80 classes, most of COCO fired at once. At confidence 61 with five classes it was
  sensible. That is the scene-domain point below, not an ARTPEC-9 problem.

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
| `ViewArea` | 0 | which view to detect on; 0 is the sensor's full view |
| `BoxColours` | `class` | `class` colours boxes by group, `single` draws them all red |
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

## View areas

`ViewArea` picks the view the detector runs on. The same number selects the VDO channel the
frames come from **and** the bbox view the boxes are drawn on, so the picture and the detections
cannot describe different crops. `0` keeps the sensor's full view.

A view area is a crop, so the detector sees only that region — at better effective resolution,
which is the point: a gate or a doorway fills the model input instead of occupying a tenth of it.
AXIS Object Analytics always runs on the full view, so this is a genuine difference rather than a
reimplementation.

Verified on a Q1656 with two view areas: VAPIX view area 2 = VDO channel 2 = bbox view 2, and the
settings page streams the same one. A `ViewArea` the camera does not have, or one that exists but
will not stream, falls back to the full view with a warning rather than exiting — under
`runMode: respawn` a fatal error there is a restart loop the settings page cannot reach.

## Box colours

`bbox` draws rectangles and no text, so colour is all a box can say about itself: person green,
the eight vehicles blue, the ten animals purple, everything else amber. The settings page uses
the same four on the class icons, the group headers and the live confidence bars, so a box on the
video and a row in the list are tied by colour. `BoxColours=single` goes back to one red for
anyone who finds that noisier than useful.

Labels on the boxes would need `axoverlay` and a Cairo render per frame. Not built — the colours
carry most of the same information for a fraction of the cost.

## Build and deploy

An unsigned `.eap` is attached to each [release](https://github.com/kotyzap/YOLOv8-on-AXIS-ACAP/releases).
It takes the DLPU exclusively, so **AXIS Object Analytics has to be stopped first**, and it has
been run on exactly one camera model. Building it yourself needs Docker and the ACAP Native SDK
image.

```sh
git clone https://github.com/kotyzap/YOLOv8-on-AXIS-ACAP
cd YOLOv8-on-AXIS-ACAP
sh acap/build.sh                     # -> acap/YOLOv8_Detector_0_10_14_aarch64.eap

curl --digest -u root:PASS -F "packfil=@acap/YOLOv8_Detector_0_10_14_aarch64.eap" \
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
- **Rotation 0 and 180 only.** Tested at 90: a quarter-turned stream is portrait, larod squashes
  it into the landscape 384x640 input, and a standing person arrives as a smear — one frame found
  nothing, the next called a person an airplane at 26 %. The app warns at 90/270. Fixing it means
  a portrait export chosen at startup, not a coordinate change.
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

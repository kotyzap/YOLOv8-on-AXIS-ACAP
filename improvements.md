# YOLOv8 on Axis — code review and improvements

> **Status, 2026-09-12.** Everything below is implemented and committed (`b879d1f`, app version
> 0.9.2) except the two items that need work only you can run:
>
> - **§2.1 / §2.2 rectangular export and real calibration set** — `tools/export_yolov8.sh` now
>   defaults to `384x640` with a few hundred calibration images, and the ACAP is already
>   dimension-agnostic, but the model itself has not been re-exported. The shipped
>   `model/yolov8n_640_int8.tflite` is still the square, 8-image-calibrated one.
> - **§3.1 `runMode: never`** — still `never`; documented in the README as a deliberate choice
>   with the one-line change to `respawn` if you want it to survive a reboot. Please confirm
>   which you want.
>
> Not verified on hardware: this session could reach neither Docker nor 192.168.1.156, so the
> build and the camera test are yours to run. What *was* verified: both changed C files compile
> clean under the project's `-Werror` flag set, and the rewritten decode produces detections
> identical to an independent NumPy reference over the real model's output tensors across 12
> threshold / IoU / area / class-filter combinations, including a saturated scene that exercises
> the new 300-candidate NMS cap.
>
> The original review follows, unchanged, as the record of why each change exists.

---

Reviewed 2026-09-12 against `acap/app/*` (0.9.1), `acap/Dockerfile`, `acap/build.sh`, `tools/export_yolov8.sh`,
`model/MODEL.md`, `acap/app/html/index.html`, `README.md`.

Overall: the port is solid. The YOLOv8 decode (channel-major, two tensors identified by size, strongest-first NMS
over survivors only, full-frame anchor rejection) is correct and clearly commented. Live axparameter reload, the
custom event with min-duration/cooldown, and the settings page are all beyond what the Axis example gives you.
What follows is ordered by how much it matters, not by file. Items marked **verify** are things I could not
confirm from the source alone and you should check on the Q1656 before acting.

---

## 1. Fix first

### 1.1 `ConfThresholdPercent=0` makes a frame take seconds
`manifest.json` allows `min=0`, and `on_parameter_changed` applies `atoi(value)/100` without clamping. At 0 every
one of the 8400 anchors survives the confidence test, so NMS runs 8400²/2 ≈ 35 M IoU computations on the CPU per
frame — seconds, not 14 ms — and the app is effectively wedged until the user saves a sane value. Same for very low
values on cluttered scenes.

Fix (pick one, both is fine):
- clamp `conf_threshold` to ≥ 0.01 in both the startup read and the callback (or set `min=1` in the manifest and
  the slider);
- cap NMS input the way Ultralytics does (`max_det`-style): after `qsort`, only consider the top N (e.g. 300)
  ranked entries and mark the rest invalid. One line in `non_maximum_suppression`, bounds the worst case forever.

### 1.2 Per-frame syslog flood
At 12.5 fps the app writes at least four `LOG_INFO` lines per frame ("Ran pre-processing", "Ran inference",
"Ran parsing", the raw box dump per object) plus the `vdo_map_dump`. That is ~50–100 syslog lines/s continuously,
which costs CPU, rotates the camera's log so fast that other apps' messages disappear, and makes the server report
useless. Move the per-frame lines to `LOG_DEBUG` (or behind a `Verbose` axparameter), and keep `LOG_INFO` for state
changes only (start, model loaded, settings changed, framerate change, event sent).

### 1.3 The reported fps is not the real fps
`live.json` reports `1000 / (inference_ms + post_processing_ms)`. That excludes the poll wait, the tensor copies,
bbox commit and the JSON build, so it is a theoretical maximum, not what the stream delivers. Measure wall-clock
time between consecutive successful frames (`g_get_monotonic_time()` at loop top) and report that. The README's
"~80 ms → ~12.5 fps" line should be re-measured the same way.

### 1.4 SIGTERM is ignored while no frames arrive
`poll(&fds, 1, -1)` is retried on `EINTR` without looking at `running`. If the stream is stalled (rotation change,
sensor reconfiguration) the app cannot be stopped cleanly. Either check `running` inside the retry loop or poll
with a timeout (e.g. 1000 ms) and `continue` on 0.

### 1.5 Off-by-one in `model_get_tensor_output_info`
`if (tensor_output_index > provider->num_outputs)` should be `>=`. Inherited from the Axis example; harmless
today because the caller iterates `0..num_outputs-1`, but it is a latent out-of-bounds read.

---

## 2. Worth doing — accuracy

### 2.1 The model sees a horizontally squashed frame (**verify**, then decide)
`channel_util_choose_stream_resolution` asks VDO for the smallest *native-aspect* resolution that fits 640×640
(on a 16:9 sensor that is something like 1280×720), and `model_preprocessing` then scales it to 640×640 with no
crop map. So the DLPU gets the full frame anamorphically compressed to ~56 % width. Box geometry still maps back
correctly (you verified that), but people look thin and small horizontal objects lose detail; the stock COCO
weights were trained on letterboxed input.

Confirm with `syslog` ("Use preprocessing with input size WxH and output size 640x640") — if input is 16:9 the
squash is real. Then the cleanest fix is a **rectangular export**: `yolo export ... imgsz=384,640` (both multiples
of 32, 1.67:1 ≈ 16:9). It removes most of the distortion *and* drops input pixels by 40 %, so inference should fall
from 54 ms to roughly 35 ms. `parameter_finder.py` already reads H and W separately; `MODEL_INPUT_WIDTH/HEIGHT`
flow through `box_at()` normalisation unchanged. The alternative — `image.fit=crop` to a 1:1 centre crop — keeps
aspect but throws away the sides of the frame, which is worse for a surveillance camera.

### 2.2 Eight calibration images is too few
`export_yolov8.sh` calibrates int8 on `datasets/coco8` (8 images). Per-tensor quantization scale for the score
tensor is fine (0..1 sigmoid), but the box tensor scale (2.58 px per step) and every intermediate activation are
set by those 8 images. Use 200–500 images, ideally a mix of COCO val and snapshots from the actual camera
(`verify/snapshot.jpg` is exactly the kind of thing to include). Cheap to do, measurable gain in confidence
stability. Also letterbox in the calibration step if you keep the square model, or resize to 640×384 if you go
rectangular, so calibration matches inference.

### 2.3 NMS is class-agnostic
`non_maximum_suppression` suppresses any overlapping box regardless of label. Ultralytics' default is per-class.
With the default `Classes=person` it makes no difference; with `person,backpack,handbag,tie` the carried item
is routinely suppressed by the person box. Passing `label[]` in and adding `label[i] != label[j] → continue`
gives per-class behaviour; whether you want that is a product choice (class-agnostic gives fewer duplicate boxes
with different labels). Suggest per-class, matching what users of YOLOv8 expect.

### 2.4 Event hold timer resets on a single missed frame
`event_sender_update` zeroes `above_since_us[c]` the moment a class is absent for one frame. At 12 fps one
flicker restarts the 1000 ms minimum-duration clock, so a person at the edge of the confidence threshold may never
trigger an event. Add a short absence grace (e.g. keep `above_since` until the class has been missing for
~500 ms) — one extra `last_seen_us[]` array.

---

## 3. Worth doing — robustness and operations

### 3.1 `runMode: never` (**verify**)
The README says this is inherited from the example. Check what happens after a camera reboot: if the app does not
come back on its own, a detector that stops after every power cycle is not deployable. `respawn` would also
recover from the "expected vdo error" path in `handle_vdo_failed`, which currently exits the process (with status 0)
on things like a global-rotation change and then stays down.

### 3.2 `live.json` is written to flash
`live_write` writes into `/usr/local/packages/yolov8_detector/html/` — persistent storage — up to twice a second
whenever detections change, i.e. constantly on a busy scene, whether or not anyone has the settings page open. The
"unchanged payload" check does not help because confidences differ every frame. Options, cheapest first:
- write only while the page is being viewed: have the page toggle a `LiveView` axparameter on open/close (or on a
  visibility change) and skip `live_write` when it is off;
- serve it from memory via FastCGI (`httpConfig` in the manifest, `libfcgi` is in the SDK) — proper fix, more code;
- at minimum lower the rate to 1 Hz and round confidences to 2 decimals before comparing, so still scenes really
  do write nothing.

### 3.3 Decode runs serially after inference
The 14 ms CPU decode happens while the DLPU idles, and vice-versa. `larodRunJobAsync` (or a second thread) lets
frame N+1 run on the DLPU while frame N is decoded, so the loop becomes inference-bound (~54 ms → ~18 fps, or
~28 fps with the rectangular model). Only worth it if fps matters to you; it adds a thread and buffer ownership
rules.

### 3.4 Decode itself can be cheaper
If you touch the decode anyway: compare scores in the quantized domain (`threshold_q = conf/scale + zp`, compare
`uint8` directly, dequantize only survivors), and iterate class-outer/anchor-inner so the 8400-byte-stride reads
become sequential. Either roughly halves the 14 ms. Not needed today; noted because it is where the CPU budget goes.

### 3.5 `ax_parameter_get_int` panics on malformed values
A parameter that fails `sscanf` kills the app at start (e.g. someone sets `Classes`-style text into an int via
`param.cgi`). Log and fall back to the manifest default instead. Same for a zero-area union in
`intersection_over_union` (returns NaN, which silently never suppresses — guard `union_area <= 0`).

---

## 4. Build, repo, docs

- **No version control.** The repo has no `.git`, no `.gitignore`, and `acap/build-arm64/` (full build tree
  including the unstripped `debug/` binary) plus the `.eap` are sitting in the source tree. `git init`, ignore
  `acap/build-arm64/`, `*.eap`, `.DS_Store`. Keep `verify/` and `model/*.tflite` tracked (or LFS).
- **`tensorflow/tensorflow:2.21.0` just to read four numbers.** The model is committed, so `model_params.h` can be
  too: run `parameter_finder.py` at the end of `tools/export_yolov8.sh` and commit the header. Build becomes one
  stage, ~2 GB smaller pull, works offline. Tradeoff: header and model can drift if someone re-exports without the
  script — mitigate with a size/sha check in `build.sh`.
- **README is stale**: says the build produces `YOLOv8_Detector_0_1_0_aarch64.eap`; manifest is 0.9.1. The
  "Status" checklist and "Phase B" note read as a work log — fold into a short "How it works" section.
- **MODEL.md contradicts itself**: "Not yet verified" lists ops that the "Verified on hardware" section below says
  are fine, and the tensor table says `int8` while the note at the bottom says the shipped model is `uint8`.
  Update the table to what is actually in `model/yolov8n_640_int8.tflite` (and consider renaming the file).
- `Makefile`: `-lm` appears in both `LDLIBS` and the link line; `$(LIBS)` is never set. Cosmetic.
- `model_params` is `malloc`'d for no reason (it is 8 fields, constant, lives for the whole run) and the three
  per-anchor arrays are VLAs sized by a compile-time constant; make them plain statics. Cosmetic, but it removes
  three `panic` paths.
- Settings page: `poll()` runs every 500 ms *and* the MJPEG stream at 5 fps runs whenever the page is open; that is
  fine, but stop both on `document.hidden` so a tab left open in the background does not keep the encoder and the
  flash writes busy.

---

## 5. Things I looked at and would leave alone

- `g_main_context_iteration(NULL, FALSE)` once per frame is the right way to pump axparameter and axevent
  callbacks in a single-threaded loop; the settings struct is only touched from that thread, so no locking needed.
- Identifying output tensors by byte size rather than index is correct and the panic when it fails is appropriate.
- Full-frame anchor rejection via `MaxAreaPercent` is a pragmatic answer to a real YOLOv8n quirk; keep it.
- `build_class_mask` falling back to "all classes" on no match is the right failure mode for a camera app.
- Event declaration is stateless with `class`/`confidence` as data — correct shape for ACS/Genetec/Milestone.

---

## Suggested order

1. §1.1 clamp / top-k (10 min, prevents a wedge), §1.2 log levels (10 min), §1.3 real fps, §1.4 SIGTERM.
2. §3.1 verify reboot behaviour; §3.2 stop writing `live.json` when nobody is watching.
3. §2.1 confirm the squash in syslog → rectangular 384×640 export with a real calibration set (§2.2). Re-measure.
4. §4 git init + .gitignore, README/MODEL.md cleanup, commit `model_params.h`.
5. §2.3, §2.4, §3.3 as product decisions.

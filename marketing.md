# YOLOv8 on Axis

**Running a modern object detector on an AXIS camera's own AI chip — and the export
recipe that makes it work.**

A 4xs.dev project by Pavel Kotyza.
Status: working prototype, verified on hardware. Version 0.9.1, September 2026.

---

## In one paragraph

YOLOv8n runs entirely on the deep-learning processor of an AXIS Q1656 camera at roughly
12.5 frames per second, with no server, no cloud and no GPU anywhere in the picture. The
camera detects objects, draws them on its own video stream, and raises a standard Axis
camera event that any video management system already knows how to consume. Getting there
required solving a quantization problem that has been an open question in the Axis developer
community, and the solution — along with the measurements that prove it — is the most
reusable part of this work.

---

## The problem this solves

Axis cameras from the ARTPEC-8 generation onward carry a dedicated deep-learning processor.
Axis publishes a worked example for running YOLOv5 on it, supported since AXIS OS 11.7.

For YOLOv8 there is no official example, and the community attempts to produce one have
stalled on the same wall: the exported model is rejected or fragmented by the hardware.
The public discussion on the Axis examples repository ends with reports of the graph
splitting into 137 partitions against a limit of 16, mixed float-and-integer tensors forcing
work back onto the CPU, and an unconfirmed suggestion to try different conversion flags.
No frame-rate figures appear anywhere in that thread.

The gap is not that YOLOv8 is too big for the hardware. It is that the obvious export
produces a model that loads, runs, and detects nothing at all.

---

## What was actually wrong

Three traps, each of which fails silently. Nothing crashes; the detector simply returns
nothing, or returns nonsense, and there is no error message to search for.

### 1. Ultralytics' own TFLite export is the wrong shape

`format=tflite` produces a model in NCHW layout with float32 inputs and outputs. The
ARTPEC-8 processor needs NHWC with integer inputs and outputs. The working path is
ONNX first, then `onnx2tf`.

### 2. One quantization scale destroys every score

This is the real trap, and it is specific to YOLOv8.

YOLOv8 emits a single output tensor of shape 1×84×8400 that **concatenates two
incompatible things**: box coordinates, which range from 0 to roughly 650 in pixels, and
class scores, which range from 0 to 1. Per-tensor integer quantization has to pick one
scale for that whole tensor. It picks 2.5758815, sized for the coordinates. Every score
then rounds to zero.

The numbers, measured on the same image:

| export | highest class score |
|---|---|
| float32 reference | 0.665 |
| whole-graph int8, single output | **0.000** |
| int8 with the head split in two | 0.625 |

The fix is to cut the graph at two named nodes so boxes and scores become separate output
tensors, each with its own scale:

```
onnx2tf -onimc /model.22/Mul_2_output_0 /model.22/Sigmoid_output_0
```

This is also why Axis's YOLOv5 example works with a single tensor and YOLOv8 does not:
YOLOv5 exports coordinates already normalised to 0–1, so a shared scale is harmless.
YOLOv8 exports them in pixels.

### 3. Signed versus unsigned integers

The model must be regenerated with `-iqd uint8 -oqd uint8`. With unsigned 8-bit input at
scale 1/255 and zero point 0, the camera's hardware pre-processing output feeds the model
byte for byte, with no conversion step at all. Get this wrong and the input type simply
does not match what the camera hands over.

---

## What the hardware actually does

Measured on an AXIS Q1656, ARTPEC-8, AXIS OS 12.11, at 640×640 input:

| stage | time |
|---|---|
| pre-processing (hardware scaler and colour conversion) | 12 ms |
| inference on the deep-learning processor | 54 ms |
| decode and overlap suppression on the CPU | 3–14 ms |
| **total** | **~69–80 ms, about 12.5–14.5 frames per second** |

Two findings worth stating plainly, because both contradict the advice you would otherwise
follow:

**The entire graph runs on the deep-learning processor, distribution-focal head included.**
The operations usually feared as CPU-fallback risks — transpose, strided slice, nearest-
neighbour resize, softmax — cost nothing measurable. A planned second phase to cut the head
out of the model and reimplement it on the CPU turned out to be unnecessary and was dropped.

**The decode cost scales with the number of classes, not the number of objects.** Scanning
80 classes across 8400 anchor positions takes 13–17 ms. Narrowed to one class, the same
step takes 3 ms. Choosing which classes matter is therefore both an accuracy decision and
the single cheapest performance lever available.

---

## What the application does

### Detection on the camera
Stock YOLOv8n with the 80 COCO classes, running on the camera's own processor. Bounding
boxes are drawn onto the live video stream by the camera itself.

### Class filtering that costs nothing
Eighty classes is nobody's deployment. Most of COCO is household and kitchen objects that
cannot occur in a camera scene, and leaving them enabled is how a test rig ends up reporting
a toilet and a suitcase in an office. Deselected classes are skipped before overlap
suppression, so they consume no time and can never produce a detection or an event.

The classes are grouped as people, vehicles, animals, street furniture, bags and carried
items, with sports, indoor and food collapsed away as the specialist cases. Presets cover
the realistic deployments in one click: people only, vehicles, or both. A fresh install
detects people.

### A frame-size filter for a specific model artifact
On close, cluttered scenes YOLOv8n emits a stable, frame-filling detection that overlap
suppression cannot remove, because its overlap with the real boxes is near zero, and that
confidence alone cannot separate, because it reaches 0.74. A configurable maximum-area
threshold discards it. This is documented as a policy choice rather than a fix: it also
discards a legitimately frame-filling subject.

### A camera event any VMS already understands
The application declares one event on the camera's own event system:

```
tnsaxis:CameraApplicationPlatform / YOLOv8Detector / Detection
data: class (string), confidence (double)
```

Any system that already consumes Axis events — AXIS Camera Station, Genetec Security
Center, Milestone XProtect — can subscribe to it with nothing additional deployed on the
network. A class fires only once it has held above the confidence threshold for a
configurable minimum duration, and a cooldown suppresses repeats, so a detector running at
12 frames per second does not flood a recording system.

Verified by capturing the camera's ONVIF metadata stream:

```xml
<wsnt:Topic>tnsaxis:CameraApplicationPlatform/YOLOv8Detector/Detection</wsnt:Topic>
<tt:Data>
  <tt:SimpleItem Name="class" Value="person"/>
  <tt:SimpleItem Name="confidence" Value="0.261719"/>
</tt:Data>
```

### A settings page with a live view
A web interface inside the camera, reached from the camera's own application list. It
combines the live video with the detector's own bounding boxes and a decoded list of the
current frame's detections, so an installer can confirm two things at once: that the
detector recognises what was selected, and that the subject is actually in view.

Settings apply while the application runs. Changing a threshold or the class list takes
effect on the next frame, with no restart and no model reload.

---

## Why it matters

**No server in the path.** The detection, the decision and the event all happen inside the
camera. There is nothing to deploy alongside it, nothing to license per channel, and no
video leaving the device for analysis. A network outage does not stop the camera from
detecting; it only delays the notification.

**It speaks a language the existing system already knows.** The output is a standard Axis
camera event, not a proprietary API. Integration work is a subscription, not a project.

**The model is replaceable.** The hard part solved here is the export and decode path for
the YOLOv8 output format, not the specific weights. A model trained on the objects that
matter to a particular site drops into the same pipeline.

Realistic applications: occupancy and presence in retail and public spaces; after-hours
detection of people in restricted areas; vehicle logging at gates and yards; and feeding
detections into a video management system that is already installed.

---

## What it is not

Stated plainly, because a prototype presented as a product is how trust is lost.

- **It is not a product.** It is a working prototype on one camera model, one firmware
  version, and an unsigned application package.
- **Accuracy has not been evaluated on a deployed scene.** Every measurement here comes
  from a desk and a monitor at close range. The timings are solid; the detection quality
  on a real site is untested.
- **There is no tracking and no identity.** Every frame is judged alone. Counting people
  through a doorway, following a vehicle across a scene, or telling the same person from a
  new one all require a tracking layer that does not exist here.
- **There are no zones or line crossings.** The detector reports that a class is present in
  the frame, not where it is allowed to be.
- **It uses stock COCO weights.** Eighty general classes trained on general photographs.
  The model has no idea what any particular site looks like.
- **It claims the deep-learning processor exclusively.** AXIS Object Analytics must be
  stopped before it will run.
- **The frame-filling artifact is contained, not cured.** The area filter removes the worst
  case. The underlying anchor still fires a little smaller, and tightening the threshold
  starts costing real detections.

---

## Technical fact sheet

| | |
|---|---|
| Hardware | AXIS Q1656, ARTPEC-8, aarch64 |
| Firmware | AXIS OS 12.11 |
| Framework | ACAP Native SDK 12.11.0, built in Docker |
| Model | YOLOv8n, 640×640, 3.1 MB, int8 weights with uint8 input and output |
| Inference device | `axis-a8-dlpu-tflite` (the camera's deep-learning processor) |
| Model outputs | boxes 1×4×8400 at scale 2.5758815; scores 1×80×8400 at scale 0.00390625 |
| Throughput | ~12.5 fps with all classes; ~14.5 fps filtered to one class |
| Package size | 2.1 MB |
| Application language | C, using the Axis vdo, larod, bbox, axevent and axparameter libraries |
| Basis | Axis's `object-detection-yolov5` example (Apache 2.0), with the video and inference plumbing unchanged and the output decoding rewritten |
| Licence of the derived work | Apache 2.0 |

**What had to be rewritten for YOLOv8**, given the YOLOv5 starting point: output is
channel-major rather than detection-major; there is no objectness term, so confidence is
the class score alone; boxes arrive in model-input pixels and must be divided by the input
size; and the two output tensors must be identified at runtime by size, because the
inference API does not guarantee their order.

---

## Status and what comes next

**Working and verified on hardware:** the export path, the decode, both filters, the camera
event, the settings page, the live view, and live configuration without a restart.

**Next, in order of value:**

1. Evaluate on a real scene over a full day, including at night.
2. Confirm consumption by a video management system end to end, rather than only proving
   the event leaves the camera.
3. Add tracking, which is what turns presence into counting.
4. Train or select a model for a specific site instead of stock COCO.
5. Sign the application package for distribution.

---

## About

Built by Pavel Kotyza as a 4xs.dev project.

The wider aim is not this particular detector. It is that a camera with a capable processor
inside it should be able to answer a useful question on its own, and hand the answer to the
systems that already exist, without a server in between.

Contact: kotyza@gmail.com · https://www.4xs.dev

---

## Sources for the public context

- Axis community discussion on deploying YOLOv8 to ARTPEC-8 cameras:
  https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/discussions/144
- The related issue thread:
  https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/issues/141
- Axis documentation on deep-learning processor model conversion:
  https://developer.axis.com/computer-vision/computer-vision-on-device/dlpu-model-conversion/
- Axis native SDK examples, including the YOLOv5 object detection example this work derives from:
  https://github.com/AxisCommunications/acap-native-sdk-examples

Statements about the absence of an official YOLOv8 example reflect the public record as of
September 2026 and should be re-checked before publication.

# Dharma — Module 1: Object Detection & Tracking

**Person 1's module.** Answers: *what objects are in the scene, where are
they, what class are they, and which detections belong to the same object
across frames?*

This module does **not** decide whether an object is dangerous (Module 4)
and does **not** plan the vehicle's path (Module 5). It is the first stage
of the pipeline:

```
Camera / Video → [THIS MODULE: YOLO11n + ByteTrack] → Motion Prediction (3)
                                                      → Collision Risk (4)
                                                      → Path Planning (5)
```

## Install

```bash
pip install -r requirements.txt
```

CPU-only works fine for development/testing (confirmed on this build); a
CUDA GPU will speed up real-time inference but isn't required to get
correct output. `yolo11n.pt` weights auto-download from Ultralytics'
GitHub releases on first use (~5.4 MB) and are then cached locally.

## Usage

```python
from perception.detector import ObjectTracker

tracker = ObjectTracker()
objects = tracker.detect_and_track(frame)   # frame: BGR np.ndarray (cv2 style)
```

or, matching the exact calling convention from the project spec:

```python
from perception.detector import detect_and_track
objects = detect_and_track(frame)
```

For a whole video file:

```python
tracker = ObjectTracker()
for objects in tracker.process_video("clip.mp4"):
    ...  # one structured list per frame, in order
```

### Output format (the stable contract with Modules 3/4/5)

```python
[
    {
        "id": 1,                       # persistent tracking ID, or None if not yet tracked
        "class": "car",
        "confidence": 0.94,
        "bbox": [120, 250, 300, 500],  # [x1, y1, x2, y2], pixel coords
        "center": [210, 375],
    },
    ...
]
```

- `id` stays stable for the same physical object across consecutive frames
  of the *same* video/session (courtesy of ByteTrack). Call
  `tracker.reset()` before starting an unrelated video so old IDs don't
  bleed into a new sequence — `process_video()` does this automatically.
- Empty list `[]` if nothing was detected — never `None` or an exception,
  on a valid frame.

## Design choices

- **Detection:** `ultralytics` YOLO11n (COCO-pretrained), used as a
  pretrained model per the constraint of not training from scratch.
- **Tracking:** ByteTrack via `ultralytics`'s built-in `.track()` /
  `bytetrack.yaml` integration, rather than hand-rolling a separate
  ByteTrack implementation — same rationale (proven library, not
  reinventing infra).
- **Class filtering:** by default the tracker only looks for classes
  relevant to a road scene (`car`, `motorcycle`, `bus`, `truck`, `person`,
  `bicycle`, `cow`, `horse`, `dog`, `sheep`) rather than all 80 COCO
  classes, to cut down irrelevant detections. Pass
  `classes_of_interest=None` to `ObjectTracker(...)` to keep all 80 if
  needed later.

## Known coverage gaps (Indian-road classes)

Per the spec's instruction *not* to rename detections into classes the
model doesn't actually support, here's what stock COCO-pretrained YOLO11n
does and doesn't cover, confirmed against the model's actual 80-class list:

| Indian-road need | Status | Notes |
|---|---|---|
| Car / motorcycle / bus / truck / person / bicycle | ✅ Covered | Standard COCO classes, detected reliably in testing. |
| Cattle | ⚠️ Partial | No `cattle` class exists in COCO. We surface the model's real `cow` class as-is rather than relabeling it `cattle`. Goats/buffalo are not a distinct class and may be missed or misclassified. |
| Auto-rickshaw | ❌ Not covered | No dedicated COCO class. Likely to be missed or occasionally picked up as `car` depending on angle. |
| Pushcart | ❌ Not covered | No dedicated COCO class. Not reliably detected. |

**Recommendation for the team:** if auto-rickshaw / pushcart detection is a
hard requirement for the demo scenarios (e.g. "dense market area with mixed
traffic"), we'll need either a different pretrained model with broader
class coverage or a small fine-tuning pass on a handful of labeled
examples — flagging this now rather than faking coverage. The interface
(`detect_and_track`) won't need to change either way, only what's inside
`ObjectTracker`.

## Project structure

```
Dharma/
├── perception/
│   ├── __init__.py
│   └── detector.py          # <- the actual module (ObjectTracker, detect_and_track)
├── tests/
│   └── test_detector.py     # pytest suite, run: python -m pytest tests/ -v
├── assets/
│   ├── make_synthetic_video.py   # regenerates the small test clip below
│   └── synthetic_test.mp4        # short clip used by the ID-persistence test
├── demo.py                  # draws boxes+IDs on a video/image so you can eyeball results
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v
```

7 tests, all passing on this build: output schema validation, correct
classes detected on a real sample image (bus + pedestrians), blank-frame
and `None`-frame edge cases, the module-level `detect_and_track()`
convenience import, ID persistence across frames of a video, and a missing
video-file error case.

## Visual sanity check

```bash
python demo.py --source assets/synthetic_test.mp4
# or point it at your own clip:
python demo.py --source path/to/your_clip.mp4 --out annotated.mp4
```

Draws `id:class confidence` labels + boxes on every frame so you can
visually confirm detection/tracking quality before other modules build on
top of it.

## Handoff notes for Persons 3, 4, 5

- **Person 3 (Motion Prediction):** use `id` + `center`/`bbox` across
  consecutive `detect_and_track()` calls (or `process_video()`) to compute
  velocity/direction per track ID. IDs are stable frame-to-frame within one
  video session.
- **Person 4 (Collision Risk):** use `class`, `center`, `bbox` — this
  module makes no judgment about risk or safety distance.
- **Person 5 (Path Planning):** consumes downstream output from 3/4, not
  this module directly, per the pipeline diagram.
- The output schema above is the stable interface. If it needs to change,
  that's a whole-team decision per the project constraints — ping the
  group before altering keys/types in `detector.py`.

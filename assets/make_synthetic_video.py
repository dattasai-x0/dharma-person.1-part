"""
Generates assets/synthetic_test.mp4 -- a short clip used only for testing
that ByteTrack IDs persist across frames.

We pan a moving crop window across a real image that YOLO can actually
detect things in (ultralytics' bundled bus.jpg, which has a bus + several
pedestrians) rather than drawing synthetic shapes, since YOLO won't
recognize hand-drawn rectangles as "car" or "person". This gives a cheap,
offline, license-free way to test multi-frame tracking identity without
needing a real traffic video.

Run:
    python assets/make_synthetic_video.py
"""

from pathlib import Path

import cv2
import ultralytics

OUT_PATH = Path(__file__).resolve().parent / "synthetic_test.mp4"
SOURCE_IMG = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"


def main(n_frames: int = 15, crop_size: int = 640, pan_step: int = 6, fps: int = 5) -> None:
    img = cv2.imread(str(SOURCE_IMG))
    if img is None:
        raise FileNotFoundError(f"Could not read source image: {SOURCE_IMG}")

    h, w = img.shape[:2]
    writer = cv2.VideoWriter(
        str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_size, crop_size)
    )
    try:
        for i in range(n_frames):
            x0 = min(i * pan_step, w - crop_size)
            crop = img[0:crop_size, x0 : x0 + crop_size]
            writer.write(crop)
    finally:
        writer.release()

    print(f"Wrote {n_frames} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()

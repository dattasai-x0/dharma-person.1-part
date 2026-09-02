"""
Quick visual demo of Module 1: runs detect_and_track on a video/image and
draws bounding boxes + track IDs + class + confidence, so you can eyeball
that detection and tracking look sane before handing output to Persons 3-5.

Usage:
    python demo.py --source path/to/video.mp4 --out annotated.mp4
    python demo.py --source path/to/image.jpg --out annotated.jpg
    python demo.py --source assets/synthetic_test.mp4  # uses default bundled clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from perception.detector import ObjectTracker

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_objects(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    annotated = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        label = f'{obj["id"]}:{obj["class"]} {obj["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2,
        )
    return annotated


def run_on_image(tracker: ObjectTracker, source: Path, out: Path) -> None:
    frame = cv2.imread(str(source))
    objects = tracker.detect_and_track(frame)
    annotated = draw_objects(frame, objects)
    cv2.imwrite(str(out), annotated)
    print(f"{len(objects)} object(s) detected. Saved annotated image to {out}")
    for obj in objects:
        print(" ", obj)


def run_on_video(tracker: ObjectTracker, source: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            objects = tracker.detect_and_track(frame)
            writer.write(draw_objects(frame, objects))
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    print(f"Processed {frame_count} frames. Saved annotated video to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("assets/synthetic_test.mp4"),
        help="input video or image path",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: <source stem>_annotated.<ext>)",
    )
    args = parser.parse_args()

    out = args.out or args.source.with_name(f"{args.source.stem}_annotated{args.source.suffix}")
    tracker = ObjectTracker(verbose=False)

    if args.source.suffix.lower() in IMAGE_EXTS:
        run_on_image(tracker, args.source, out)
    else:
        run_on_video(tracker, args.source, out)


if __name__ == "__main__":
    main()

"""
extract_face_landmarks.py
─────────────────────────
Extract per-frame facial bounding boxes and 5-point keypoints from a video
using InsightFace, and write results to a .txt file.

Output format (one line per frame, space-separated):
    x1 y1 x2 y2  kp0x kp0y  kp1x kp1y  kp2x kp2y  kp3x kp3y  kp4x kp4y

Keypoint order (standard RetinaFace / InsightFace):
    0 - left eye
    1 - right eye
    2 - nose tip
    3 - left mouth corner
    4 - right mouth corner

Frames with no detected face produce an empty line (or a placeholder line
if --placeholder is passed).

Usage:
    python extract_face_landmarks.py --video input.mp4 --output landmarks.txt

    # Keep only the largest face per frame (useful for single-subject videos)
    python extract_face_landmarks.py --video input.mp4 --output landmarks.txt --largest

    # Write -1 placeholders for frames with no face
    python extract_face_landmarks.py --video input.mp4 --output landmarks.txt --placeholder

    # Use GPU (requires onnxruntime-gpu)
    python extract_face_landmarks.py --video input.mp4 --output landmarks.txt --gpu 0
"""

import argparse
import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


# ── helpers ──────────────────────────────────────────────────────────────────

def build_app(gpu_id: int = -1) -> FaceAnalysis:
    """Initialise InsightFace FaceAnalysis with detection + keypoint models."""
    ctx_id = gpu_id  # -1 → CPU, 0/1/… → GPU device index
    app = FaceAnalysis(
        name="buffalo_l",          # recommended model pack; downloads automatically
        # we only need bbox + kps, skip recognition
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def largest_face(faces):
    """Return the face with the biggest bounding-box area."""
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)
    return max(faces, key=area)


def face_to_line(face) -> str:
    """Convert a single Face object to a space-separated string of 14 numbers."""
    x1, y1, x2, y2 = face.bbox.round().astype(int)          # float32 array, convert to int
    kps = face.kps.round().astype(int)                       # shape (5, 2), float32 array (convert to int)

    values = [x1, y1, x2, y2] + kps.flatten().tolist()
    return ",".join(str(v) for v in values)


def placeholder_line() -> str:
    return " ".join(["-1"] * 14)


# ── main ─────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_path: str,
    keep_largest: bool = True,
    use_placeholder: bool = False,
    gpu_id: int = -1,
) -> None:
    print(f"Loading InsightFace model …")
    app = build_app(gpu_id=gpu_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.2f} fps")

    lines = []
    frame_idx = 0
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # InsightFace expects BGR (same as OpenCV) — no conversion needed
        faces = app.get(frame)

        if not faces:
            no_face_count += 1
            lines.append(placeholder_line() if use_placeholder else "")
        else:
            face = largest_face(faces) if keep_largest else faces[0]
            lines.append(face_to_line(face))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames …")

    cap.release()

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nDone.")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Frames with face : {frame_idx - no_face_count}")
    print(f"  Frames no face   : {no_face_count}")
    print(f"  Output written to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract face bbox + 5-pt keypoints from video")
    p.add_argument("--video",       required=True,
                   help="Path to input video file")
    p.add_argument("--output",      required=True,
                   help="Path to output .txt file")
    p.add_argument("--largest",     action="store_true",
                   help="Keep only the largest face per frame (default: first detected)")
    p.add_argument("--placeholder", action="store_true",
                   help="Write '-1' placeholders for frames with no detected face")
    p.add_argument("--gpu",         type=int, default=-1,
                   help="GPU device id (-1 = CPU, 0 = first GPU)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.video,
        output_path=args.output,
        keep_largest=args.largest,
        use_placeholder=args.placeholder,
        gpu_id=args.gpu,
    )

from insightface import app
from insightface.app import FaceAnalysis
import cv2
import sys
import os


def build_app(gpu_id: int = -1, size=(512, 512)) -> FaceAnalysis:
    """Initialise InsightFace FaceAnalysis with detection + keypoint models."""
    ctx_id = gpu_id  # -1 → CPU, 0/1/… → GPU device index
    app = FaceAnalysis(
        name="buffalo_l",          # recommended model pack; downloads automatically
        # we only need bbox + kps, skip recognition
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=ctx_id, det_size=size)
    return app


def crop_video(video_path: str, output_path: str, gpu_id: int = -1, size=(512,512)) -> None:
    print(f"Loading InsightFace model …")
    app = build_app(gpu_id=gpu_id, size=size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.2f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    frame_idx = 0
    out = cv2.VideoWriter(output_path, fourcc, fps, size)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # InsightFace expects BGR (same as OpenCV) — no conversion needed
        faces = app.get(frame)

        if faces:
            face = faces[0]
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped_face = frame[y1:y2, x1:x2]

            out.write(cropped_face)
        else:
            print(f"No face detected in frame {frame_idx}.")
            continue

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames …")

    cap.release()
    if out is not None:
        out.release()

    print(f"\nDone.")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Output written to: {output_path}")    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop faces from a video using InsightFace.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, help="Path to save the cropped video.")
    parser.add_argument("--replace", action="store_true", default=False, help="Replace the original video with the cropped version.")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID to use (default: -1 for CPU).")
    args = parser.parse_args()

    replace_output = args.replace
    if not args.output_path or replace_output or args.video_path == args.output_path:
        replace_output = True
        args.output_path = args.video_path.rsplit(".", 1)[0] + "_cropped.mp4"
    crop_video(args.video_path, args.output_path, gpu_id=args.gpu_id, size=(512, 512))
    if replace_output:
        os.replace(args.output_path, args.video_path)
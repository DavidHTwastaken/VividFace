from insightface import app
from insightface.app import FaceAnalysis
import cv2
import sys
import os
import imageio
import subprocess


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
    scale_factor = 1.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # InsightFace expects BGR (same as OpenCV) — no conversion needed
        faces = app.get(frame)

        if faces:
            face = faces[0]
            x1, y1, x2, y2 = map(int, face.bbox)

            # 1. Calculate current bounding box dimensions and center
            box_w = x2 - x1
            box_h = y2 - y1
            center_x = x1 + box_w // 2
            center_y = y1 + box_h // 2

            # 2. Add a margin factor (e.g., 1.5 = 50% larger, 2.0 = double the size)
            margin_factor = 1.5
            box_size = int(max(box_w, box_h) * margin_factor)
            half_size = box_size // 2

            # 3. Calculate new expanded square coordinates
            new_x1 = center_x - half_size
            new_y1 = center_y - half_size
            new_x2 = center_x + half_size
            new_y2 = center_y + half_size

            # 4. Handle frame boundaries safely
            h, w, _ = frame.shape

            pad_x1 = max(0, -new_x1)
            pad_y1 = max(0, -new_y1)
            pad_x2 = max(0, new_x2 - w)
            pad_y2 = max(0, new_y2 - h)

            crop_x1 = max(0, new_x1)
            crop_y1 = max(0, new_y1)
            crop_x2 = min(w, new_x2)
            crop_y2 = min(h, new_y2)

            cropped_face = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Pad with black pixels if the margin pushes the box past the video edges
            if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
                cropped_face = cv2.copyMakeBorder(
                    cropped_face,
                    pad_y1, pad_y2, pad_x1, pad_x2,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

            resized_face = cv2.resize(cropped_face, size)
            out.write(resized_face)
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

def ffmpeg_crop_video(video_path: str, output_path: str, size=512) -> None:
    # Use ffmpeg to crop the video to the center square and resize to the desired size
    cmd = f"ffmpeg -i {video_path} -vf \"crop='min(iw,ih)':'min(iw,ih)',scale=-2:{size}\" -c:a copy {output_path}"
    subprocess.run(cmd, shell=True, check=True)

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
        args.output_path = args.video_path.split(".")[0] + "_cropped.mp4"
    crop_video(args.video_path, args.output_path, gpu_id=args.gpu_id, size=(512, 512))
    if replace_output:
        os.replace(args.output_path, args.video_path)
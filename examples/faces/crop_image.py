import cv2
import sys
from insightface import app
from insightface.app import FaceAnalysis


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

def crop_image(image_path: str, output_path: str, gpu_id: int = -1, size=(512, 512)) -> None:
    print(f"Loading InsightFace model …")
    app = build_app(gpu_id=gpu_id, size=size)

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")

    faces = app.get(img)
    if not faces:
        print(f"No face detected in image: {image_path}")
        return

    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    cropped_face = img[y1:y2, x1:x2]

    cv2.imwrite(output_path, cropped_face)
    print(f"Cropped face saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop faces from an image using InsightFace.")
    parser.add_argument("image_path", type=str,
                        help="Path to the input image file.")
    parser.add_argument("--output_path", type=str,
                        help="Path to save the cropped image.")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Replace the original image with the cropped version.")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="GPU ID to use (default: -1 for CPU).")
    args = parser.parse_args()

    replace_output = args.replace
    if not args.output_path or replace_output or args.image_path == args.output_path:
        replace_output = True
        args.output_path = args.image_path.rsplit(".", 1)[0] + "_cropped.jpg"
    crop_image(args.image_path, args.output_path,
               gpu_id=args.gpu_id, size=(512, 512))
    if replace_output:
        import os
        os.replace(args.output_path, args.image_path)

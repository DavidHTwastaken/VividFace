import os
import subprocess
import pandas as pd
from infer import run
root = os.path.join('..','diverse-face-dataset')

m = pd.read_csv(os.path.join(root,'map.csv'), header=0)
print(m)
vids_dir = os.path.join('examples','videos')
imgs_dir = os.path.join('examples','faces')

# preprocess each video and image
videos = m['file'][m['is_video'] == 1]
images = m['file'][m['is_video'] == 0]
for v in videos:
    if not os.path.exists(os.path.join(vids_dir,v)):
        print(f"Processing video: {v}")
        subprocess.run(["python", "examples/videos/crop_video.py", os.path.join(root,v), "--output_path", os.path.join(vids_dir,v)])
    landmarks_path = os.path.join(vids_dir,v.replace('.mp4','.txt'))
    if not os.path.exists(landmarks_path):
        print(f"Extracting landmarks for video: {v}")
        subprocess.run(["python", "examples/videos/extract_face_landmarks.py", "--video", os.path.join(root,v), "--output", landmarks_path, "--gpu", "0"])

for img in images:
    if not os.path.exists(os.path.join(imgs_dir,img)):
        print(f"Processing image: {img}")
        subprocess.run(["python", "examples/faces/crop_image.py", os.path.join(root,img), "--output_path", os.path.join(imgs_dir,img), "--gpu", "0"])

# run infer.py on each image with each video; save results in outputs/{image}_{video}
cropped_videos = []
cropped_images = []
for img in images:
    for v in videos:
        cropped_images.append(os.path.join(imgs_dir,img))
        cropped_videos.append(os.path.join(vids_dir,v))
        # subprocess.run(["python", "infer.py", 'examples', "--source", os.path.join(img), "--target", os.path.join(v), "--output", f'{img.split(".")[0]}_{v.split(".")[0]}'])
run(cropped_videos, cropped_images)

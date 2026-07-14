import os
import subprocess
import pandas as pd
from infer import run
from tools.vid_crop import Crop
from tools import load_video, images2video
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='..')
parser.add_argument('--output', type=str, default='single_e2e_output')
parser.add_argument('--replace_crop', action='store_true', help='Replace existing cropped videos and images')
parser.add_argument('--replace_infer', action='store_true', help='Replace existing inference results')
parser.add_argument('--driving_video', type=str, default='01-01-06-02-01-02-08.mp4',
                    help='Path to a specific driving video to use for inference')
parser.add_argument('--source_image', type=str, default='044F18FF002NFO.jpg',
                    help='Path to a specific source image to use for inference')
args = parser.parse_args()
root = os.path.join(args.root, 'diverse-face-dataset')


def same_gender(vid_name: str, img_name: str, image_csv: pd.DataFrame):
    # vid is from RAVDESS, even actor number is female
    vid_is_female = int(vid_name.split('-')[-1].split('.')[0]) % 2 == 0
    # img has a CSV file showing sex of the subject
    img_is_female = image_csv.loc[image_csv['filename']
                                  == img_name, 'sex'].values[0] == 'female'
    # print(vid_name, img_name, 'skipped' if vid_is_female != img_is_female else '')
    return vid_is_female == img_is_female

def get_output_path(video_path, face_path, output_dir):
    short_video_path = os.path.splitext(
        os.path.basename(video_path))[0]
    short_face_path = os.path.splitext(
        os.path.basename(face_path))[0]
    out_dir = os.path.join('outputs', output_dir)
    video_saved_path = os.path.join(out_dir,'videos', f'{short_video_path}--{short_face_path}.mp4')
    return video_saved_path

def pasteback_video(driving_video_path, output_vid_path, cropper):
    vid_path = driving_video_path
    output_frames = load_video(output_vid_path)
    original_frames = load_video(vid_path)
    masks, target_M_c2o_lst = cropper.read_files(args.driving_video, vids_dir)
    pasteback_frames = cropper.pasteback(
        original_frames, output_frames, masks, target_M_c2o_lst)
    output_fps = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)
    wfp = output_vid_path.replace('.mp4', '_pasteback.mp4')
    images2video(pasteback_frames, wfp=wfp, fps=output_fps)

# m = pd.read_csv(os.path.join(root,'map.csv'), header=0)
# print(m)
vids_dir = os.path.join('examples', 'videos')
imgs_dir = os.path.join('examples', 'faces')

# preprocess each video and image
# videos = m['file'][m['is_video'] == 1]
# images = m['file'][m['is_video'] == 0]
vid_data_dir = os.path.join(root, 'targets')
# videos = list(sorted(os.path.join(vid_data_dir, v) for v in os.listdir(vid_data_dir) if v.endswith('.mp4')))
videos = [os.path.join(vid_data_dir, args.driving_video)]
img_data_dir = os.path.join(root, 'sources')
# images = list(sorted(os.path.join(img_data_dir, i) for i in os.listdir(img_data_dir) if i.lower().endswith('jpg')))
images = [os.path.join(img_data_dir, args.source_image)]

image_csv = pd.read_csv(os.path.join(img_data_dir, 'identities.csv'))

cropper = Crop()
# video_paths = [os.path.join(vids_dir, v) for v in videos]
cropper.crop_videos(videos, vids_dir, replace=args.replace_crop, save_pasteback=True)
# for v in videos:
#     if not os.path.exists(os.path.join(vids_dir,v)):
#         print(f"Processing video: {v}")
#         subprocess.run(["python", "examples/videos/crop_video.py", os.path.join(root,v), "--output_path", os.path.join(vids_dir,v)])
#     landmarks_path = os.path.join(vids_dir,v.replace('.mp4','.txt'))
#     if not os.path.exists(landmarks_path):
#         print(f"Extracting landmarks for video: {v}")
#         subprocess.run(["python", "examples/videos/extract_face_landmarks.py", "--video", os.path.join(root,v), "--output", landmarks_path, "--gpu", "0"])
# image_paths = [os.path.join(imgs_dir, img) for img in images]
cropper.crop_source_images(images, imgs_dir)
# for img in images:
#     if not os.path.exists(os.path.join(imgs_dir,img)):
#         print(f"Processing image: {img}")
#         subprocess.run(["python", "examples/faces/crop_image.py", os.path.join(root,img), "--output_path", os.path.join(imgs_dir,img), "--gpu", "0"])

# run infer.py on each image with each video; save results in outputs/{image}_{video}
cropped_videos = []
cropped_images = []
for img in images:
    img = os.path.basename(img)
    cropped_images.append(os.path.join(imgs_dir, img))
for v in videos:
    v = os.path.basename(v)
    cropped_videos.append(os.path.join(vids_dir, v))
# subprocess.run(["python", "infer.py", 'examples', "--source", os.path.join(img), "--target", os.path.join(v), "--output", f'{img.split(".")[0]}_{v.split(".")[0]}'])
run(cropped_videos, cropped_images, output=args.output, replace=args.replace_infer)
vid_path = os.path.join(vid_data_dir, args.driving_video)
    
output_vid_path = get_output_path(
    cropped_videos[0], cropped_images[0], args.output)
if not os.path.exists(output_vid_path):
    import subprocess
    video_fps = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)
    frames_path = os.path.join('outputs', args.output, 'frames', f'{os.path.splitext(args.driving_video)[0]}--{os.path.splitext(args.source_image)[0]}')
    cmd = [
        'ffmpeg',
        '-y',                    # overwrite output file if it already exists
        '-framerate', str(video_fps),
        '-i', f'{frames_path}/%d.jpg',          # matches 0.jpg, 1.jpg, 2.jpg, ...
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_vid_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

output_frames = load_video(output_vid_path)
original_frames = load_video(vid_path)
masks, target_M_c2o_lst = cropper.read_files(args.driving_video, vids_dir)
pasteback_frames = cropper.pasteback(original_frames, output_frames, masks, target_M_c2o_lst)
output_fps = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)
wfp = output_vid_path.replace('.mp4', '_pasteback.mp4')
images2video(pasteback_frames, wfp=wfp, fps=output_fps)

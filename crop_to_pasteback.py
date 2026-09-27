import os
import subprocess
import pandas as pd
from tools.vid_crop import Crop
import argparse
from tools import load_video, images2video
import cv2
from infer import pasteback_video
from tqdm import tqdm

root = os.path.join('..', 'diverse-face-dataset')
data = os.path.join(root, '250_vids','VividFace')
original_videos = os.path.join(root, 'targets')
work = os.path.join('examples','temp')
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
    video_saved_path = os.path.join(
        out_dir, 'videos', f'{short_video_path}--{short_face_path}.mp4')
    return video_saved_path

cropper = Crop()
# video_paths = [os.path.join(vids_dir, v) for v in videos]
to_crop = []
to_pasteback = []
for v in os.listdir(data):
    if v.endswith('.mp4'):
        original_name = v.split('--')[0]+'.mp4'
        vid = load_video(os.path.join(data, v))
        # if the shape is 512x512, process it, otherwise skip
        print(len(vid), vid[0].shape)
        if vid[0].shape[0] == 512 and vid[0].shape[1] == 512:
            if not os.path.exists(os.path.join(work, original_name)):
                original = os.path.join(original_videos, original_name)
                to_crop.append(original)
            to_pasteback.append(os.path.join(data, v))
cropper.crop_videos(to_crop, work, save_pasteback=True)
# perform pasteback on the identified 512x512 videos
for v in tqdm(to_pasteback, desc='Pasteback videos'):
    original_name = os.path.basename(v).split('--')[0]+'.mp4'
    original = os.path.join(original_videos, original_name)
    driving = os.path.join(work, original_name)
    pasteback_video(original, driving, work, v, cropper)
    wfp_with_audio = os.path.join(os.path.dirname(v), f'{os.path.basename(v).split(".")[0]}_with_audio.mp4')
    cmd = [
        'ffmpeg',
        '-y',
        '-i', f'"{v}"',
        '-i', f'"{original}"',
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        f'"{wfp_with_audio}"'
    ]
    subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.replace(wfp_with_audio, v)
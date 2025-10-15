import random
import torch
from decord import VideoReader
import cv2


def get_sorted_random_integers(a, b, n):
    if a >= b:
        raise ValueError("Lower bound 'a' must be less than upper bound 'b'")
    if n < 0:
        raise ValueError("'n' must be a non-negative integer")
    if n > (b - a):
        raise ValueError("The number of selected integers 'n' cannot exceed the total count in the range")

    random_integers = random.sample(range(a, b), n)
    random_integers.sort()
    return random_integers

def latent_process(latent, bsz=2):

    use_prob = 0.6
    drop_num = 3
    drop_patch_num = 7
    patch_size = [3, 7]

    if random.random() > use_prob:
        return latent

    param_dict = {
        'drop_num': drop_num,
        'drop_patch_num': drop_patch_num,
        'patch_size': patch_size
    }

    def random_drop(this_latent, param_dict):
        drop_num = param_dict['drop_num']
        masked_frame_idxs = []
        single_bz_frames_num = this_latent.shape[0] // bsz
        for b in range(bsz):
            masked_frame_idxs.extend(
                get_sorted_random_integers(single_bz_frames_num * b, single_bz_frames_num * (b+1), drop_num)
            )
        this_latent[masked_frame_idxs] *= 0
        return this_latent

    def mid_drop(this_latent, param_dict):
        masked_frame_idxs = []
        single_bz_frames_num = this_latent.shape[0] // bsz
        masked_frame_idxs = [0, single_bz_frames_num-1, single_bz_frames_num, this_latent.shape[0]-1]
        total_idx = [i for i in range(this_latent.shape[0])]
        used_idx = sorted(
            set(total_idx) - set(masked_frame_idxs)
        )
        this_latent[used_idx] *= 0
        return this_latent

    def random_patch_zeroing(this_latent, param_dict):
        drop_patch_num = param_dict['drop_patch_num']
        patch_size = param_dict['patch_size']

        frames, channels, height, width = this_latent.shape
        for f in range(frames):
            num_patches = random.randint(1, drop_patch_num)
            for _ in range(num_patches):
                patch_height = random.randint(patch_size[0], patch_size[1])
                patch_width = random.randint(patch_size[0], patch_size[1])
                if height - patch_height > 0:
                    y = random.randint(0, height - patch_height)
                else:
                    y = 0
                if width - patch_width > 0:
                    x = random.randint(0, width - patch_width)
                else:
                    x = 0
                this_latent[f, :, y:y+patch_height, x:x+patch_width] *= 0
        return this_latent

    return random.choice([
        random_drop,
        random_drop,
        mid_drop,
    ])(latent, param_dict)


if __name__ == '__main__':
    import decord
    import torch
    import numpy as np
    from moviepy.editor import VideoClip

    def load_video_as_tensor(video_path):
        vr = decord.VideoReader(video_path)
        frames = vr.get_batch(range(len(vr))).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return frames

    def tensor_to_video(tensor):
        tensor = tensor.permute(0, 2, 3, 1).contiguous().numpy()
        return tensor

    def save_video(video_array, save_path, fps=30):
        def make_frame(t):
            frame_idx = int(t * fps)
            if frame_idx >= len(video_array):
                frame_idx = len(video_array) - 1
            return video_array[frame_idx]

        duration = len(video_array) / fps
        clip = VideoClip(make_frame, duration=duration)
        clip.write_videofile(save_path, fps=fps)

    def process_video_with_func(video_path, save_path, func):
        video_tensor = load_video_as_tensor(video_path)

        processed_tensor = func(video_tensor)

        processed_video = tensor_to_video(processed_tensor)

        save_video(processed_video, save_path)

    def example_func(tensor):
        return tensor // 2

    video_path = 'example.mp4'
    save_path = 'output.mp4'
    process_video_with_func(video_path, save_path, latent_process)

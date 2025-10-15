import random
import math
import os

import cv2
import numpy as np
from scipy.ndimage import affine_transform

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps <= 0:
        raise ValueError("Failed to retrieve FPS from video file")

    return fps

def save_videos(video_array1, video_array2, output_path1, output_path2, fps=30):
    if video_array1.shape != video_array2.shape:
        raise ValueError("Input video arrays must have the same shape")

    frames, height, width, channels = video_array1.shape

    # Ensure video_array1 and video_array2 are in RGB format (OpenCV uses BGR by default)
    if video_array1.dtype != np.uint8:
        raise ValueError("Input video arrays must be of type np.uint8")

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer1 = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    video_writer2 = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    # Write frames to video files
    for i in range(frames):
        frame1 = cv2.cvtColor(video_array1[i], cv2.COLOR_RGB2BGR)
        frame2 = cv2.cvtColor(video_array2[i], cv2.COLOR_RGB2BGR)

        video_writer1.write(frame1)
        video_writer2.write(frame2)

    # Release video writers
    video_writer1.release()
    video_writer2.release()

    print(f"Videos saved to {output_path1} and {output_path2} at {fps} FPS")

def load_videos(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open one or both video files")

    frames1 = []
    frames2 = []

    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Convert frames to RGB (OpenCV uses BGR by default)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        frames1.append(frame1)
        frames2.append(frame2)

    cap1.release()
    cap2.release()

    # Convert lists to numpy arrays
    video_array1 = np.array(frames1)
    video_array2 = np.array(frames2)

    return video_array1, video_array2

def list_files_recursive(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

texs_file = list_files_recursive('./OCCdata/textures')
objs_file = list_files_recursive('./OCCdata/temps/objs')
hands_file = list_files_recursive('./OCCdata/temps/hands')

def generate_random_mask(image_shape, num_shapes=10):
    """
    Generate a random irregular mask with a variety of shapes.

    Parameters:
        image_shape (tuple): Shape of the image (height, width).
        num_shapes (int): Number of random shapes to generate in the mask.

    Returns:
        np.ndarray: Random mask with shapes.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_shapes):
        shape_type = random.choice(['ellipse', 'rectangle', 'polygon'])

        # Random parameters
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        size = random.randint(20, 50)  # Radius or size of the shape
        thickness = -1  # Filled shape

        if shape_type == 'ellipse':
            axes = (size, size)
            cv2.ellipse(mask, (center_x, center_y), axes,
                        angle=random.uniform(0, 360),
                        startAngle=0, endAngle=360,
                        color=255, thickness=thickness)

        elif shape_type == 'rectangle':
            x1 = random.randint(max(0, center_x - size), min(width, center_x + size))
            y1 = random.randint(max(0, center_y - size), min(height, center_y + size))
            x2 = random.randint(x1, min(width, x1 + size))
            y2 = random.randint(y1, min(height, y1 + size))
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=thickness)

        elif shape_type == 'polygon':
            num_vertices = random.randint(3, 6)  # Random number of vertices
            vertices = np.array([[random.randint(0, width - 1), random.randint(0, height - 1)] for _ in range(num_vertices)], np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [vertices], color=255)

    return mask

def apply_mask(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Image not found at {img_path}")

    # Generate random mask
    mask = generate_random_mask(image.shape[:2])

    # Apply the mask
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image

def gen_texture_temp():
    this_texture = random.choice(texs_file)
    return apply_mask(this_texture)

def random_color_transform(image):
    """
    Apply random color transformations to the input image including contrast, brightness, saturation, and hue.

    Parameters:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The transformed image.
    """
    # Convert to HSV color space for hue and saturation adjustments
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Random contrast adjustment
    alpha = random.uniform(0.5, 1.5)  # Contrast control (1.0: no change)
    img_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    # Random brightness adjustment
    beta = random.uniform(-50, 50)  # Brightness control
    img_bright = cv2.convertScaleAbs(img_contrast, alpha=1, beta=beta)

    # Convert the brightness-adjusted image to HSV for saturation and hue adjustments
    img_bright_hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)

    # Random saturation adjustment
    saturation_scale = random.uniform(0.5, 1.5)  # Saturation control
    img_bright_hsv[:, :, 1] = np.clip(img_bright_hsv[:, :, 1] * saturation_scale, 0, 255)

    # Random hue adjustment
    hue_shift = random.randint(-10, 10)  # Hue shift in degrees
    img_bright_hsv[:, :, 0] = (img_bright_hsv[:, :, 0] + hue_shift) % 180

    # Convert back to BGR
    img_bright_hsv = np.clip(img_bright_hsv, 0, 255).astype(np.uint8)
    img_transformed = cv2.cvtColor(img_bright_hsv, cv2.COLOR_HSV2BGR)

    return img_transformed

def random_transform(img, target_area):
    """Resize, rotate, and flip an image randomly."""
    # Calculate the new size to achieve target_area
    original_area = img.shape[0] * img.shape[1]
    scale_factor = np.sqrt(target_area / original_area)
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

    # Resize image
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # Random rotation
    angle = random.uniform(0, 360)
    center = (new_size[0] // 2, new_size[1] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(resized_img, rot_mat, new_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Random flip
    flip_code = random.choice([-1, 0, 1])  # -1: both axes, 0: vertical, 1: horizontal
    flipped_img = cv2.flip(rotated_img, flip_code)

    return flipped_img


def apply_moving_mask_to_videos(occ_img, video1, video2=None):

    frames, h, w, c = video1.shape
    mask_h, mask_w, _ = occ_img.shape

    # Calculate new mask size
    new_mask_size = (int(h / (3**0.5)), int(w / (3**0.5)))

    # Resize occ_img to 1/3 of its size
    resized_mask = cv2.resize(occ_img, (new_mask_size[1], new_mask_size[0]), interpolation=cv2.INTER_LINEAR)

    # Create masks for video frames
    masked_video1 = np.copy(video1)
    masked_video2 = np.copy(video2)

    # Initialize the starting position and direction of the mask
    start_y = np.random.randint(0, h - new_mask_size[0])
    start_x = np.random.randint(0, w - new_mask_size[1])
    direction = np.random.uniform(-1, 1, 2)  # Random direction vector

    # Normalize direction
    direction /= np.linalg.norm(direction)

    # Define the step size for mask movement
    step_size = 5  # You can adjust this value to control the speed of movement

    # transformed_resized_mask = random_color_transform(mask)
    resized_mask = np.where(resized_mask > 0, random_color_transform(resized_mask), resized_mask)
    for i in range(frames):
        # Calculate new position based on direction
        start_y = int(np.clip(start_y + direction[0] * step_size, 0, h - new_mask_size[0]))
        start_x = int(np.clip(start_x + direction[1] * step_size, 0, w - new_mask_size[1]))

        # Create a mask with the resized mask
        mask = np.zeros((h, w, c), dtype=np.float32)
        mask[start_y:start_y+new_mask_size[0], start_x:start_x+new_mask_size[1], :] = resized_mask
        # Apply mask to both videos
        masked_video1[i] = np.where(mask > 0, mask, video1[i])
        if video2 is None:
            pass
        else:
            masked_video2[i] = np.where(mask > 0, mask, video2[i])

    return masked_video1, masked_video2


def get_temp():
    this_seed = random.random()
    if this_seed <= 0.25:
        return gen_texture_temp()
    elif 0.25 < this_seed < 0.75:
        return cv2.cvtColor(cv2.imread(
            random.choice(hands_file)
        ), cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(cv2.imread(
            random.choice(objs_file)
        ), cv2.COLOR_BGR2RGB)


def video_get_occ_aug(video1, video2):
    # cv2 img: h, w, c
    return apply_moving_mask_to_videos(
        get_temp(),
        video1,
        video2
    )


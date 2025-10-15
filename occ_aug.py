import random
import os

import cv2
import numpy as np

def list_files_recursive(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)

    return file_paths

texs_file = list_files_recursive('path/to/OCCdata/textures')
objs_file = list_files_recursive('path/to/OCCdata/temps/objs')
hands_file = list_files_recursive('path/to/OCCdata/temps/hands')

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

def overlay_images(img1, img2, img3=None):
    # Target area to cover is 1/3 of img2's area
    target_area = (img2.shape[0] * img2.shape[1]) / 4

    # Transform img1
    transformed_img1 = random_transform(img1, target_area)

    # Determine random position for overlay
    max_y = img2.shape[0] - transformed_img1.shape[0]
    max_x = img2.shape[1] - transformed_img1.shape[1]
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)

    # Create masks
    transformed_img1_mask = transformed_img1 > 0  # Mask for non-zero areas in transformed_img1
    transformed_img1 = random_color_transform(transformed_img1)
    # Overlay on img2 and img3
    for c in range(img2.shape[2]):  # Iterate over each channel
        img2[start_y:start_y + transformed_img1.shape[0], start_x:start_x + transformed_img1.shape[1], c] = \
            np.where(transformed_img1_mask[:, :, c], transformed_img1[:, :, c], img2[start_y:start_y + transformed_img1.shape[0], start_x:start_x + transformed_img1.shape[1], c])

        if not img3 is None:

            img3[start_y:start_y + transformed_img1.shape[0], start_x:start_x + transformed_img1.shape[1], c] = \
                np.where(transformed_img1_mask[:, :, c], transformed_img1[:, :, c], img3[start_y:start_y + transformed_img1.shape[0], start_x:start_x + transformed_img1.shape[1], c])

    return img2, img3


def get_temp():
    this_seed = random.random()
    if this_seed <= 0.25:
        return gen_texture_temp()
    elif 0.25 < this_seed < 0.75:
        return cv2.imread(
            random.choice(hands_file), cv2.IMREAD_COLOR
        )
    else:
        return cv2.imread(
            random.choice(objs_file), cv2.IMREAD_COLOR
        )

def img_get_occ_aug(img1, img2):
    # cv2 img: h, w, c
    return overlay_images(
        get_temp(),
        img1,
        img2
    )


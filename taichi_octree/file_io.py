# taichi_octree/file_io.py

import cv2
import numpy as np
import os

def load_images(folder_path, target_size=(800, 800)):
    """
    Loads all images from the specified folder, resizes them to the target size,
    and normalizes pixel values to the range [0, 1].

    Args:
        folder_path (str): Path to the folder containing image files.
        target_size (tuple): Desired image dimensions (width, height).

    Returns:
        np.ndarray: An array of images with shape (N, height, width, channels).
    """
    images = []
    file_names = sorted(os.listdir(folder_path))
    for filename in file_names:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
            images.append(img.astype(np.float32) / 255.0)
    return np.array(images)

def load_camera_parameters(file_path):
    """
    Loads camera parameters from a file. This function assumes that the parameters
    are stored in a NumPy file (.npy) or can be modified to support other formats.

    Args:
        file_path (str): Path to the file containing camera parameters.

    Returns:
        np.ndarray: Array of camera parameters.
    """
    try:
        cam_params = np.load(file_path)
        return cam_params
    except Exception as e:
        print(f"Error loading camera parameters from {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    images = load_images("data/images", target_size=(800, 800))
    print(f"Loaded {len(images)} images with shape: {images.shape}")

    cam_params = load_camera_parameters("data/cam_params.npy")
    if cam_params is not None:
        print(f"Loaded camera parameters with shape: {cam_params.shape}")

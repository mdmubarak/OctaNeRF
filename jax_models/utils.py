import cv2
import numpy as np
import os

def load_images_from_folder(folder_path, target_size=(800,800)):
    images = []
    file_names = sorted(os.listdir(folder_path))
    for filename in file_names:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(images)

def load_camera_parameters(file_path):
    # For example, load camera poses from a .npy file
    return np.load(file_path)

def positional_encoding(x, L=10):
    # x is assumed to be a numpy array of shape (..., dim)
    enc = [x]
    for i in range(L):
        enc.append(np.sin(2.0**i * x))
        enc.append(np.cos(2.0**i * x))
    return np.concatenate(enc, axis=-1)

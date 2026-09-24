import os

import numpy as np
import cv2 as cv


def load_images(split_dir):
    images, labels, image_paths = [], [], []
    for member_name in sorted(os.listdir(split_dir)):
        member_dir = os.path.join(split_dir, member_name)
        if not os.path.isdir(member_dir):
            continue
        for file_name in os.listdir(member_dir):
            image_path = os.path.join(member_dir, file_name)
            image = cv.imread(image_path)
            if image is None:
                continue
            images.append(image)
            labels.append(member_name)
            image_paths.append(image_path)

    return images, np.array(labels), np.array(image_paths)

import numpy as np
import cv2 as cv

from preprocess import load_grayscale


def augment_image(image):
    variants = []

    # flip horizontal
    variants.append(cv.flip(image, 1))

    # rotasi kecil (15 derajat)
    height, width = image.shape
    for angle in (-15, 15):
        rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        variants.append(cv.warpAffine(image, rotation_matrix, (width, height), borderMode=cv.BORDER_REPLICATE))

    # brightness
    variants.append(cv.convertScaleAbs(image, alpha=1.2, beta=15))
    variants.append(cv.convertScaleAbs(image, alpha=0.8, beta=-15))

    # translasi kecil
    shift_x, shift_y = int(0.05 * width), int(0.05 * height)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    variants.append(cv.warpAffine(image, translation_matrix, (width, height), borderMode=cv.BORDER_REPLICATE))

    return variants


def augment_train_set(image_paths, labels):
    augmented_images, augmented_labels, augmented_paths = [], [], []

    for image_path, label in zip(image_paths, labels):
        image = load_grayscale(image_path)

        for variant in augment_image(image):
            augmented_images.append(variant)
            augmented_labels.append(label)
            augmented_paths.append(image_path + " (aug)")

    return augmented_images, np.array(augmented_labels), np.array(augmented_paths)

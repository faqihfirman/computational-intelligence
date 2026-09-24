import cv2 as cv

IMAGE_SIZE = (224, 224)


def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def resize_image(image, size=IMAGE_SIZE):
    return cv.resize(image, size)


def preprocess_image(image, size=IMAGE_SIZE):
    return resize_image(to_grayscale(image), size)


def load_grayscale(image_path, size=IMAGE_SIZE):
    image = cv.imread(image_path)
    if image is None:
        return None
    return preprocess_image(image, size)

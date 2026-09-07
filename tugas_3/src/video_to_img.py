import cv2 as cv
import os


def video_to_img(video_path, train_folder, test_folder, nama_anggota,
                  frame_skip=5, img_size=(224, 224), train_ratio=0.8):

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    split_frame = int(total_frames * train_ratio)

    count = 0
    train_count = 0
    test_count = 0

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        if count % frame_skip == 0:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            resized_frame = cv.resize(gray_frame, img_size)

            if count < split_frame:
                file_name = f"{train_folder}/{nama_anggota}_train_{train_count}.jpg"
                cv.imwrite(file_name, resized_frame)
                train_count += 1
            else:
                file_name = f"{test_folder}/{nama_anggota}_test_{test_count}.jpg"
                cv.imwrite(file_name, resized_frame)
                test_count += 1

        count += 1

    capture.release()
    print(f"Selesai! {nama_anggota}: {train_count} train, {test_count} test")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_folder = os.path.join(base_dir, "data", "videos")
    image_folder = os.path.join(base_dir, "data", "images")

    for file_name in os.listdir(video_folder):
        if not file_name.lower().endswith((".mov", ".mp4")):
            continue

        nama_anggota = os.path.splitext(file_name)[0]
        video_path = os.path.join(video_folder, file_name)
        train_folder = os.path.join(image_folder, "train", nama_anggota)
        test_folder = os.path.join(image_folder, "test", nama_anggota)

        video_to_img(video_path, train_folder, test_folder, nama_anggota)

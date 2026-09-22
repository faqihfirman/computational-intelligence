import cv2 as cv
import os


def video_to_img(video_path, image_folder, nama_anggota, split, start_index=0,
                  frame_skip=5, img_size=(224, 224)):

    os.makedirs(image_folder, exist_ok=True)

    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return start_index

    count = 0
    img_count = start_index

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        if count % frame_skip == 0:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            resized_frame = cv.resize(gray_frame, img_size)

            file_name = f"{image_folder}/{nama_anggota}_{split}_{img_count:05d}.jpg"
            cv.imwrite(file_name, resized_frame)
            img_count += 1

        count += 1

    capture.release()
    print(f"Selesai! {nama_anggota}: {img_count - start_index} images ({video_path})")
    return img_count


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_root = os.path.join(base_dir, "data", "videos")
    image_root = os.path.join(base_dir, "data", "images")

    for split in ("train", "test"):
        split_video_folder = os.path.join(video_root, split)
        if not os.path.isdir(split_video_folder):
            continue

        for nama_anggota in os.listdir(split_video_folder):
            person_video_folder = os.path.join(split_video_folder, nama_anggota)
            if not os.path.isdir(person_video_folder):
                continue

            video_files = sorted(
                f for f in os.listdir(person_video_folder)
                if f.lower().endswith((".mov", ".mp4"))
            )
            if not video_files:
                continue

            person_image_folder = os.path.join(image_root, split, nama_anggota)
            img_count = 0
            for file_name in video_files:
                video_path = os.path.join(person_video_folder, file_name)
                img_count = video_to_img(video_path, person_image_folder,
                                          nama_anggota, split, start_index=img_count)

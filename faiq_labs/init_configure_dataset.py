from pathlib import Path
import tempfile
from tempfile import tempdir

import cv2
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "raihanfaiq72/my_face"
VIDEO_FILE = "my_face.mov"

PICTURE_FILDER = "picture/faiq"

FRAME_INTERVAL = 1

api = HfApi()

# Proses download, ectrak, upload
print("=" * 60)
print("Muka set GW download - Ekstrak per frame")
print("=" * 60)

print(f"Repo ID: {REPO_ID}")
print(f"Video File: {VIDEO_FILE}")
print()

# Direktori sementara

with tempfile.TemporaryDirectory() as tmpdir:
    print("Downloading video file...")

    local_video = hf_hub_download(
        repo_id=REPO_ID,
        filename=VIDEO_FILE,
        repo_type="dataset",
        local_dir=tmpdir,
    )

    print(f"Video berhasil di download : ")
    print(local_video)
    print()


    # Buka video
    cap = cv2.VideoCapture(local_video)

    if not cap.isOpened():
        raise RuntimeError(f"Video file {local_video} gaada.")

    # info vid
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print("Info vid : ")
    print(f"FPS           : {fps:.2f}")
    print(f"Total frame   : {total_frames}")
    print(f"Durasi        : {duration:.2f} detik")
    print()


    ## menyiapkan output folder
    local_output = (
        Path(tmpdir) / "picture"
    )

    local_output.mkdir(parents=True, exist_ok=True)

    ## ekstrak frame
    print("memecah video ke frame ...")
    print(f"Frame interval: setiap {FRAME_INTERVAL} frame")
    print()

    frame_number = 0
    saved_number = 0

    while True:
        success, frame = cap.read()

        # vid selesai
        if not success:
            break

        # ambil frame berdasarkan interval
        if frame_number % FRAME_INTERVAL == 0:
            filename = (
                f"frame_{saved_number:06d}.jpg"
            )

            output_file = (local_output / filename)

            # simpan sebagai jpg
            success_write = cv2.imwrite(str(output_file), frame)

            if not success_write:
                print(f"Gagal menyimpan : {filename}")
            else:
                saved_number += 1

        frame_number += 1
    cap.release()

    # tampilkanhasil
    print()
    print("Extraction selesai.")
    print(f"Frame dibaca    : {frame_number}")
    print(f"Frame disimpan  : {saved_number}")
    print()

    # proses upload ke hugging face

    if saved_number > 0:
        print("proses upload frame ke hugging face ...")

        api.upload_folder(
            folder_path = str(local_output),
            path_in_repo=PICTURE_FILDER,
            repo_id = REPO_ID,
            repo_type = 'dataset',
            commit_message="feat: extraction video ftrames"
        )

        print()
        print("done uplaod!")
print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print()
print(
    f"Frame tersedia di: "
    f"{REPO_ID}/{PICTURE_FILDER}/"
)
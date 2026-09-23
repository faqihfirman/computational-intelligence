import os
import shutil
from huggingface_hub import HfApi

REPO_ID = "raihanfaiq72/my_face"
REPO_TYPE = "dataset"

# Isi bagian ini 
SOURCE_VIDEO_PATH = "../data/videos/test/faqih/faqih_test_video.mp4"   # path video mentah (local)
NAMA_ANGGOTA = "faqih"                          # nama folder tujuan hugging face
SPLIT = "test"                                 # "train" atau "test"


def upload_video(source_video_path, nama_anggota, split,
                  repo_id=REPO_ID, repo_type=REPO_TYPE):
    if split not in ("train", "test"):
        raise ValueError('SPLIT harus "train" atau "test"')

    source_video_path = os.path.abspath(source_video_path)
    if not os.path.isfile(source_video_path):
        raise FileNotFoundError(f"File tidak ditemukan: {source_video_path}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_name = os.path.basename(source_video_path)

    person_folder = os.path.join(base_dir, "data", "videos", split, nama_anggota)
    os.makedirs(person_folder, exist_ok=True)
    local_dest = os.path.join(person_folder, file_name)
    if os.path.abspath(source_video_path) != os.path.abspath(local_dest):
        shutil.copy2(source_video_path, local_dest)

    path_in_repo = f"videos/{split}/{nama_anggota}/{file_name}"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_dest,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Selesai! Lokal: {local_dest}")
    print(f"Selesai! Repo : {repo_id}/{path_in_repo}")


if __name__ == "__main__":
    upload_video(SOURCE_VIDEO_PATH, NAMA_ANGGOTA, SPLIT)

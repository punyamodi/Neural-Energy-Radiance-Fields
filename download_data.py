import subprocess
import gdown
from tqdm import tqdm
import os

def install_pytorch():
    """Install PyTorch with CUDA 12.8 using pip."""
    print("🔧 Installing PyTorch...")
    subprocess.check_call([
        "pip3", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    ])
    print("✅ PyTorch installation complete.\n")

def download_with_progress(gdrive_url, output_path):
    """Download file from Google Drive using gdown and show progress bar."""
    file_id = gdrive_url.split('/d/')[1].split('/')[0]
    direct_url = f"https://drive.google.com/uc?id={file_id}"

    temp_path = output_path + ".download"
    gdown.download(direct_url, temp_path, quiet=False)

    total_size = os.path.getsize(temp_path)
    with open(temp_path, 'rb') as src, open(output_path, 'wb') as dst, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=f"Saving {output_path}"
    ) as pbar:
        for chunk in iter(lambda: src.read(8192), b''):
            dst.write(chunk)
            pbar.update(len(chunk))

    os.remove(temp_path)
    print(f"✅ Download completed: {output_path}\n")

# === Main execution ===
if __name__ == "__main__":
    install_pytorch()

    file1_url = 'https://drive.google.com/file/d/1WkEFwufyAWEEKngR4BLsLyDlh_1AaZAW/view?usp=sharing'
    file2_url = 'https://drive.google.com/file/d/1yPxbii0AkF87sHG0h-biDj0-J-3mB_1X/view?usp=sharing'

    download_with_progress(file1_url, 'training_data.pkl')
    download_with_progress(file2_url, 'testing_data.pkl')

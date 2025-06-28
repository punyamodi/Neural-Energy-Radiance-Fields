import gdown
from tqdm import tqdm
import os

def download_with_progress(gdrive_url, output_path):
    # Use gdown to get the file id and construct a direct URL
    file_id = gdrive_url.split('/d/')[1].split('/')[0]
    direct_url = f"https://drive.google.com/uc?id={file_id}"

    # Download file to temporary path
    temp_path = output_path + ".download"
    gdown.download(direct_url, temp_path, quiet=False)

    # Show progress bar as we rename (or copy in chunks if needed)
    total_size = os.path.getsize(temp_path)
    with open(temp_path, 'rb') as src, open(output_path, 'wb') as dst, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=f"Saving {output_path}"
    ) as pbar:
        for chunk in iter(lambda: src.read(8192), b''):
            dst.write(chunk)
            pbar.update(len(chunk))

    os.remove(temp_path)
    print(f"✅ Download completed: {output_path}")

# === Example usage ===
file1_url = 'https://drive.google.com/file/d/1WkEFwufyAWEEKngR4BLsLyDlh_1AaZAW/view?usp=sharing'
file2_url = 'https://drive.google.com/file/d/1yPxbii0AkF87sHG0h-biDj0-J-3mB_1X/view?usp=sharing'

download_with_progress(file1_url, 'training_data.pkl')
download_with_progress(file2_url, 'testing_data.pkl')
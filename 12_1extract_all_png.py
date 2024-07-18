import os
import shutil

def extract_and_rename_pngs(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for subdir, _, files in os.walk(src_folder):
        subdir_name = os.path.basename(subdir)
        for file in files:
            if file.endswith(".png"):
                new_name = f"{subdir_name}_{file}"
                src_file_path = os.path.join(subdir, file)
                dst_file_path = os.path.join(dst_folder, new_name)
                shutil.copy(src_file_path, dst_file_path)
                print(f"Copied {src_file_path} to {dst_file_path}")

src_folder = r'D:\yan\shujuji\Diffusion_dataset\compare_4'
dst_folder = r'D:\yan\shujuji\Diffusion_dataset\png_all'

extract_and_rename_pngs(src_folder, dst_folder)

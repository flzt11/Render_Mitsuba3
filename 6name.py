import os
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def save_subfolder_names_to_txt(base_folder, output_file):
    # Get list of all subfolders, skipping those that don't start with an alphanumeric character
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir() and re.match(r'^[a-zA-Z0-9]', f.name)]

    # Sort subfolders using natural order
    subfolders_sorted = sorted(subfolders, key=natural_sort_key)

    # Write subfolder names to the output file
    with open(output_file, 'w') as file:
        for subfolder in subfolders_sorted:
            file.write(subfolder + '\n')

if __name__ == "__main__":
    base_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 请替换为你的主文件夹路径
    output_file = 'subfolder_names.txt'  # 输出的txt文件名，可以根据需要修改
    save_subfolder_names_to_txt(base_folder, output_file)

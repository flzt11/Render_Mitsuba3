import os
import shutil


def extract_specific_png_files(source_folder, target_folder):
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Walk through all directories in the source folder
    for root, dirs, _ in os.walk(source_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            # Look for PNG files named after the directory
            png_file_name = f"{dir_name}.png"
            png_file_path = os.path.join(dir_path, png_file_name)

            # Check if the file exists
            if os.path.exists(png_file_path):
                target_file_path = os.path.join(target_folder, png_file_name)

                # Copy the file
                shutil.copyfile(png_file_path, target_file_path)
                print(f"Copied: {png_file_path} to {target_file_path}")


# Example usage
source_folder = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'  # Replace with your source folder path
target_folder = r'D:\yan\shujuji\Diffusion_dataset\chart'  # Replace with your target folder path

extract_specific_png_files(source_folder, target_folder)

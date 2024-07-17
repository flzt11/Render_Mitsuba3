import os
import shutil

def extract_and_rename_hdr_files(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Initialize a counter for the new file names
    counter = 1

    # Walk through all directories and subdirectories in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.hdr'):
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Construct the new file name
                new_file_name = f"envlight{counter}.hdr"
                new_file_path = os.path.join(target_dir, new_file_name)

                # Copy the file to the new location with the new name
                shutil.copyfile(file_path, new_file_path)

                # Increment the counter
                counter += 1

    print(f"Extracted and renamed {counter - 1} .hdr files to {target_dir}")

# Example usage
source_folder = r"D:\yan\shujuji\Diffusion_dataset\envlight"
target_folder = r"D:\yan\shujuji\Diffusion_dataset\env_right"

extract_and_rename_hdr_files(source_folder, target_folder)

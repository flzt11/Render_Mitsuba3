import os

def save_subfolder_names_to_txt(base_folder, output_file):
    # Get list of all subfolders
    subfolders = [f.name for f in os.scandir(base_folder) if f.is_dir()]

    # Write subfolder names to the output file
    with open(output_file, 'w') as file:
        for subfolder in subfolders:
            file.write(subfolder + '\n')

if __name__ == "__main__":
    base_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 请替换为你的主文件夹路径
    output_file = 'subfolder_names.txt'  # 输出的txt文件名，可以根据需要修改
    save_subfolder_names_to_txt(base_folder, output_file)

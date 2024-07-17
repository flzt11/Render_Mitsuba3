import os
import random
import shutil


def move_random_hdr_files(source_folder, target_folder):
    # 获取所有.hdr文件
    hdr_files = [file for file in os.listdir(source_folder) if file.endswith('.hdr')]

    if not hdr_files:
        print("源文件夹中没有.hdr文件")
        return

    # 获取目标文件夹下的所有子文件夹
    subfolders = [os.path.join(target_folder, folder) for folder in os.listdir(target_folder) if
                  os.path.isdir(os.path.join(target_folder, folder))]

    hdr_index = 0

    for subfolder in subfolders:
        # 计算需要重复利用的文件索引
        if hdr_index >= len(hdr_files):
            hdr_index = 0

        # 选择当前的hdr文件
        random_file = hdr_files[hdr_index]
        hdr_index += 1

        # 目标文件路径
        target_file_path = os.path.join(subfolder, "env.hdr")

        # 复制并重命名hdr文件
        shutil.copy(os.path.join(source_folder, random_file), target_file_path)

        print(f"已将 {random_file} 复制到 {subfolder} 并重命名为 env.hdr")


# 示例用法
source_folder_path = r"D:\yan\shujuji\Diffusion_dataset\env_right"  # 替换为实际源文件夹路径
target_folder_path = r"D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust"  # 替换为实际目标文件夹路径

move_random_hdr_files(source_folder_path, target_folder_path)

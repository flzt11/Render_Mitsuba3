import os
import shutil

def copy_file_to_all_subfolders(source_file, target_folder):
    # 遍历目标文件夹中的所有子文件夹
    for root, dirs, files in os.walk(target_folder):
        for dir in dirs:
            # 构建子文件夹路径
            subfolder_path = os.path.join(root, dir)
            # 构建目标文件路径
            target_file_path = os.path.join(subfolder_path, os.path.basename(source_file))
            # 复制文件
            shutil.copy(source_file, target_file_path)
            print(f"Copied {source_file} to {target_file_path}")

# 示例用法
source_file = r'D:\yan\Mitsuba3\Render_Mitsuba3\config.xml'  # 源文件路径
target_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 目标文件夹路径

copy_file_to_all_subfolders(source_file, target_folder)

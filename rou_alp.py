import os
import shutil

# 原文件夹路径
source_dir = r'D:\yan\shujuji\jilong_shujuji\target'
# 目标文件夹路径
target_dir = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'

# 遍历原文件夹中的所有子目录
for root, dirs, files in os.walk(source_dir):
    for dir in dirs:
        source_subdir = os.path.join(root, dir)
        target_subdir = os.path.join(target_dir, dir)

        # 检查目标子目录是否存在，如果不存在则创建
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        # 定义要移动的文件名
        files_to_move = ['albedo.png', 'roughness_metallic.png']

        for file_name in files_to_move:
            source_file = os.path.join(source_subdir, file_name)
            target_file = os.path.join(target_subdir, file_name)

            # 检查文件是否存在，如果存在则移动
            if os.path.exists(source_file):
                shutil.move(source_file, target_file)
                print(f'Moved {source_file} to {target_file}')

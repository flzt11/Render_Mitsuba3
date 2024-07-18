import os
import shutil


def filter_and_copy_images(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否为PNG图片
        if filename.endswith(".png"):
            # 检查文件名中是否包含指定的子字符串
            if not any(sub in filename for sub in ["I0", "I45", "I90", "I135", "hist"]):
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(source_folder, filename)
                target_file = os.path.join(target_folder, filename)

                # 复制文件到目标文件夹
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} to {target_file}")


# 示例使用
source_folder = r'D:\yan\shujuji\Diffusion_dataset\png_all' # 替换为你的源文件夹路径
target_folder = r'D:\yan\shujuji\Diffusion_dataset\png_all_without_I'  # 替换为你的目标文件夹路径

filter_and_copy_images(source_folder, target_folder)

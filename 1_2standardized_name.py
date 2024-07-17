import os

def pad_numbers_in_folder_names(folder_path):
    # 遍历指定文件夹中的所有子文件夹
    for folder_name in os.listdir(folder_path):
        print(f'Checking folder: {folder_name}')  # 调试信息
        # 检查子文件夹名称是否以'objects_'开头
        if folder_name.startswith('objects_'):
            # 提取数字部分
            number_str = folder_name[8:]
            # 将数字补全为5位
            new_number_str = number_str.zfill(5)
            # 生成新的文件夹名称
            new_folder_name = f'objects_{new_number_str}'
            # 获取完整的原文件夹路径和新文件夹路径
            original_folder_path = os.path.join(folder_path, folder_name)
            new_folder_path = os.path.join(folder_path, new_folder_name)
            # 重命名文件夹
            os.rename(original_folder_path, new_folder_path)
            print(f'Renamed {original_folder_path} to {new_folder_path}')
        else:
            print(f'Skipped folder: {folder_name}')  # 调试信息

# 指定要处理的文件夹路径
# folder_path = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'
folder_path = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'
# 调用函数处理文件夹
pad_numbers_in_folder_names(folder_path)

import os
import cv2
import numpy as np

# 文件夹路径
folder3 = r'D:\yan\shujuji\Diffusion_dataset\processed_data_ng'
folder2 = r'D:\yan\shujuji\Diffusion_dataset\processed_data_gn'
folder1 = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'
output_folder = r'D:\yan\shujuji\Diffusion_dataset\compare_3'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置字体和位置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1

# 获取文件夹名称
folder_name1 = os.path.basename(folder1)
folder_name2 = os.path.basename(folder2)
folder_name3 = os.path.basename(folder3)

# 遍历文件夹中的子文件夹
subfolders = [f.name for f in os.scandir(folder1) if f.is_dir()]

for subfolder in subfolders:
    subfolder_path1 = os.path.join(folder1, subfolder)
    subfolder_path2 = os.path.join(folder2, subfolder)
    subfolder_path3 = os.path.join(folder3, subfolder)

    # 检查三个文件夹中是否都存在相同的子文件夹
    if os.path.exists(subfolder_path2) and os.path.exists(subfolder_path3):
        output_subfolder_path = os.path.join(output_folder, subfolder)
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # 获取子文件夹中的所有PNG文件
        png_files = [f for f in os.listdir(subfolder_path1) if f.endswith('.png')]

        for png_file in png_files:
            img_path1 = os.path.join(subfolder_path1, png_file)
            img_path2 = os.path.join(subfolder_path2, png_file)
            img_path3 = os.path.join(subfolder_path3, png_file)

            # 检查三个文件夹中是否都存在相同的图片
            if os.path.exists(img_path2) and os.path.exists(img_path3):
                img1 = cv2.imread(img_path1)
                img2 = cv2.imread(img_path2)
                img3 = cv2.imread(img_path3)

                # 获取图像的尺寸
                height, width, _ = img1.shape

                # 创建新图像用于比较，并添加文本
                new_img = np.zeros((height + 30, width * 3, 3), dtype=np.uint8)
                new_img[30:, :width] = img1
                new_img[30:, width:width*2] = img2
                new_img[30:, width*2:width*3] = img3

                cv2.putText(new_img, subfolder + ' (' + folder_name1 + ')', (10, 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(new_img, subfolder + ' (' + folder_name2 + ')', (width + 10, 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(new_img, subfolder + ' (' + folder_name3 + ')', (2 * width + 10, 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                # 保存比较后的图像
                output_img_path = os.path.join(output_subfolder_path, png_file)
                cv2.imwrite(output_img_path, new_img)
                print(f"Saved combined image: {output_img_path}")

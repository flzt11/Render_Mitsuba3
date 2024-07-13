import mitsuba as mi  # type: ignore
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

# 设置渲染的 variant
mi.set_variant('cuda_spectral_polarized')

def render_and_save(scene_folder, file_name, save_root):
    scene_path = os.path.join(scene_folder, file_name, 'config.xml')
    scene = mi.load_file(scene_path)
    save_path = os.path.join(save_root, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = mi.render(scene, spp=512)
    bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())

    S0 = np.array(mi.TensorXf(channels['S0']))
    S1 = np.array(mi.TensorXf(channels['S1']))
    S2 = np.array(mi.TensorXf(channels['S2']))

    # 打印S0、S1、S2的类型和基本信息
    print(f"S0 type: {type(S0)}, shape: {S0.shape}, dtype: {S0.dtype}")
    print(f"S1 type: {type(S1)}, shape: {S1.shape}, dtype: {S1.dtype}")
    print(f"S2 type: {type(S2)}, shape: {S2.shape}, dtype: {S2.dtype}")

    I0 = (S0 + S1) / 2
    I45 = (S0 + S2) / 2
    I90 = (S0 - S1) / 2
    I135 = (S0 - S2) / 2

    # 输出每张图像的最大最小值
    print(f"I0 max: {I0.max()}, I0 min: {I0.min()}")
    print(f"I45 max: {I45.max()}, I45 min: {I45.min()}")
    print(f"I90 max: {I90.max()}, I90 min: {I90.min()}")
    print(f"I135 max: {I135.max()}, I135 min: {I135.min()}")

    # 保存图像
    cv2.imwrite(os.path.join(save_path, 'I0.png'), I0)
    cv2.imwrite(os.path.join(save_path, 'I45.png'), I45)
    cv2.imwrite(os.path.join(save_path, 'I90.png'), I90)
    cv2.imwrite(os.path.join(save_path, 'I135.png'), I135)

    # 保存每张图像素的直方图
    for image, name in zip([I0, I45, I90, I135], ['I0', 'I45', 'I90', 'I135']):
        plt.figure()
        plt.hist(image.ravel(), bins=256, fc='k', ec='k')
        plt.title(f'Histogram of {name}')
        plt.savefig(os.path.join(save_path, f'{name}_histogram.png'))
        plt.close()

if __name__ == "__main__":
    input_txt = r'D:\yan\Mitsuba3\Render_Mitsuba3\subfolder_names.txt'  # 包含文件名的txt文件路径
    scene_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 替换为包含场景文件的主文件夹路径
    save_root = r'D:\yan\shujuji\Diffusion_dataset\light_test'  # 请替换为你指定的保存路径

    with open(input_txt, 'r') as file:
        file_names = [line.strip() for line in file if line.strip()]

    total_files = len(file_names)
    start_time = time.time()

    for i, file_name in enumerate(tqdm(file_names, desc="Rendering", unit="file")):
        try:
            render_and_save(scene_folder, file_name, save_root)
        except Exception as e:
            tqdm.write(f"Error processing {file_name}: {e}")
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (i + 1)) * (total_files - (i + 1))
        tqdm.write(f"Estimated time remaining: {remaining_time:.2f} seconds")

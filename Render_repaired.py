import mitsuba as mi  # type: ignore
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

def visualize_aop(aop):
    # 确保 aop 的范围在 [-π/2, π/2] 之间
    aop = np.clip(aop, -np.pi/2, np.pi/2)
    aop_normalized = (aop + np.pi/2) / np.pi  # 将 aop 范围从 [-0.5 * π, 0.5 * π] 转换到 [0, 1]
    aop_uint8 = np.uint8(aop_normalized * 255)  # 将 aop 归一化后的值转换为 0-255 范围的 uint8
    aop_map = cv2.applyColorMap(aop_uint8, cv2.COLORMAP_JET)
    return aop_map

def visualize_dolp(dolp):
    dolp_uint8 = np.uint8(dolp * 255)  # 将 dolp 值转换为 0-255 范围的 uint8
    dolp_map = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
    return dolp_map

# 设置渲染的 variant
mi.set_variant('cuda_spectral_polarized')

def save_with_colorbar(image, title, save_path, unit=None, vmin=None, vmax=None):
    plt.figure()
    im = plt.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    if unit:
        cbar.set_label(unit)  # 设置颜色条的标签（单位）
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def render_and_save(scene_folder, file_name, save_root):
    scene_path = os.path.join(scene_folder, file_name, 'config.xml')
    scene = mi.load_file(scene_path)
    resolution = 512
    save_path = os.path.join(save_root, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image = mi.render(scene, spp=512)
    bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())

    S0 = np.array(mi.TensorXf(channels['S0']))
    S1 = np.array(mi.TensorXf(channels['S1']))
    S2 = np.array(mi.TensorXf(channels['S2']))

    # 保留Depth和Normal的保存部分，并保存为16位图像
    depth = np.array(mi.TensorXf(channels['dd']))
    depth = np.clip(depth / depth.max(), 0, 1)
    cv2.imwrite(os.path.join(save_path, 'depth.png'), (depth * 65535).astype(np.uint16))

    normal = np.array(mi.TensorXf(channels['nn']))
    normal_copy = normal.copy()
    normal_copy[..., 0] = -normal[..., 2]
    normal_copy[..., 1] = normal[..., 1]
    normal_copy[..., 2] = -normal[..., 0]
    normal = normal_copy
    normal = (normal + 1) / 2
    cv2.imwrite(os.path.join(save_path, 'normal.png'), (normal * 65535).astype(np.uint16))

    # 计算I0, I45, I90, I135
    I0 = (S0 + S1) / 2
    I45 = (S0 + S2) / 2
    I90 = (S0 - S1) / 2
    I135 = (S0 - S2) / 2

    # 裁剪I0, I45, I90, I135到[0,1]范围并保存为16位图像
    I0 = np.clip(I0, 0, 1)
    I45 = np.clip(I45, 0, 1)
    I90 = np.clip(I90, 0, 1)
    I135 = np.clip(I135, 0, 1)

    I0_img = (I0 * 65535).astype(np.uint16)
    I45_img = (I45 * 65535).astype(np.uint16)
    I90_img = (I90 * 65535).astype(np.uint16)
    I135_img = (I135 * 65535).astype(np.uint16)

    # 计算每个通道的平均值
    I0_mean = I0.mean(-1)
    I45_mean = I45.mean(-1)
    I90_mean = I90.mean(-1)
    I135_mean = I135.mean(-1)

    # 输出 I0, I45, I90, I135 的形状
    print("I0 shape:", I0_mean.shape)
    print("I45 shape:", I45_mean.shape)
    print("I90 shape:", I90_mean.shape)
    print("I135 shape:", I135_mean.shape)

    cv2.imwrite(os.path.join(save_path, 'I0.png'), I0_img)
    cv2.imwrite(os.path.join(save_path, 'I45.png'), I45_img)
    cv2.imwrite(os.path.join(save_path, 'I90.png'), I90_img)
    cv2.imwrite(os.path.join(save_path, 'I135.png'), I135_img)

    # 重新计算S0, S1, S2
    S0_new = (I0_mean + I45_mean + I90_mean + I135_mean) / 2
    S1_new = I0_mean - I90_mean
    S2_new = I45_mean - I135_mean
    S0_image = (((I0 + I45 + I90 + I135) / 4) * 65535).astype(np.uint16)

    cv2.imwrite(os.path.join(save_path, 'S1_new.png'), (S1_new * 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(save_path, 'S2_new.png'), (S2_new * 65535).astype(np.uint16))

    aop = np.arctan2(S2_new, S1_new + 1e-8) / 2

    # 将 aop 范围从 [-0.5 * π, 0.5 * π] 转换到 [0, 1]
    aop_normalized = (aop + 0.5 * np.pi) / np.pi

    # 将范围从 [0, 1] 转换到 [0, 65535] 并转换为16位整数
    aop_16bit = (aop_normalized * 65535).astype(np.uint16)

    dop = np.clip(np.sqrt(S1_new ** 2 + S2_new ** 2) / (S0_new + 1e-8), a_min=0, a_max=1)

    # 将 im_dop 直接映射到 [0, 65535] 并转换为16位整数
    dop_16bit = (dop * 65535).astype(np.uint16)

    aop_map = visualize_aop(aop)
    dop_map = visualize_dolp(dop)

    # 带有colorbar的AOP和DOP图像保存
    save_with_colorbar(aop_map[:, :, [2, 1, 0]], 'AoP', os.path.join(save_path, 'AoP_with_colorbar.png'), unit='radians', vmin=-np.pi/2, vmax=np.pi/2)
    save_with_colorbar(dop_map[:, :, [2, 1, 0]], 'DoP', os.path.join(save_path, 'DoP_with_colorbar.png'), unit='dimensionless', vmin=0, vmax=1)

    # 计算正弦和余弦
    sin_aop = np.sin(aop * 2)
    cos_aop = np.cos(aop * 2)

    # 将范围从 [-1, 1] 映射到 [0, 1]
    sin_aop_normalized = (sin_aop + 1) / 2
    cos_aop_normalized = (cos_aop + 1) / 2

    # 将范围从 [0, 1] 映射到 [0, 65535] 并转换为16位整数
    sin_aop_16bit = (sin_aop_normalized * 65535).astype(np.uint16)
    cos_aop_16bit = (cos_aop_normalized * 65535).astype(np.uint16)

    # 合并三个通道成一个三通道图像
    three_channel_image = np.stack((sin_aop_16bit, cos_aop_16bit, dop_16bit), axis=-1)

    fig, ax = plt.subplots(ncols=5, figsize=(30, 5))
    ax[0].imshow((S0_image / 257).astype(np.uint8))
    ax[0].set_xlabel("S0_new: Intensity", size=14, weight='bold')
    ax[1].imshow(S1_new.astype(np.float32))
    ax[1].set_xlabel("S1_new: Horizontal vs. vertical", size=14, weight='bold')
    ax[2].imshow(S2_new.astype(np.float32))
    ax[2].set_xlabel("S2_new: Diagonal", size=14, weight='bold')
    ax[3].imshow(dop_map[:, :, [2, 1, 0]])
    ax[3].set_xlabel("DoP", size=14, weight='bold')
    ax[4].imshow(aop_map[:, :, [2, 1, 0]])
    ax[4].set_xlabel("AoP", size=14, weight='bold')

    plt.savefig(os.path.join(save_path, f'{file_name}.png'))
    # 保存为16位PNG格式
    cv2.imwrite(os.path.join(save_path, 'S0_image' + '.png'), S0_image)
    cv2.imwrite(os.path.join(save_path, 'DOP_16' + '.png'), dop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_16' + '.png'), aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_sin' + '.png'), sin_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_cos' + '.png'), cos_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_DOP' + '.png'), three_channel_image)

    # 保存最大最小值到txt文件
    with open(os.path.join(save_path, f'{file_name}.txt'), 'w') as f:
        f.write(f'I0_mean_max: {I0.max()}\n')
        f.write(f'I0_mean_min: {I0.min()}\n')
        f.write(f'I45_mean_max: {I45.max()}\n')
        f.write(f'I45_mean_min: {I45.min()}\n')
        f.write(f'I90_mean_max: {I90.max()}\n')
        f.write(f'I90_mean_min: {I90.min()}\n')
        f.write(f'I135_mean_max: {I135.max()}\n')
        f.write(f'I135_mean_min: {I135.min()}\n')
        f.write(f'S0_new_max: {S0_new.max()}\n')
        f.write(f'S0_new_min: {S0_new.min()}\n')
        f.write(f'S1_new_max: {S1_new.max()}\n')
        f.write(f'S1_new_min: {S1_new.min()}\n')
        f.write(f'S2_new_max: {S2_new.max()}\n')
        f.write(f'S2_new_min: {S2_new.min()}\n')
        f.write(f'S0_image_max: {S0_image.max()}\n')
        f.write(f'S0_image_min: {S0_image.min()}\n')
        f.write(f'aop_max: {aop.max()}\n')
        f.write(f'aop_min: {aop.min()}\n')
        f.write(f'aop_normalized_max: {aop_normalized.max()}\n')
        f.write(f'aop_normalized_min: {aop_normalized.min()}\n')
        f.write(f'aop_16bit_max: {aop_16bit.max()}\n')
        f.write(f'aop_16bit_min: {aop_16bit.min()}\n')
        f.write(f'dop_max: {dop.max()}\n')
        f.write(f'dop_min: {dop.min()}\n')
        f.write(f'dop_16bit_max: {dop_16bit.max()}\n')
        f.write(f'dop_16bit_min: {dop_16bit.min()}\n')

    print(f"Processed and saved: {file_name}")

if __name__ == "__main__":
    input_txt = r'D:\yan\Mitsuba3\Render_Mitsuba3\subfolder_names.txt'  # 包含文件名的txt文件路径
    scene_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 替换为包含场景文件的主文件夹路径
    save_root = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'  # 请替换为你指定的保存路径

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

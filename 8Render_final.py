import mitsuba as mi  # type: ignore
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
import time
import OpenEXR
import Imath
import json

def visualize_aop(aop):
    aop = np.clip(aop, -np.pi/2, np.pi/2)
    aop_normalized = (aop + np.pi/2) / np.pi
    aop_uint8 = np.uint8(aop_normalized * 255)
    aop_map = cv2.applyColorMap(aop_uint8, cv2.COLORMAP_JET)
    return aop_map

def visualize_dolp(dolp):
    dolp_uint8 = np.uint8(dolp * 255)
    dolp_map = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
    return dolp_map

mi.set_variant('cuda_spectral_polarized')

def save_with_colorbar(image, title, save_path, unit=None, vmin=None, vmax=None):
    plt.figure()
    im = plt.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    if unit:
        cbar.set_label(unit)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def save_exr(image, save_path):
    # 创建一个EXR文件的头部信息，指定图像的宽度和高度
    header = OpenEXR.Header(image.shape[1], image.shape[0])
    # 定义每个通道的像素类型为FLOAT（32位浮点数）
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    # 设置头部信息中的通道信息，指定图像包含红、绿、蓝三个通道
    header['channels'] = dict(R=float_chan, G=float_chan, B=float_chan)
    # 创建一个EXR输出文件，使用指定的头部信息
    exr = OpenEXR.OutputFile(save_path, header)
    # 提取图像的红色通道，并将其转换为32位浮点数，再转为字节数据
    r = (image[:, :, 0]).astype(np.float32).tobytes()
    # 提取图像的绿色通道，并将其转换为32位浮点数，再转为字节数据
    g = (image[:, :, 1]).astype(np.float32).tobytes()
    # 提取图像的蓝色通道，并将其转换为32位浮点数，再转为字节数据
    b = (image[:, :, 2]).astype(np.float32).tobytes()
    # 将三个通道的数据写入EXR文件
    exr.writePixels({'R': r, 'G': g, 'B': b})
    # 关闭EXR文件
    exr.close()


def save_histogram(image, title, save_path):
    colors = ('b', 'g', 'r')
    plt.figure()
    bins = np.linspace(image.min(), image.max(), 256)
    for i, color in enumerate(colors):
        hist, bin_edges = np.histogram(image[:, :, i], bins=bins)
        plt.plot(bin_edges[:-1], hist, color=color)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

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

    I0 = (S0 + S1) / 2
    I45 = (S0 + S2) / 2
    I90 = (S0 - S1) / 2
    I135 = (S0 - S2) / 2

    for image, name in zip([I0, I45, I90, I135], ['I0', 'I45', 'I90', 'I135']):
        plt.figure()
        plt.hist(image.ravel(), bins=256, fc='k', ec='k')
        plt.title(f'Histogram of {name}')
        plt.savefig(os.path.join(save_path, f'{name}_origin_hist.png'))
        plt.close()

    I0 = np.clip(I0, 0, 1)
    I45 = np.clip(I45, 0, 1)
    I90 = np.clip(I90, 0, 1)
    I135 = np.clip(I135, 0, 1)

    I0_img = (I0 * 65535).astype(np.uint16)
    I45_img = (I45 * 65535).astype(np.uint16)
    I90_img = (I90 * 65535).astype(np.uint16)
    I135_img = (I135 * 65535).astype(np.uint16)

    I0_mean = I0.mean(-1)
    I45_mean = I45.mean(-1)
    I90_mean = I90.mean(-1)
    I135_mean = I135.mean(-1)

    for image, name in zip([I0_mean, I45_mean, I90_mean, I135_mean], ['I0', 'I45', 'I90', 'I135']):
        plt.figure()
        plt.hist(image.ravel(), bins=256, fc='k', ec='k')
        plt.title(f'Histogram of {name}')
        plt.savefig(os.path.join(save_path, f'{name}_mean_hist.png'))
        plt.close()


    print("")
    print("S0 shape:", S0.shape)
    print("S1 shape:", S1.shape)
    print("S2 shape:", S2.shape)
    print("I0 shape:", I0_mean.shape)
    print("I45 shape:", I45_mean.shape)
    print("I90 shape:", I90_mean.shape)
    print("I135 shape:", I135_mean.shape)

    cv2.imwrite(os.path.join(save_path, 'I0_color.png'), cv2.cvtColor(I0_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I45_color.png'), cv2.cvtColor(I45_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I90_color.png'), cv2.cvtColor(I90_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I135_color.png'), cv2.cvtColor(I135_img, cv2.COLOR_BGR2RGB))

    S0_new = (I0_mean + I45_mean + I90_mean + I135_mean) / 2
    S1_new = I0_mean - I90_mean
    S2_new = I45_mean - I135_mean
    S0_image = (((I0 + I45 + I90 + I135) / 4) * 65535).astype(np.uint16)
    S0_image_RGB = cv2.cvtColor(S0_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(save_path, 'S1_new.png'), (S1_new * 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(save_path, 'S2_new.png'), (S2_new * 65535).astype(np.uint16))

    aop = np.arctan2(S2_new, S1_new + 1e-8) / 2
    aop_normalized = (aop + 0.5 * np.pi) / np.pi
    aop_16bit = (aop_normalized * 65535).astype(np.uint16)
    dop = np.clip(np.sqrt(S1_new ** 2 + S2_new ** 2) / (S0_new + 1e-8), a_min=0, a_max=1)
    dop_16bit = (dop * 65535).astype(np.uint16)

    aop_map = visualize_aop(aop)
    dop_map = visualize_dolp(dop)

    save_with_colorbar(aop_map[:, :, [2, 1, 0]], 'AoP', os.path.join(save_path, 'AoP_with_colorbar.png'), unit='radians', vmin=-np.pi/2, vmax=np.pi/2)
    save_with_colorbar(dop_map[:, :, [2, 1, 0]], 'DoP', os.path.join(save_path, 'DoP_with_colorbar.png'), unit='dimensionless', vmin=0, vmax=1)

    sin_aop = np.sin(aop * 2)
    cos_aop = np.cos(aop * 2)
    sin_aop_normalized = (sin_aop + 1) / 2
    cos_aop_normalized = (cos_aop + 1) / 2
    sin_aop_16bit = (sin_aop_normalized * 65535).astype(np.uint16)
    cos_aop_16bit = (cos_aop_normalized * 65535).astype(np.uint16)

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

    plt.savefig(os.path.join(save_path, f'{file_name}_chart.png'))
    cv2.imwrite(os.path.join(save_path, 'S0_image' + '.png'), S0_image_RGB)
    cv2.imwrite(os.path.join(save_path, 'DOP_16' + '.png'), dop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_16' + '.png'), aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_sin' + '.png'), sin_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_cos' + '.png'), cos_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_DOP' + '.png'), three_channel_image)

    save_exr(S0, os.path.join(save_path, 'S0.exr'))
    save_exr(S1, os.path.join(save_path, 'S1.exr'))
    save_exr(S2, os.path.join(save_path, 'S2.exr'))

    save_histogram(S0, 'Histogram of S0', os.path.join(save_path, 'S0_hist.png'))
    save_histogram(S1, 'Histogram of S1', os.path.join(save_path, 'S1_hist.png'))
    save_histogram(S2, 'Histogram of S2', os.path.join(save_path, 'S2_hist.png'))

    with open(os.path.join(save_path, f'{file_name}.txt'), 'w') as f:
        f.write(f'I0_mean_max: {I0.max()}\n')
        f.write(f'I0_mean_min: {I0.min()}\n')
        f.write(f'I45_mean_max: {I45.max()}\n')
        f.write(f'I45_mean_min: {I45.min()}\n')
        f.write(f'I90_mean_max: {I90.max()}\n')
        f.write(f'I90_mean_min: {I90.min()}\n')
        f.write(f'I135_mean_max: {I135.max()}\n')
        f.write(f'I135_mean_min: {I135.min()}\n')
        f.write(f'S0_origin_max: {S0.max()}\n')
        f.write(f'S0_origin_min: {S0.min()}\n')
        f.write(f'S1_origin_max: {S1.max()}\n')
        f.write(f'S1_origin_min: {S1.min()}\n')
        f.write(f'S2_origin_max: {S2.max()}\n')
        f.write(f'S2_origin_min: {S2.min()}\n')
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

def save_progress(progress_file, processed_files):
    with open(progress_file, 'w') as f:
        json.dump(processed_files, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    input_txt = r'D:\yan\Mitsuba3\Render_Mitsuba3\subfolder_names.txt'
    scene_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'
    save_root = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'
    progress_file = r'D:\yan\Mitsuba3\Render_Mitsuba3\progress.json'

    with open(input_txt, 'r') as file:
        file_names = [line.strip() for line in file if line.strip()]

    processed_files = load_progress(progress_file)
    files_to_process = [f for f in file_names if f not in processed_files]

    total_files = len(files_to_process)
    start_time = time.time()

    for i, file_name in enumerate(tqdm(files_to_process, desc="Rendering", unit="file")):
        try:
            render_and_save(scene_folder, file_name, save_root)
            processed_files.append(file_name)
            save_progress(progress_file, processed_files)
        except Exception as e:
            tqdm.write(f"Error processing {file_name}: {e}")
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (i + 1)) * (total_files - (i + 1))
        tqdm.write(f"Estimated time remaining: {remaining_time:.2f} seconds")

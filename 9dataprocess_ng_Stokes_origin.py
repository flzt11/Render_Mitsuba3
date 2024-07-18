import OpenEXR
import Imath
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    data = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32).reshape(size) for c in channels]
    exr_file.close()

    return np.stack(data, axis=-1)


def save_histogram(image, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ('b', 'g', 'r')
    plt.figure()
    bins = np.linspace(image.min(), image.max(), 256)
    if image.ndim == 3:
        for i, color in enumerate(colors):
            hist, bin_edges = np.histogram(image[:, :, i], bins=bins)
            plt.plot(bin_edges[:-1], hist, color=color)
    else:
        hist, bin_edges = np.histogram(image, bins=bins)
        plt.plot(bin_edges[:-1], hist, color='gray')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()


# def process_image(image, threshold):
#     normalized = np.clip(image / threshold, 0, 1)
#     return normalized
#
# def unified_normalization(I0, I45, I90, I135, percentage):
#     # Flatten all images and calculate the threshold for each
#     I0_threshold = np.percentile(I0.flatten(), 100 - percentage)
#     I45_threshold = np.percentile(I45.flatten(), 100 - percentage)
#     I90_threshold = np.percentile(I90.flatten(), 100 - percentage)
#     I135_threshold = np.percentile(I135.flatten(), 100 - percentage)
#
#     # Find the maximum threshold
#     max_threshold = max(I0_threshold, I45_threshold, I90_threshold, I135_threshold)
#
#     # Normalize all images using the maximum threshold
#     I0_normalized = process_image(I0, max_threshold)
#     I45_normalized = process_image(I45, max_threshold)
#     I90_normalized = process_image(I90, max_threshold)
#     I135_normalized = process_image(I135, max_threshold)
#
#     return I0_normalized, I45_normalized, I90_normalized, I135_normalized, max_threshold


def process_image(image, threshold, max_image_value):
    # 如果阈值小于1，则使用图像的最大值
    if threshold < 1:
        threshold = max_image_value
    normalized = np.clip(image / threshold, 0, 1)
    return normalized

def unified_normalization(I0, I45, I90, I135, percentage):
    # Flatten all images and calculate the threshold for each
    I0_threshold = np.percentile(I0.flatten(), 100 - percentage)
    I45_threshold = np.percentile(I45.flatten(), 100 - percentage)
    I90_threshold = np.percentile(I90.flatten(), 100 - percentage)
    I135_threshold = np.percentile(I135.flatten(), 100 - percentage)

    # Find the maximum threshold
    max_threshold = max(I0_threshold, I45_threshold, I90_threshold, I135_threshold)

    # Find the maximum value in all images
    max_image_value = max(I0.max(), I45.max(), I90.max(), I135.max())

    # Normalize all images using the dynamically adjusted threshold
    I0_normalized = process_image(I0, max_threshold, max_image_value)
    I45_normalized = process_image(I45, max_threshold, max_image_value)
    I90_normalized = process_image(I90, max_threshold, max_image_value)
    I135_normalized = process_image(I135, max_threshold, max_image_value)

    return I0_normalized, I45_normalized, I90_normalized, I135_normalized, max_threshold


def save_png(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)


def save_with_colorbar(image, title, save_path, unit=None, vmin=None, vmax=None):
    plt.figure()
    im = plt.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    if unit:
        cbar.set_label(unit)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_aop(aop):
    aop = np.clip(aop, -np.pi / 2, np.pi / 2)
    aop_normalized = (aop + np.pi / 2) / np.pi
    aop_uint8 = np.uint8(aop_normalized * 255)
    aop_map = cv2.applyColorMap(aop_uint8, cv2.COLORMAP_JET)
    return aop_map


def visualize_dolp(dolp):
    dolp_uint8 = np.uint8(dolp * 255)
    dolp_map = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
    return dolp_map


def process_exr_files(dir_path, save_path, percentage):
    S0_path = os.path.join(dir_path, 'S0.exr')
    S1_path = os.path.join(dir_path, 'S1.exr')
    S2_path = os.path.join(dir_path, 'S2.exr')

    if not (os.path.exists(S0_path) and os.path.exists(S1_path) and os.path.exists(S2_path)):
        return

    S0 = read_exr(S0_path)
    S1 = read_exr(S1_path)
    S2 = read_exr(S2_path)

    save_histogram(S0, 'Histogram of S0', os.path.join(save_path, 'S0_hist.png'))
    save_histogram(S1, 'Histogram of S1', os.path.join(save_path, 'S1_hist.png'))
    save_histogram(S2, 'Histogram of S2', os.path.join(save_path, 'S2_hist.png'))

    I0 = (S0 + S1) / 2
    I45 = (S0 + S2) / 2
    I90 = (S0 - S1) / 2
    I135 = (S0 - S2) / 2

    save_histogram(I0, 'Histogram of I0', os.path.join(save_path, 'I0_origin_hist.png'))
    save_histogram(I45, 'Histogram of I45', os.path.join(save_path, 'I45_origin_hist.png'))
    save_histogram(I90, 'Histogram of I90', os.path.join(save_path, 'I90_origin_hist.png'))
    save_histogram(I135, 'Histogram of I135', os.path.join(save_path, 'I135_origin_hist.png'))

    I0_normalized, I45_normalized, I90_normalized, I135_normalized, max_threshold = unified_normalization(
        I0, I45, I90, I135, percentage
    )

    I0_img_color = (I0_normalized * 65535).astype(np.uint16)
    I45_img_color = (I45_normalized * 65535).astype(np.uint16)
    I90_img_color = (I90_normalized * 65535).astype(np.uint16)
    I135_img_color = (I135_normalized * 65535).astype(np.uint16)

    cv2.imwrite(os.path.join(save_path, 'I0_color.png'), cv2.cvtColor(I0_img_color, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I45_color.png'), cv2.cvtColor(I45_img_color, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I90_color.png'), cv2.cvtColor(I90_img_color, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(save_path, 'I135_color.png'), cv2.cvtColor(I135_img_color, cv2.COLOR_BGR2RGB))

    I0_mean = I0_normalized.mean(axis=-1)
    I45_mean = I45_normalized.mean(axis=-1)
    I90_mean = I90_normalized.mean(axis=-1)
    I135_mean = I135_normalized.mean(axis=-1)

    save_histogram(I0_mean, 'Histogram of I0_mean', os.path.join(save_path, 'I0_mean_hist.png'))
    save_histogram(I45_mean, 'Histogram of I45_mean', os.path.join(save_path, 'I45_mean_hist.png'))
    save_histogram(I90_mean, 'Histogram of I90_mean', os.path.join(save_path, 'I90_mean_hist.png'))
    save_histogram(I135_mean, 'Histogram of I135_mean', os.path.join(save_path, 'I135_mean_hist.png'))

    I0_img_gray = (I0_mean * 65535).astype(np.uint16)
    I45_img_gray = (I45_mean * 65535).astype(np.uint16)
    I90_img_gray = (I90_mean * 65535).astype(np.uint16)
    I135_img_gray = (I135_mean * 65535).astype(np.uint16)

    save_png(I0_img_gray, os.path.join(save_path, 'I0_img_gray.png'))
    save_png(I45_img_gray, os.path.join(save_path, 'I45_img_gray.png'))
    save_png(I90_img_gray, os.path.join(save_path, 'I90_img_gray.png'))
    save_png(I135_img_gray, os.path.join(save_path, 'I135_img_gray.png'))

    S0_new = S0.mean(axis=-1)
    S1_new = S1.mean(axis=-1)
    S2_new = S2.mean(axis=-1)

    S0_image = (((I0_normalized + I45_normalized + I90_normalized + I135_normalized) / 4) * 65535).astype(np.uint16)
    S0_image_RGB = cv2.cvtColor(S0_image, cv2.COLOR_BGR2RGB)

    aop = np.arctan2(S2_new, S1_new + 1e-8) / 2
    aop_normalized = (aop + 0.5 * np.pi) / np.pi
    aop_16bit = (aop_normalized * 65535).astype(np.uint16)
    dop = np.clip(np.sqrt(S1_new ** 2 + S2_new ** 2) / (S0_new + 1e-8), a_min=0, a_max=1)
    dop_16bit = (dop * 65535).astype(np.uint16)

    aop_map = visualize_aop(aop)
    dop_map = visualize_dolp(dop)

    save_with_colorbar(aop_map[:, :, [2, 1, 0]], 'AoP', os.path.join(save_path, 'AoP_with_colorbar.png'),
                       unit='radians', vmin=-np.pi / 2, vmax=np.pi / 2)
    save_with_colorbar(dop_map[:, :, [2, 1, 0]], 'DoP', os.path.join(save_path, 'DoP_with_colorbar.png'),
                       unit='dimensionless', vmin=0, vmax=1)

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

    plt.savefig(os.path.join(save_path, os.path.basename(dir_path) + '_chart.png'))
    cv2.imwrite(os.path.join(save_path, 'S0_image.png'), S0_image_RGB)
    cv2.imwrite(os.path.join(save_path, 'DOP_16.png'), dop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_16.png'), aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_sin.png'), sin_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_cos.png'), cos_aop_16bit)
    cv2.imwrite(os.path.join(save_path, 'AOP_DOP.png'), three_channel_image)
    plt.close()

    with open(os.path.join(save_path, 'thresholds.txt'), 'w') as f:
        f.write(f'I threshold: {max_threshold}\n')
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


def main(input_folder_path, save_path, percentage):
    # 读取已经处理过的文件夹列表
    processed_dirs = set()
    log_file = os.path.join(save_path, 'processed_dirs.log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_dirs = set(line.strip() for line in f)

    # 获取所有子文件夹，不按字母顺序排序
    sub_dirs = [d for d in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, d))]

    for dir in sub_dirs:
        if dir in processed_dirs:
            continue

        dir_path = os.path.join(input_folder_path, dir)
        save_dir_path = os.path.join(save_path, os.path.relpath(dir_path, input_folder_path))
        os.makedirs(save_dir_path, exist_ok=True)
        process_exr_files(dir_path, save_dir_path, percentage)

        # 更新日志文件
        with open(log_file, 'a') as f:
            f.write(dir + '\n')

        print(f"Processed folder: {dir}")


if __name__ == "__main__":
    input_folder = r'D:\yan\shujuji\Diffusion_dataset\synthetic_data'
    save_path = r'D:\yan\shujuji\Diffusion_dataset\processed_data_origin_Stokes'
    main(input_folder, save_path, 0.01)

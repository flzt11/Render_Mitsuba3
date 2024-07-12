import mitsuba as mi  # type: ignore
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def visualize_aop(aop):
    aop = (aop < 0) * np.pi + aop
    aop = aop / np.pi
    aop_map = cv2.applyColorMap(cv2.cvtColor(np.uint8(aop * 255), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    return aop_map


def visualize_dolp(dolp):
    dolp_map = cv2.applyColorMap(cv2.cvtColor(np.uint8(dolp * 255), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    return dolp_map


# 设置渲染的 variant
mi.set_variant('cuda_spectral_polarized')


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
    cv2.imwrite(os.path.join(save_path, 'S0.png'), cv2.cvtColor(S0, cv2.COLOR_BGR2RGB) * 255)
    S1 = np.array(mi.TensorXf(channels['S1']))
    cv2.imwrite(os.path.join(save_path, 'S1.png'), cv2.cvtColor(S1, cv2.COLOR_BGR2RGB) * 255)
    S2 = np.array(mi.TensorXf(channels['S2']))
    cv2.imwrite(os.path.join(save_path, 'S2.png'), cv2.cvtColor(S2, cv2.COLOR_BGR2RGB) * 255)
    depth = np.array(mi.TensorXf(channels['dd']))
    cv2.imwrite(os.path.join(save_path, 'depth.png'), depth / depth.max() * 255)
    normal = np.array(mi.TensorXf(channels['nn']))
    normal_copy = normal.copy()
    normal_copy[..., 0] = -normal[..., 2]
    normal_copy[..., 1] = normal[..., 1]
    normal_copy[..., 2] = -normal[..., 0]
    normal = normal_copy
    normal = (normal + 1) / 2 * 255
    cv2.imwrite(os.path.join(save_path, 'normal.png'), normal)

    aop = np.arctan2(S2, S1) / 2
    dop = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)
    aop_map = visualize_aop(aop)
    dop_map = visualize_dolp(dop)

    fig, ax = plt.subplots(ncols=5, figsize=(30, 5))
    ax[0].imshow(S0)
    ax[0].set_xlabel("S0: Intensity", size=14, weight='bold')
    ax[1].imshow(S1)
    ax[1].set_xlabel("S1: Horizontal vs. vertical", size=14, weight='bold')
    ax[2].imshow(S2)
    ax[2].set_xlabel("S2: Diagonal", size=14, weight='bold')
    ax[3].imshow(dop_map[:, :, [2, 1, 0]])
    ax[3].set_xlabel("DoP", size=14, weight='bold')
    ax[4].imshow(aop_map[:, :, [2, 1, 0]])
    ax[4].set_xlabel("AoP", size=14, weight='bold')

    plt.savefig(os.path.join(save_path, f'{file_name}.png'))
    cv2.imwrite(os.path.join(save_path, 'DoP.png'), dop_map)
    cv2.imwrite(os.path.join(save_path, 'AoP.png'), aop_map)

    I0 = (S0 + S1) / 2
    cv2.imwrite(os.path.join(save_path, 'I0.png'), cv2.cvtColor(I0, cv2.COLOR_BGR2RGB) * 255)
    I45 = (S0 + S2) / 2
    cv2.imwrite(os.path.join(save_path, 'I45.png'), cv2.cvtColor(I45, cv2.COLOR_BGR2RGB) * 255)
    I90 = (S0 - S1) / 2
    cv2.imwrite(os.path.join(save_path, 'I90.png'), cv2.cvtColor(I90, cv2.COLOR_BGR2RGB) * 255)
    I135 = (S0 - S2) / 2
    cv2.imwrite(os.path.join(save_path, 'I135.png'), cv2.cvtColor(I135, cv2.COLOR_BGR2RGB) * 255)

if __name__ == "__main__":
    input_txt = r'D:\yan\Mitsuba3\tool\file_name.txt'  # 包含文件名的txt文件路径
    scene_folder = r'D:\yan\shujuji\jilong_shujuji\target'  # 替换为包含场景文件的主文件夹路径
    save_root = r'D:\yan\Mitsuba3\Diffusion_dataset\synthetic_data'  # 请替换为你指定的保存路径

    with open(input_txt, 'r') as file:
        file_names = [line.strip() for line in file if line.strip()]

    for file_name in file_names:
        render_and_save(scene_folder, file_name, save_root)
#

# import mitsuba as mi  # type: ignore
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import os
#
#
# def visualize_aop(aop):
#     aop = (aop < 0) * np.pi + aop
#     aop = aop / np.pi
#     aop_map = cv2.applyColorMap(cv2.cvtColor(np.uint8(aop * 255), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
#     return aop_map
#
#
# def visualize_dolp(dolp):
#     dolp_map = cv2.applyColorMap(cv2.cvtColor(np.uint8(dolp * 255), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
#     return dolp_map
#
#
# def clip_and_convert_to_uint8(image):
#     image = np.clip(image, 0, 1)  # 将数据剪辑到 [0, 1] 范围
#     return np.uint8(image * 255)
#
#
# # 设置渲染的 variant
# mi.set_variant('cuda_spectral_polarized')
#
#
# def render_and_save(scene_folder, file_name, save_root):
#     scene_path = os.path.join(scene_folder, file_name, 'config.xml')
#     scene = mi.load_file(scene_path)
#     resolution = 512
#     save_path = os.path.join(save_root, file_name)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     image = mi.render(scene, spp=512)
#     bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
#     channels = dict(bitmap.split())
#
#     S0 = np.array(mi.TensorXf(channels['S0']))
#     cv2.imwrite(os.path.join(save_path, 'S0.png'), clip_and_convert_to_uint8(S0))
#     S1 = np.array(mi.TensorXf(channels['S1']))
#     cv2.imwrite(os.path.join(save_path, 'S1.png'), clip_and_convert_to_uint8(S1))
#     S2 = np.array(mi.TensorXf(channels['S2']))
#     cv2.imwrite(os.path.join(save_path, 'S2.png'), clip_and_convert_to_uint8(S2))
#     depth = np.array(mi.TensorXf(channels['dd']))
#     cv2.imwrite(os.path.join(save_path, 'depth.png'), clip_and_convert_to_uint8(depth / depth.max()))
#     normal = np.array(mi.TensorXf(channels['nn']))
#     normal_copy = normal.copy()
#     normal_copy[..., 0] = -normal[..., 2]
#     normal_copy[..., 1] = normal[..., 1]
#     normal_copy[..., 2] = -normal[..., 0]
#     normal = normal_copy
#     normal = (normal + 1) / 2
#     cv2.imwrite(os.path.join(save_path, 'normal.png'), clip_and_convert_to_uint8(normal))
#
#     aop = np.arctan2(S2, S1) / 2
#     dop = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)
#     aop_map = visualize_aop(aop)
#     dop_map = visualize_dolp(dop)
#
#     fig, ax = plt.subplots(ncols=5, figsize=(30, 5))
#     ax[0].imshow(clip_and_convert_to_uint8(S0))
#     ax[0].set_xlabel("S0: Intensity", size=14, weight='bold')
#     ax[1].imshow(clip_and_convert_to_uint8(S1))
#     ax[1].set_xlabel("S1: Horizontal vs. vertical", size=14, weight='bold')
#     ax[2].imshow(clip_and_convert_to_uint8(S2))
#     ax[2].set_xlabel("S2: Diagonal", size=14, weight='bold')
#     ax[3].imshow(dop_map[:, :, [2, 1, 0]])
#     ax[3].set_xlabel("DoP", size=14, weight='bold')
#     ax[4].imshow(aop_map[:, :, [2, 1, 0]])
#     ax[4].set_xlabel("AoP", size=14, weight='bold')
#
#     plt.savefig(os.path.join(save_path, f'{file_name}.png'))
#     cv2.imwrite(os.path.join(save_path, 'DoP.png'), dop_map)
#     cv2.imwrite(os.path.join(save_path, 'AoP.png'), aop_map)
#
#     I0 = (S0 + S1) / 2
#     cv2.imwrite(os.path.join(save_path, 'I0.png'), clip_and_convert_to_uint8(I0))
#     I45 = (S0 + S2) / 2
#     cv2.imwrite(os.path.join(save_path, 'I45.png'), clip_and_convert_to_uint8(I45))
#     I90 = (S0 - S1) / 2
#     cv2.imwrite(os.path.join(save_path, 'I90.png'), clip_and_convert_to_uint8(I90))
#     I135 = (S0 - S2) / 2
#     cv2.imwrite(os.path.join(save_path, 'I135.png'), clip_and_convert_to_uint8(I135))
#
#
# if __name__ == "__main__":
#     input_txt = r'D:\yan\Mitsuba3\tool\file_name.txt'  # 包含文件名的txt文件路径
#     scene_folder = r'D:\yan\shujuji\jilong_shujuji\target'  # 替换为包含场景文件的主文件夹路径
#     save_root = r'D:\yan\Mitsuba3\Diffusion_dataset\synthetic_data'  # 请替换为你指定的保存路径
#
#     with open(input_txt, 'r') as file:
#         file_names = [line.strip() for line in file if line.strip()]
#
#     for file_name in file_names:
#         render_and_save(scene_folder, file_name, save_root)
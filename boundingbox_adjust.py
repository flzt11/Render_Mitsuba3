import os
import numpy as np

def get_obj_bounding_box(obj_file):
    vertices = []
    with open(obj_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    if not vertices:
        raise ValueError(f"No vertices found in {obj_file}")
    vertices = np.array(vertices)
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    return min_corner, max_corner

def scale_obj_to_fixed_size(obj_file, output_file, target_size):
    min_corner, max_corner = get_obj_bounding_box(obj_file)
    current_size = max_corner - min_corner
    scale_factors = target_size / current_size
    scale_factor = min(scale_factors)  # Use the smallest scale factor to keep the object inside the target bounding box

    with open(obj_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                scaled_vertex = (vertex - min_corner) * scale_factor + min_corner
                file.write(f"v {scaled_vertex[0]} {scaled_vertex[1]} {scaled_vertex[2]}\n")
            else:
                file.write(line)

def process_all_obj_files(input_folder, output_folder, target_size):
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    obj_files_found = False

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.obj'):
                obj_files_found = True
                input_file = os.path.join(root, file)
                print(f"Found OBJ file: {input_file}")
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                output_file = os.path.join(output_subfolder, file)
                try:
                    scale_obj_to_fixed_size(input_file, output_file, target_size)
                    print(f"Processed {input_file} -> {output_file}")
                except ValueError as e:
                    print(e)

    if not obj_files_found:
        print("No OBJ files found.")

# 设置输入文件夹、输出文件夹和目标尺寸
input_folder = r'D:\yan\shujuji\jilong_shujuji\target'  # 替换为你的输入文件夹路径
output_folder = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'  # 替换为你的输出文件夹路径
target_size = np.array([1.0, 1.0, 1.0])  # 目标尺寸，单位根据需要调整

process_all_obj_files(input_folder, output_folder, target_size)

import os


def adjust_obj_y_position(file_path, target_y=-0.1):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    min_y = float('inf')
    vertices = []
    total_x = 0
    total_y = 0
    total_z = 0
    vertex_count = 0

    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            min_y = min(min_y, y)
            total_x += x
            total_y += y
            total_z += z
            vertex_count += 1
            vertices.append((x, y, z))

    y_offset = target_y - min_y
    center_x = total_x / vertex_count
    center_y = total_y / vertex_count
    center_z = total_z / vertex_count

    with open(file_path, 'w') as file:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x = float(parts[1]) - center_x
                y = float(parts[2]) + y_offset - center_y
                z = float(parts[3]) - center_z
                file.write(f'v {x} {y} {z}\n')
            else:
                file.write(line)

    print(f'Adjusted {file_path}')


def adjust_all_obj_files_in_folder(folder_path, target_y=-0.1):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.obj'):
                file_path = os.path.join(root, filename)
                adjust_obj_y_position(file_path, target_y)


# 使用示例
folder_path = r'D:\yan\shujuji\Diffusion_dataset\boundingbox_adjust'
adjust_all_obj_files_in_folder(folder_path, target_y=-0.1)

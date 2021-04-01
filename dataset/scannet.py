from torch.utils.data import Dataset, DataLoader
import trimesh as tri
import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
from collections import Counter
import os

import dataset.prepare_data as labels_loader


class RGBA(IntEnum):
    RED = 0,
    GREEN = 1,
    BLUE = 2


class VoxelData(IntEnum):
    RED = RGBA.RED,
    GREEN = RGBA.GREEN,
    BLUE = RGBA.BLUE,
    LABEL = 3,
    LENGTH = 4,
    NUMBER_OF_ATTRIBUTES = 5


voxel_grid_dimX = 64
voxel_grid_dimY = 64
voxel_grid_dimZ = 64


class VoxelSceneDataset(Dataset):
    def __init__(self, dataset_location):
        self.location = dataset_location
        self.sample_directories = [item for item in os.listdir(dataset_location) if not os.path.isfile(item)]

    def __len__(self):
        return len(self.sample_directories)

    def __getitem__(self, item_index):
        scene_path = os.path.join(self.location, self.sample_directories[item_index])
        clean_scene_path = ''
        for path in os.listdir(scene_path):
            if 'clean' in path:
                clean_scene_path = os.path.join(scene_path, path)
        coordinates, colors, labels = labels_loader.f(clean_scene_path)
        voxel_grid = np.ndarray((voxel_grid_dimX, voxel_grid_dimY, voxel_grid_dimZ, VoxelData.NUMBER_OF_ATTRIBUTES),
                                dtype=np.float64)
        xs = []
        ys = []
        zs = []
        for coordinate in coordinates:
            coordinate[1], coordinate[2] = coordinate[2], -coordinate[1]
            xs.append(coordinate[0])
            ys.append(coordinate[1])
            zs.append(coordinate[2])
        min_xs = min(xs)
        max_xs = max(xs)
        min_ys = min(ys)
        max_ys = max(ys)
        min_zs = min(zs)
        max_zs = max(zs)
        max_range = max((max_xs - min_xs), (max_ys - min_ys), (max_zs - min_zs))
        for vertex_index in range(len(coordinates)):
            #if vertex_index % 1000 == 0:
            #    print(vertex_index)
            normalized_x = (xs[vertex_index] - min_xs) / max_range
            normalized_y = (ys[vertex_index] - min_ys) / max_range
            normalized_z = (zs[vertex_index] - min_zs) / max_range
            voxel_x = np.floor(normalized_x * voxel_grid_dimX)
            voxel_y = np.floor(normalized_y * voxel_grid_dimY)
            voxel_z = np.floor(normalized_z * voxel_grid_dimZ)
            voxel_x = int(voxel_x)
            voxel_y = int(voxel_y)
            voxel_z = int(voxel_z)
            if voxel_x == 64:
                voxel_x -= 1
            if voxel_y == 64:
                voxel_y -= 1
            if voxel_z == 64:
                voxel_z -= 1
            voxel = voxel_grid[voxel_x][voxel_y][voxel_z]
            color = colors[vertex_index]
            voxel[VoxelData.RED] += color[RGBA.RED] * color[RGBA.RED]
            voxel[VoxelData.GREEN] += color[RGBA.GREEN] * color[RGBA.GREEN]
            voxel[VoxelData.BLUE] += color[RGBA.BLUE] * color[RGBA.BLUE]
            voxel[VoxelData.LABEL] += labels[vertex_index]
            voxel[VoxelData.LENGTH] += 1
        for x in range(len(voxel_grid[0])):
            for y in range(len(voxel_grid[1])):
                for z in range(len(voxel_grid[2])):
                    voxel = voxel_grid[x][y][z]
                    if voxel[VoxelData.LENGTH] > 0:
                        voxel[VoxelData.RED] = int(np.floor(np.sqrt(voxel[VoxelData.RED] / voxel[VoxelData.LENGTH])))
                        voxel[VoxelData.GREEN] = int(
                            np.floor(np.sqrt(voxel[VoxelData.GREEN] / voxel[VoxelData.LENGTH])))
                        voxel[VoxelData.BLUE] = int(np.floor(np.sqrt(voxel[VoxelData.BLUE] / voxel[VoxelData.LENGTH])))
                        voxel[VoxelData.LABEL] = voxel[VoxelData.LABEL] / voxel[VoxelData.LENGTH]
                        # TODO: wrong label count! Need to find most frequent label, not the mean of all labels!
        return voxel_grid


def visualize(voxel_grid):
    ply = open('visualization.ply', 'w')
    ply.write('ply\n')
    ply.write('format ascii 1.0\n')
    n_vertices = 0
    for x in range(len(voxel_grid[0])):
        for y in range(len(voxel_grid[1])):
            for z in range(len(voxel_grid[2])):
                voxel = voxel_grid[x][y][z]
                if voxel[VoxelData.LENGTH] > 0:
                    n_vertices += 1
    ply.write('element vertex %s\n' % n_vertices)
    ply.write('property float x\n')
    ply.write('property float y\n')
    ply.write('property float z\n')
    ply.write('property uchar red\n')
    ply.write('property uchar green\n')
    ply.write('property uchar blue\n')
    ply.write('end_header\n')
    for x in range(len(voxel_grid[0])):
        for y in range(len(voxel_grid[1])):
            for z in range(len(voxel_grid[2])):
                voxel = voxel_grid[x][y][z]
                if voxel[VoxelData.LENGTH] > 0:
                    red = int(voxel[VoxelData.RED])
                    green = int(voxel[VoxelData.GREEN])
                    blue = int(voxel[VoxelData.BLUE])
                    ply.write('%s %s %s %s %s %s\n' % (x, y, z, red, green, blue))
    ply.close()


if __name__ == "__main__":
    dataset = VoxelSceneDataset('scannet_subset20')
    scene = dataset[0]
    visualize(scene)

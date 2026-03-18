import os
import cv2
import scipy.ndimage
import numpy as np
import pickle

from bresenham import bresenham
from PIL import Image

def draw_image(vector_image, side=350):
    raster_image = np.zeros((int(side), int(side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < side and cord[1] < side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return raster_image

def preprocess(sketch_points, side=400):
    sketch_points = sketch_points.astype(np.float32)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images = draw_image(sketch_points)
    return raster_images
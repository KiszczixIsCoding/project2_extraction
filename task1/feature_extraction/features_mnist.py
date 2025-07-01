import math
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def flatten_image(data):
    img_vec, digit_vec = [], []

    for image_elem in data:
        image, digit = image_elem
        image_vector = np.asarray(image, dtype=np.float32) / 255.0
        img_vec.append(image_vector.flatten())
        digit_vec.append(digit)

    return [img_vec, digit_vec]


def verge_points(data):
    vector, img_vec, digit_vec = [], [], []

    for image_elem in data:
        image, digit = image_elem
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        for row in image_array:
            begin, end = 0, 0
            reversed_row = row[::-1]

            for index, elem in np.ndenumerate(row):
                if elem != 0:
                    begin = index[0]
                    break

            for index, elem in np.ndenumerate(reversed_row):
                if elem != 0:
                    end = 28 - index[0]
                    break
            vector.append((end - begin) / 28)

        img_vec.append(vector)
        digit_vec.append(digit)
        vector = []

    return [img_vec, digit_vec]


# def centroid_point(data):
#     img_vec, digit_vec = [], []
#     for image_elem in data:
#         image, digit = image_elem
#         image = np.asarray(image, dtype=np.float32)
#         y_coords, x_coords = np.indices(image.shape)
#         total = image.sum()
#
#         if total == 0:
#             return [14, 14]  # środek obrazu
#
#         x_center = (x_coords * image).sum() / total
#         y_center = (y_coords * image).sum() / total
#
#         bin_img = (np.array(image) > 0.1).astype(int)
#         row_trans = np.sum(np.abs(np.diff(bin_img, axis=1)))
#         col_trans = np.sum(np.abs(np.diff(bin_img, axis=0)))
#
#         img_vec.append([x_center, row_trans + col_trans])
#         digit_vec.append(digit)
#
#     return [img_vec, digit_vec]

def centroid_point1(image):
    image = np.asarray(image, dtype=np.float32)
    y_coords, x_coords = np.indices(image.shape)
    total = image.sum()

    if total == 0:
        return [14, 14]  # środek obrazu

    x_center = (x_coords * image).sum() / total

    return x_center

# def centroid_point(data):
#     img_vec, digit_vec = [], []
#     for image_elem in data:
#         image, digit = image_elem
#         image = np.array(image, dtype=np.float32) / 255.0  # normalizacja pikseli 0-1
#         bin_img = (image > 0.1).astype(float)  # binarizacja obrazu
#
#         # centroid X
#         x_coords = np.indices(bin_img.shape)[1]
#         total = bin_img.sum()
#         x_center = (x_coords * bin_img).sum() / total if total > 0 else bin_img.shape[1] / 2
#
#         # gęstość czarnych pikseli
#         density = total / bin_img.size  # wartość w [0,1]
#         img_vec.append([x_center, density])
#         digit_vec.append(digit)
#     return [img_vec, digit_vec]

def vertical_symmetry(image):
    image = np.array(image, dtype=np.float32) / 255.0
    flipped = np.fliplr(image)  # flip left-right (pionowa oś)
    diff = np.abs(image - flipped)
    symmetry_score = 1 - (diff.sum() / diff.size)  # normalizacja do [0,1], 1 - idealna symetria
    return symmetry_score


def horizontal_symmetry(image):
    image = np.array(image, dtype=np.float32) / 255.0
    flipped = np.flipud(image)  # flip up-down (pozioma oś)
    diff = np.abs(image - flipped)
    symmetry_score = 1 - (diff.sum() / diff.size)
    return symmetry_score

def x_variance(image):
    image = np.array(image, dtype=np.float32) / 255.0
    y_coords, x_coords = np.indices(image.shape)
    total = image.sum()
    if total == 0:
        return 0.0
    x_center = (x_coords * image).sum() / total
    variance = ((x_coords - x_center) ** 2 * image).sum() / total
    return variance

def pixel_density(image):
    image = np.array(image, dtype=np.float32) / 255.0
    return image.sum() / image.size

# def y_variance(image):
#     image = np.array(image, dtype=np.float32) / 255.0
#     y_coords, x_coords = np.indices(image.shape)
#     total = image.sum()
#     if total == 0:
#         return 0.0
#     y_center = (y_coords * image).sum() / total
#     variance = ((y_coords - y_center) ** 2 * image).sum() / total
#     return variance / 182.25

def y_variance(image):
    image = np.array(image, dtype=np.float32) / 255.0
    height = image.shape[0]
    y_coords, _ = np.indices(image.shape)
    total = image.sum()
    if total == 0:
        return 0.0
    y_center = (y_coords * image).sum() / total
    variance = ((y_coords - y_center) ** 2 * image).sum() / total
    max_variance = ((height - 1) / 2) ** 2  # np. dla 28x28: (27/2)^2 = 182.25
    return variance / max_variance

def central_moment(image):
    image = np.array(image, dtype=np.float32) / 255.0
    y_coords, x_coords = np.indices(image.shape)
    total = image.sum()
    if total == 0:
        return 0.0
    x_center = (x_coords * image).sum() / total
    y_center = (y_coords * image).sum() / total
    moment = ((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2) * image
    return moment.sum() / total

def centroid_x(image):
    image = np.array(image, dtype=np.float32) / 255.0
    y_coords, x_coords = np.indices(image.shape)
    total = image.sum()
    if total == 0:
        return 14.0
    return (x_coords * image).sum() / total

def centroid_point(data):
    img_vec, digit_vec = [], []
    for image_elem in data:
        image, digit = image_elem
        ver = vertical_symmetry(image)
        hor = horizontal_symmetry(image)
        img_vec.append([ver, y_variance(image)])
        # (ver + hor) / 2
        digit_vec.append(digit)

    # arr = np.array(img_vec, dtype=np.float32)
    #
    # # Initialize scaler
    # scaler = StandardScaler()
    #
    # # Fit and transform
    # scaled_data = scaler.fit_transform(arr)

    return [img_vec, digit_vec]

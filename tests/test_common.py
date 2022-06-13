import random

import numpy as np


def random_float(low=0.0, high=1.0):
    return random.uniform(a=low, b=high)


def seed_random_generators(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def random_array(shape, dtype=np.float32):
    random_array = np.random.rand(*shape)
    return random_array.astype(dtype)


def random_affine(dtype=np.float32):
    phi = random_float(-np.pi, np.pi)
    aff = np.eye(4)
    aff[0, 0] = np.cos(phi)
    aff[0, 1] = -np.sin(phi)
    aff[1, 0] = np.sin(phi)
    aff[1, 1] = np.cos(phi)
    aff[0, 3] = 0.5 + random_float(-0.1, 0.1)
    aff[1, 3] = 1.0 + random_float(-0.1, 0.1)
    return aff.astype(dtype)


def random_rigid(dtype=np.float32):
    phi = random_float(-np.pi, np.pi)
    aff = np.eye(4)
    aff[0, 0] = np.cos(phi)
    aff[0, 1] = -np.sin(phi)
    aff[1, 0] = np.sin(phi)
    aff[1, 1] = np.cos(phi)
    return aff.astype(dtype)


def random_tuple(N):
    return (random_float(),) * N


def create_circle(length, c=None, r=None, dtype=np.float32):

    if c is None:
        c = length // 2, length // 2
    elif c != tuple:
        c = (c, c)
    if r is None:
        r = min(c[0], c[1], length - c[0], length - c[1])

    Y, X = np.ogrid[:length, :length]
    circ = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2) <= r

    return circ.astype(dtype)


def create_rect(length, c=None, h=None, w=None, dtype=np.float32):

    if c is None:
        c = length // 2, length // 2
    elif c != tuple:
        c = (c, c)
    if w is None:
        w = min(c[0], length - c[0])
    if h is None:
        h = min(c[1], length - c[1])

    h = int(np.floor(h))
    w = int(np.floor(w))

    rect = np.zeros((length, length))
    rect[c[0] - w // 2 : c[0] + w // 2, c[1] - h // 2 : c[1] + h // 2] = 1.0

    return rect.astype(dtype)


def add_noise(array, mean=0, std=0.1):
    return array + np.random.normal(mean, std, array.shape)


def rotate(array, phi=0.0):

    if np.mod(phi, 2 * np.pi) == 0:
        return array

    height, width = array.shape

    new_height = round(abs(height * np.cos(phi)) + abs(width * np.sin(phi))) + 1
    new_width = round(abs(width * np.cos(phi)) + abs(height * np.sin(phi))) + 1

    array_out = np.zeros((new_height, new_width))

    original_centre_height = round(((height + 1) / 2) - 1)
    original_centre_width = round(((width + 1) / 2) - 1)

    new_centre_height = round(((new_height + 1) / 2) - 1)
    new_centre_width = round(((new_width + 1) / 2) - 1)

    for i in range(height):
        for j in range(width):
            y = height - 1 - i - original_centre_height
            x = width - 1 - j - original_centre_width

            tangent = np.tan(phi / 2)
            new_x = round(x - y * tangent)
            new_y = round(new_x * np.sin(phi) + y)
            new_x = round(new_x - new_y * tangent)

            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x

            array_out[new_y, new_x] = array[i, j]

    return array_out


def jaccard(array1, array2):
    array1 = np.asarray(array1).astype(bool)
    array2 = np.asarray(array2).astype(bool)
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    return intersection.sum() / float(union.sum())

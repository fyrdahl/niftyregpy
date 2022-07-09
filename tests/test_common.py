import random

import numpy as np
from skimage import transform


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


def create_square(length, c=None, size=None, dtype=np.float32):

    if c is None:
        c = length // 2, length // 2
    elif c != tuple:
        c = (c, c)
    if size != tuple:
        size = (size, size)

    h = int(np.floor(size[0]))
    w = int(np.floor(size[1]))

    rect = np.zeros((length, length))
    rect[c[0] - w // 2 : c[0] + w // 2, c[1] - h // 2 : c[1] + h // 2] = 1.0

    return rect.astype(dtype)


def add_noise(array, mean=0, std=0.1):
    return array + np.random.normal(mean, std, array.shape)


def jaccard(array1, array2):
    array1 = array1.astype(bool)
    array2 = array2.astype(bool)
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    return intersection.sum() / float(union.sum())


def dice(array1, array2):
    array1 = array1.astype(bool)
    array2 = array2.astype(bool)
    intersection = np.logical_and(array1, array2)
    return 2.0 * intersection.sum() / (array1.sum() + array2.sum())


def apply_swirl(array, c=None, s=10, r=None):

    if c is None:
        c = tuple(x // 2 for x in array.shape)
    elif c != tuple:
        c = (c, c)

    if r is None:
        r = sum(list(array.shape)) // 4

    return transform.swirl(
        array,
        center=c,
        strength=s,
        radius=r,
    )


def rotate_array(array, angle):
    angle_rad = np.deg2rad(angle)
    tform_trans = transform.AffineTransform(translation=-array.shape[0] // 2)
    tform_rot = transform.AffineTransform(rotation=angle_rad)
    return transform.warp(array, tform_trans + tform_rot + tform_trans.inverse)


def shear_array(array, angle):
    angle_rad = np.deg2rad(angle)
    tform_trans = transform.AffineTransform(translation=-array.shape[0] // 2)
    tform_rot = transform.AffineTransform(shear=angle_rad)
    return transform.warp(array, tform_trans + tform_rot + tform_trans.inverse)

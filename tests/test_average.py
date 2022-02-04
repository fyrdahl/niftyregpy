import test_common as common

from niftyregpy import average


def test_avg_affine():
    aff1 = common.create_random_affine()
    aff2 = common.create_random_affine()
    output = average.avg((aff1, aff2))
    assert output is not None


def test_avg_image():
    length = 256
    img1 = common.create_random_array((length, length))
    img2 = common.create_random_array((length, length))
    output = average.avg((img1, img2))
    assert output is not None


def test_avg_lts():
    aff1 = common.create_random_affine()
    aff2 = common.create_random_affine()
    output = average.avg_lts((aff1, aff2))
    assert output is not None

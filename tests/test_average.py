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


def test_avg_tran():
    length = 256
    ref = common.create_random_array((length, length))
    flo1 = common.create_random_array((length, length))
    flo2 = common.create_random_array((length, length))
    tran1 = common.create_random_array((length, length))
    tran2 = common.create_random_array((length, length))
    output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2))
    assert output is not None


def demean1():
    length = 256
    ref = common.create_random_array((length, length))
    aff1 = common.create_random_affine()
    aff2 = common.create_random_affine()
    flo1 = common.create_random_array((length, length))
    flo2 = common.create_random_array((length, length))
    output = average.avg_tran(ref, (aff1, aff2), (flo1, flo2))
    assert output is not None


def demean2():
    length = 256
    ref = common.create_random_array((length, length))
    tran1 = common.create_random_array((length, length))
    tran2 = common.create_random_array((length, length))
    flo1 = common.create_random_array((length, length))
    flo2 = common.create_random_array((length, length))
    output = average.avg_tran(ref, (tran1, tran2), (flo1, flo2))
    assert output is not None


def demean3():
    length = 256
    ref = common.create_random_array((length, length))
    aff1 = common.create_random_affine()
    aff2 = common.create_random_affine()
    flo1 = common.create_random_array((length, length))
    flo2 = common.create_random_array((length, length))
    tran1 = common.create_random_array((length, length))
    tran2 = common.create_random_array((length, length))
    output = average.avg_tran(ref, (aff1, aff2), (flo1, flo2), (tran1, tran2))
    assert output is not None

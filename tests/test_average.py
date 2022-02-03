import test_common as common

from niftyregpy import average, reg


def test_avg_affine():
    length = 256
    size = common.random_float(length / 8, length * 7 / 8)
    ref = common.create_rect(length, w=size, h=size)
    flo = common.create_rect(length, w=size, h=size)
    noisy_ref = common.add_noise(ref)
    noisy_flo = common.add_noise(flo)
    output_forward = reg.aladin(noisy_ref, noisy_flo, verbose=False)
    output_backward = reg.aladin(noisy_flo, noisy_ref, verbose=False)
    output = average.avg((output_forward["aff"], output_backward["aff"]))
    assert output is not None


def test_avg_image():
    length = 256
    rad_ref = common.random_float(length / 8, length * 7 / 8)
    ref = common.create_circle(length, r=rad_ref)
    noisy_ref = common.add_noise(ref)
    rad_flo = common.random_float(length / 8, length * 7 / 8)
    flo = common.create_circle(length, r=rad_flo)
    noisy_flo = common.add_noise(flo)
    output_forward = reg.f3d(noisy_ref, noisy_flo, pad=0, verbose=False)
    output_backward = reg.f3d(noisy_flo, noisy_ref, pad=0, verbose=False)
    output = average.avg((output_forward["res"], output_backward["res"]))
    assert output is not None


def test_avg_lts():
    length = 256
    size = common.random_float(length / 8, length * 7 / 8)
    ref = common.create_rect(length, w=size, h=size)
    flo = common.create_rect(length, w=size, h=size)
    noisy_ref = common.add_noise(ref)
    noisy_flo = common.add_noise(flo)
    output_forward = reg.aladin(noisy_ref, noisy_flo, verbose=False)
    output_backward = reg.aladin(noisy_flo, noisy_ref, verbose=False)
    output = average.avg_lts((output_forward["aff"], output_backward["aff"]))
    assert output is not None

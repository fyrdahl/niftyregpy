import numpy as np
from niftyregpy import reg
import test_common as common

def test_aladin_default():

    length = 256

    size_ref = common.random_float(length/8,length*3/8)
    ref = common.create_rect(length,w=size_ref,h=size_ref)

    size_flo = common.random_float(length/8,length*3/8)
    flo = common.create_rect(length,w=size_ref,h=size_flo)

    output = reg.aladin(ref,flo,verbose=False)
    assert(output is not None)

def test_f3d_default():

    length = 256

    rad_ref = common.random_float(length/8,length*3/8)
    ref = common.create_circle(length,r=rad_ref)
    noisy_ref = common.add_noise(ref)

    rad_flo = common.random_float(length/8,length*3/8)
    flo = common.create_circle(length,r=rad_flo)
    noisy_flo = common.add_noise(flo)
    
    output = reg.f3d(noisy_ref,noisy_flo,verbose=False)
    
    assert(output is not None)

def test_reg_float():
    input = common.create_random_array((256, 256), np.float32)
    output = reg.tools(input, float=input)
    assert(output is not None)

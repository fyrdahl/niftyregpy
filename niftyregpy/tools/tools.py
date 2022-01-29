import numpy as np
import tempfile as tmp
from ..utils import read_nifti, write_nifti, call_niftyreg

DEBUG = False
NAME=tmp.NamedTemporaryFile().name
BASE_STR = f'reg_tools -in {NAME}input.nii -out {NAME}output.nii '

def float(input):
    
    """
    The input image is converted to float
    """    
    
    cmd_str = BASE_STR
    cmd_str += '-float '
    
    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def down(input):
    
    """
    The input image is downsampled 2 times
    """
    
    cmd_str = BASE_STR
    cmd_str += '-down '

    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)
    
    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def smoS(input,sx=0.,sy=0.,sz=0.):
    
    """
    The input image is smoothed using a cubic b-spline kernel
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-smoS {sx} {sy} {sz} '

    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)
    
    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def smoG(input,sx=0.,sy=0.,sz=0.):
    
    """
    The input image is smoothed using a Gaussian kernel
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-smoG {sx} {sy} {sz} '
    
    write_nifti(f'{NAME}input.nii', input)
 
    if DEBUG:
        print(cmd_str)
    
    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def smoL(input,sx=0.,sy=0.,sz=0.):
    
    """
    The input label image is smoothed using a Gaussian kernel
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-smoL {sx} {sy} {sz} '

    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def add(input,x):
    
    """
    This image (or value) is added to the input
    """
    
    cmd_str = BASE_STR

    if np.isscalar(x):
        cmd_str += f'-add {x}'
    else:
        cmd_str += f'-add {NAME}x.nii'
        write_nifti(f'{NAME}x.nii', x)
        
    write_nifti(f'{NAME}input.nii', input)

    if DEBUG:
        print(cmd_str)
    
    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def sub(input,x):
    
    """
    This image (or value) is subtracted from the input
    """
    
    cmd_str = BASE_STR

    if np.isscalar(x):
        cmd_str += f'-sub {x}'
    else:
        cmd_str += f'-sub {NAME}x.nii'
        write_nifti(f'{NAME}x.nii', x)
    
    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def mul(input,x):
    
    """
    This image (or value) is multiplied with the input
    """
    
    cmd_str = BASE_STR

    if np.isscalar(x):
        cmd_str += f'-mul {x}'
    else:
        cmd_str += f'-mul {NAME}x.nii'
        write_nifti(f'{NAME}x.nii', x)
    
    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def div(input,x):
    
    """
    This image (or value) is divided to the input
    """
    
    cmd_str = BASE_STR
    
    if np.isscalar(x):
        cmd_str += f'-div {x}'
    else:
        cmd_str += f'-div {NAME}x.nii'
        write_nifti(f'{NAME}x.nii', x)

    write_nifti(f'{NAME}input', input)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def rms(input, x, verbose=False):
    
    """
    Compute the mean rms between both images
    """

    cmd_str = BASE_STR
    cmd_str += f'-rms {NAME}x.nii'
    
    write_nifti(f'{NAME}input.nii', input)
    write_nifti(f'{NAME}x.nii', x)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def bin(input):
    
    """
    Binarise the input image (val!=0?val=1:val=0)
    """
    
    cmd_str = BASE_STR
    cmd_str +=  '-bin'

    write_nifti(f'{NAME}input.nii', input)
    
    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def thr(input, thr, verbose=False):
    
    """
    Threshold the input image (val<thr?val=0:val=1)
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-thr {thr}'

    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def nan(input, x, verbose=False):
    
    """
    This image is used to mask the input image.
	Voxels outside of the mask are set to nan
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-nan {NAME}x.nii'
    
    write_nifti(f'{NAME}input.nii', input)
    write_nifti(f'{NAME}x.nii', x)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def iso(input, verbose=False):
    
    """
    The resulting image is made isotropic
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-iso'
    
    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

def noscl(input, verbose=False):
    
    """
	The scl_slope and scl_inter are set to 1 and 0 respectively
    """
    
    cmd_str = BASE_STR
    cmd_str += f'-noscl'
    
    write_nifti(f'{NAME}input.nii', input)
    
    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

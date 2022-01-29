import shlex
import numpy as np
import nibabel as nib
import subprocess as sp

def read_nifti(name: str)  -> np.array:
    
    try:
        array = np.array(nib.load(name).dataobj)
        return array
    except FileNotFoundError:
        return None
    
def write_nifti(name, array, __affine=None) -> bool:
    
    try:
        nib.save(nib.Nifti1Image(array, affine=__affine), name)
        return True
    except Exception as e:
        raise e
    else:
        return False

def read_txt(name: str):
    
    try:
        array = np.loadtxt(name)
        return array
    except FileNotFoundError:
        return None

def write_txt(name: str, array) -> bool:
    
    try:
        np.savetxt(name, array)
        return True
    except Exception as e:
        raise e
    else:
        return False

def call_niftyreg(cmd_str: str, verbose=False) -> bool:
    
    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    
    if verbose:
        for line in stdout.decode(encoding='utf-8').split('\n'):
            print(line)
        for line in stderr.decode(encoding='utf-8').split('\n'):
            print(line)
       
    if stderr:
        return False
    else:
        return True

def get_help_string(tool: str) -> str:
    
    cmd_str = f'{tool} --help'
    
    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    
    if stdout:
        return stdout
    else:
        return None

import numpy as np
import tempfile as tmp

from ..utils import read_nifti, write_nifti, call_niftyreg

DEBUG=False
NAME=tmp.NamedTemporaryFile().name

def avg():
    raise NotImplementedError
def avg_lts():
    raise NotImplementedError
def avg_tran():
    raise NotImplementedError
def demean1():
    raise NotImplementedError
def demean2():
    raise NotImplementedError
def demean3():
    raise NotImplementedError

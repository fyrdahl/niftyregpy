import shlex
import signal
import subprocess as sp

import nibabel as nib
import numpy as np


def read_nifti(name: str, output_nan=False) -> np.array:

    try:
        array = np.ascontiguousarray(nib.load(name).dataobj)
        if any_nans(array) and not output_nan:
            return np.nan_to_num(array, nan=0.0)
        else:
            return array

    except FileNotFoundError:
        return None


def write_nifti(name, array, __affine=None) -> bool:

    try:
        nib.save(
            nib.Nifti1Image(np.ascontiguousarray(array), affine=__affine),
            name,
        )
        return True
    except Exception as e:
        raise e


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


def call_niftyreg(cmd_str: str, verbose=False, output_stdout=False) -> bool:

    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    if verbose:
        print(cmd_str)
        for line in stdout.decode(encoding="utf-8").split("\n"):
            print(line)
        for line in stderr.decode(encoding="utf-8").split("\n"):
            print(line)

    if stderr or p.returncode == -signal.SIGSEGV:
        return False

    if output_stdout:
        return stdout.decode(encoding="utf-8")

    return True


def get_help_string(tool: str) -> str:

    cmd_str = f"{tool} --help"

    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, _ = p.communicate()

    if stdout:
        return stdout

    return None


def _any_nans(a) -> bool:
    for x in a:
        if np.isnan(x):
            return True
    return False


def any_nans(a) -> bool:
    if not a.dtype.kind == "f":
        return False
    return _any_nans(a.flat)

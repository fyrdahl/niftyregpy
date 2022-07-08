import shlex
import signal
import subprocess as sp

import nibabel as nib
import numpy as np


def read_nifti(name: str, output_nan=False) -> np.array:

    try:
        array = np.ascontiguousarray(nib.load(name).dataobj)
        if not output_nan:
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

    cmd_str = f"{tool} -h"

    try:
        p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = p.communicate()

        if stdout and not stderr:
            return stdout.decode(encoding="utf-8")

    except FileNotFoundError:
        return None


def is_function_available(tool: str, name: str) -> bool:

    cmd_str = f"{tool} -h"

    try:
        p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = p.communicate()

        if stderr or "-" + name.lower() not in stdout.decode(encoding="utf-8").lower():
            raise FileNotFoundError

        return True

    except FileNotFoundError:
        return False

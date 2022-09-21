# -*- coding: utf-8 -*-
"""Utility functions.
"""

import random
import shlex
import signal
import subprocess as sp

import nibabel as nib
import numpy as np


def read_nifti(name: str, output_nan=False) -> np.array:

    try:
        array = np.ascontiguousarray(nib.load(name).dataobj)
        return array if output_nan else np.nan_to_num(array, nan=0.0)
    except OSError:
        return None


def write_nifti(name, array, _affine=np.eye(4)) -> bool:

    try:
        img = nib.Nifti1Image(array, affine=_affine)
        nib.save(img, name)
        return True
    except Exception as e:
        print(e)
        return False


def read_txt(name: str):

    try:
        return np.loadtxt(name)
    except OSError:
        return None


def write_txt(name: str, array) -> bool:

    try:
        np.savetxt(name, array)
        return True
    except Exception as e:
        print(e)
        return False


def call_niftyreg(cmd_str: str, verbose=False, output_stdout=False) -> bool:

    if not cmd_str.startswith("reg_"):
        return False

    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    if verbose:
        print(cmd_str)
        for line in stdout.decode(encoding="utf-8").split("\n"):
            print(line)

    if stderr:
        for line in stderr.decode(encoding="utf-8").split("\n"):
            print(line)

    if stderr or p.returncode == -signal.SIGSEGV:
        return False

    if output_stdout:
        return stdout.decode(encoding="utf-8")

    return True


def get_help_string(tool: str) -> str:

    if not tool.startswith("reg_"):
        raise FileNotFoundError

    cmd_str = f"{tool} -h"

    p = sp.Popen(shlex.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    if stdout and not stderr:
        return stdout.decode(encoding="utf-8")
    else:
        raise FileNotFoundError


def is_function_available(tool: str, name: str) -> bool:

    try:
        stdout = get_help_string(tool)

        if f"-{name.lower()}" not in stdout.lower():
            raise FileNotFoundError

        return True

    except FileNotFoundError:
        return False


def create_test_image(length=256, blobs=6, min_rad=3, max_rad=32, dtype=float):

    array = np.zeros((length, length))

    for _ in range(blobs):
        r = random.uniform(a=min_rad, b=max_rad)
        c = (random.uniform(a=r, b=length - r), random.uniform(a=r, b=length - r))
        array += _create_circle(length, c=c, r=r, dtype=dtype)

    return (array > 0).astype(dtype)


def _create_circle(length, c=None, r=None, dtype=float):

    Y, X = np.ogrid[:length, :length]
    circ = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2) <= r

    return circ.astype(dtype)

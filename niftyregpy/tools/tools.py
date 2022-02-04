import tempfile as tmp
from os import path

import numpy as np

from ..utils import call_niftyreg, read_nifti, write_nifti


def float(input, verbose=False):

    """
    The input image is converted to float
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = (
            f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -float "
        )

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def down(input, verbose=False):

    """
    The input image is downsampled 2 times
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = (
            f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -down "
        )
        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def smoS(input, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input image is smoothed using a cubic b-spline kernel
    """
    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "
        cmd_str += f"-smoS {sx} {sy} {sz} "

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def smoG(input, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input image is smoothed using a Gaussian kernel
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "
        cmd_str += f"-smoG {sx} {sy} {sz} "

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def smoL(input, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input label image is smoothed using a Gaussian kernel
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "
        cmd_str += f"-smoL {sx} {sy} {sz} "

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def add(input, x, verbose=False):

    """
    This image (or value) is added to the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "

        if np.isscalar(x):
            cmd_str += f"-add {x}"
        else:
            cmd_str += f"-add {tmp_folder}x.nii"
            write_nifti(f"{tmp_folder}x.nii", x)

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def sub(input, x, verbose=False):

    """
    This image (or value) is subtracted from the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "

        if np.isscalar(x):
            cmd_str += f"-sub {x}"
        else:
            cmd_str += f"-sub {tmp_folder}x.nii"
            write_nifti(f"{tmp_folder}x.nii", x)

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def mul(input, x, verbose=False):

    """
    This image (or value) is multiplied with the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "

        if np.isscalar(x):
            cmd_str += f"-mul {x}"
        else:
            cmd_str += f"-mul {tmp_folder}x.nii"
            write_nifti(f"{tmp_folder}x.nii", x)

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def div(input, x, verbose=False):

    """
    This image (or value) is divided to the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii "

        if np.isscalar(x):
            cmd_str += f"-div {x}"
        else:
            cmd_str += f"-div {tmp_folder}x.nii"
            write_nifti(f"{tmp_folder}x.nii", x)

        write_nifti(f"{tmp_folder}input", input)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def rms(input, x, verbose=False):

    """
    Compute the mean rms between both images
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -rms {tmp_folder}x.nii "

        write_nifti(f"{tmp_folder}input.nii", input)
        write_nifti(f"{tmp_folder}x.nii", x)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def bin(input, verbose=False):

    """
    Binarise the input image (val!=0?val=1:val=0)
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = (
            f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -bin "
        )

        write_nifti(f"{tmp_folder}input.nii", input)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def thr(input, thr, verbose=False):

    """
    Threshold the input image (val<thr?val=0:val=1)
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -thr {thr} "

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def nan(input, x, verbose=False):

    """
    This image is used to mask the input image.
    Voxels outside of the mask are set to nan
    """
    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -nan {tmp_folder}x.nii "

        write_nifti(f"{tmp_folder}input.nii", input)
        write_nifti(f"{tmp_folder}x.nii", x)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def iso(input, verbose=False):

    """
    The resulting image is made isotropic
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = (
            f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -iso "
        )

        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def noscl(input, verbose=False):

    """
    The scl_slope and scl_inter are set to 1 and 0 respectively
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = (
            f"reg_tools -in {tmp_folder}input.nii -out {tmp_folder}output.nii -noscl "
        )
        write_nifti(f"{tmp_folder}input.nii", input)

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None

import builtins
import tempfile as tmp
from os import path

import numpy as np

from ..utils import call_niftyreg, is_function_available, read_nifti, write_nifti


def float(input, output=None, verbose=False):

    """
    The input image is converted to float
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = f"reg_tools -in {path.join(tmp_folder, 'input.nii')}"

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += " -float"

        write_nifti(path.join(tmp_folder, "input.nii"), input)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def down(input, output=None, verbose=False):

    """
    The input image is downsampled 2 times
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = f"reg_tools -in {path.join(tmp_folder, 'input.nii')}"

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += " -down"

        write_nifti(path.join(tmp_folder, "input.nii"), input)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def smoS(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input image is smoothed using a cubic b-spline kernel
    """
    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += f" -smoS {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def smoG(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input image is smoothed using a Gaussian kernel
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += f" -smoG {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def smoL(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input label image is smoothed using a Gaussian kernel
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += f" -smoL {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def add(input, x, output=None, verbose=False):

    """
    This image (or value) is added to the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output

        if np.isscalar(x):
            cmd_str += f" -add {x}"
        else:
            write_nifti(path.join(tmp_folder, "x.nii"), x)
            cmd_str += " -add " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def sub(input, x, output=None, verbose=False):

    """
    This image (or value) is subtracted from the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output

        if np.isscalar(x):
            cmd_str += f" -sub {x}"
        else:
            write_nifti(path.join(tmp_folder, "x.nii"), x)
            cmd_str += " -sub " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def mul(input, x, output=None, verbose=False):

    """
    This image (or value) is multiplied with the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output

        if np.isscalar(x):
            cmd_str += f" -mul {x}"
        else:
            write_nifti(path.join(tmp_folder, "x.nii"), x)
            cmd_str += " -mul " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def div(input, x, output=None, verbose=False):

    """
    This image (or value) is divided to the input
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output

        if np.isscalar(x):
            cmd_str += f" -div {x}"
        else:
            write_nifti(path.join(tmp_folder, "x.nii"), x)
            cmd_str += " -div " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def rms(input, x, output=None, verbose=False):

    """
    Compute the mean rms between both images
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        write_nifti(path.join(tmp_folder, "x.nii"), x)

        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += " -rms " + path.join(tmp_folder, "x.nii")

        out = call_niftyreg(cmd_str, verbose, output_stdout=True)

        if out:
            return builtins.float(out)

    return None


def bin(input, output=None, verbose=False):

    """
    Binarise the input image (val!=0?val=1:val=0)
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += " -bin"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def thr(input, thr, output=None, verbose=False):

    """
    Threshold the input image (val<thr?val=0:val=1)
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += f" -thr {thr}"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

        return None


def nan(input, x, output=None, verbose=False):

    """
    This image is used to mask the input image.
    Voxels outside of the mask are set to nan
    """
    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        write_nifti(path.join(tmp_folder, "x.nii"), x)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += " -nan " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output, output_nan=True)

    return None


def iso(input, output=None, verbose=False):

    """
    The resulting image is made isotropic
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += " -iso"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def noscl(input, output=None, verbose=False):

    """
    The scl_slope and scl_inter are set to 1 and 0 respectively
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += " -noscl"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def chgres(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    Resample the input image to the specified resolution (in mm)
    """
    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "chgres"):
        return NotImplemented

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += f" -chgres {sx} {sy} {sz}"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def rmNanInf(input, output=None, x=0.0, verbose=False):

    """
    Remove the nan and inf from the input image and replace them by the specified value
    """

    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "rmNanInf"):
        return NotImplemented

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += f" -rmNanInf {x}"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def testActiveBlocks(input, output=None, verbose=False):

    """
    Generate an image highlighting the active blocks for reg_aladin (block variance is shown)
    """

    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "testActiveBlocks"):
        return NotImplemented

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + output
        cmd_str += " -testActiveBlocks"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None

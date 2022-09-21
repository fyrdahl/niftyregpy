import builtins
import tempfile as tmp
from os import path

import numpy as np

from ..utils import call_niftyreg, is_function_available, read_nifti, write_nifti


def float(input, output=None, verbose=False):

    """
    The input image is converted to float.

    Args:
        input (array): Input array to be converted.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Converted input array.

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
    The input image is downsampled 2 times.

    Args:
        input (array): Input array to be downsampled.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Input array downsampled 2 times.

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
    The input image is smoothed using a cubic b-spline kernel.

    Args:
        input (array): Input array to be smoothed.
        sx (float): Smoothing in x.
        sy (float): Smoothing in y.
        sz (float): Smoothing in z.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Input array smoothed using a cubic b-spline kernel.

    """
    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += f" -smoS {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def smoG(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input image is smoothed using a Gaussian kernel.

    Args:
        input (array): Input array to be smoothed.
        sx (float): Smoothing in x.
        sy (float): Smoothing in y.
        sz (float): Smoothing in z.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Input array smoothed using a Gaussian kernel.

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += f" -smoG {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def smoL(input, output=None, sx=0.0, sy=0.0, sz=0.0, verbose=False):

    """
    The input label image is smoothed using a Gaussian kernel.

    Args:
        input (array): Input array to be smoothed.
        sx (float): Smoothing in x.
        sy (float): Smoothing in y.
        sz (float): Smoothing in z.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Input label array smoothed using a Gaussian kernel.

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += f" -smoL {sx} {sy} {sz} "

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def add(input, x, output=None, verbose=False):

    """
    This image (or value) is added to the input.

    Args:
        input (array): Input array.
        x (array/float): Image or value to be added to the input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Sum of input array and image (or value).

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"

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

    Args:
        input (array): Input array.
        x (array/float): Image or value to be subtracted from the input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Difference of input array and image (or value).

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"

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

    Args:
        input (array): Input array.
        x (array/float): Image or value to be multiplied with the input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Product of input array and image (or value).

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"

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

    Args:
        input (array): Input array.
        x (array/float): Image or value to divide the input array by.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Product of input array and image (or value).

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"

        if np.isscalar(x):
            cmd_str += f" -div {x}"
        else:
            write_nifti(path.join(tmp_folder, "x.nii"), x)
            cmd_str += " -div " + path.join(tmp_folder, "x.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def rms(input, input2, output=None, verbose=False):

    """
    Compute the mean rms between both images

    Args:
        input (array): First input array.
        input2 (array): Second input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        write_nifti(path.join(tmp_folder, "input2.nii"), input2)

        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += " -rms " + path.join(tmp_folder, "input2.nii")

        out = call_niftyreg(cmd_str, verbose, output_stdout=True)

        if out:
            return builtins.float(out)

    return None


def bin(input, output=None, verbose=False):

    """
    Binarize the input image (val!=0?val=1:val=0)

    Args:
        input (array): Input array to be binarized.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
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

    Args:
        input (array): Input array to be thresholded.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
    """

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += f" -thr {thr}"

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def nan(input, mask, output=None, verbose=False):

    """
    Mask the input image. Voxels outside of the mask are set to NaN.

    Args:
        input (array): Input array.
        mask (array): Input mask, values outside mask is set to NaN.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
    """
    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_tools"

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        write_nifti(path.join(tmp_folder, "mask.nii"), mask)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += " -out " + path.join(tmp_folder, "output.nii")
        cmd_str += " -nan " + path.join(tmp_folder, "mask.nii")

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output, output_nan=True)

    return None


def iso(input, output=None, verbose=False):

    """
    The resulting image is made isotropic

    Args:
        input (array): Input array to be made isotropic.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
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

    Args:
        input (array): Input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
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


def chgres(input, sx=0.0, sy=0.0, sz=0.0, output=None, verbose=False):

    """
    Resample the input image to the specified resolution (in mm)

    Args:
        input (array): Input array to be resampled.
        sx (float): Resolution in x.
        sy (float): Resolution in y.
        sz (float): Resolution in z.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
    """
    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "chgres"):
        return NotImplemented  # pragma: no cover

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += f" -chgres {sx} {sy} {sz}"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def rmNanInf(input, x=0.0, output=None, verbose=False):

    """
    Remove NaN and Inf values from the input image and replace with specified value

    Args:
        input (array): Input array.
        x (float): Value that should be used to replace NaN and Inf (default = 0.0).
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
    """

    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "rmNanInf"):
        return NotImplemented  # pragma: no cover

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += f" -rmNanInf {x}"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None


def testActiveBlocks(input, output=None, verbose=False):

    """
    Generate image showing the active blocks for reg.aladin (block variance is shown)

    Args:
        input (array): Input array.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).
    """

    cmd_str = "reg_tools"

    if not is_function_available(cmd_str, "testActiveBlocks"):
        return NotImplemented  # pragma: no cover

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "input.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "input.nii")

        if output is None:
            output = path.join(tmp_folder, "output.nii")

        cmd_str += f" -out {output}"
        cmd_str += " -testActiveBlocks"

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(output)

    return None

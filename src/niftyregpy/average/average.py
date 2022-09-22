import os
import tempfile as tmp

from ..utils import call_niftyreg, read_nifti, read_txt, write_nifti, write_txt


def avg(input, output=None, verbose=False):

    """
    If input are images, their intensities are averaged.

    Args:
        input (tuple): Input images or affines to be averaged.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged input array.

    If input are affine matrices, the result will correspond to
        >>> from scipy.linalg import logm, expm
        >>> out = expm(sum(logm(aff) for aff in input) / len(input))
    """

    if all(a.shape == (4, 4) for a in input):
        return _avg_txt(input, output, verbose)
    else:
        return _avg_nii(input, output, verbose)


def _avg_txt(input, output=None, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.txt")

        cmd_str = f"reg_average {output}"
        cmd_str += " -avg "

        for i, x in enumerate(input):
            write_txt(os.path.join(tmp_folder, f"avg_{i}.txt"), x)
            cmd_str += os.path.join(tmp_folder, f"avg_{i}.txt") + " "

        return read_txt(output) if call_niftyreg(cmd_str, verbose) else None


def _avg_nii(input, output=None, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -avg "

        for i, x in enumerate(input):
            write_nifti(os.path.join(tmp_folder, f"avg_{i}.nii"), x)
            cmd_str += os.path.join(tmp_folder, f"avg_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def avg_lts(aff, output=None, verbose=False):

    """
    Estimate the robust average affine matrix by considering half of the
    matrices as outliers.

    Args:
        aff (tuple): Affines to be averaged.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged input array.

    """

    assert all(a.shape == (4, 4) for a in aff), "Not affine matrices"

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.txt")

        cmd_str = f"reg_average {output}"
        cmd_str += " -avg_lts "

        for i, x in enumerate(aff):
            write_txt(os.path.join(tmp_folder, f"avg_{i}.txt"), x)
            cmd_str += os.path.join(tmp_folder, f"avg_{i}.txt") + " "

        return read_txt(output) if call_niftyreg(cmd_str, verbose) else None


def avg_tran(ref, tran, flo, output=None, verbose=False):

    """
    All input images are resampled into the space of ``ref`` and
    averaged. A cubic spline interpolation scheme is used for resampling.

    Args:
        ref (array): Reference image.
        tran (tuple): Transforms.
        flo (tuple): Floating images.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged floating images.

    """

    assert len(flo) == len(
        tran
    ), "Non-matching number of floating images and transforms"

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -avg_tran "

        write_nifti(os.path.join(tmp_folder, "ref.nii"), ref)
        cmd_str += os.path.join(tmp_folder, "ref.nii") + " "

        for i, x in enumerate(zip(tran, flo)):
            write_nifti(os.path.join(tmp_folder, f"avg_tran_{i}.nii"), x[0])
            write_nifti(os.path.join(tmp_folder, f"avg_flo_{i}.nii"), x[1])
            cmd_str += os.path.join(tmp_folder, f"avg_tran_{i}.nii") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_flo_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def demean1(ref, aff, flo, output=None, verbose=False):

    """
    Average images and demean average image that have affine transformations to
    a common space

    The demean1 option enforces the mean of all affine matrices to have a
    Jacobian determinant equal to one. This is done by computing the average
    transformation by considering only the scaling and shearing arguments.
    The inverse of this computed average matrix is then removed to all input
    affine matrix before resampling all floating images to the user-defined
    reference space

    Args:
        ref (array): Reference image.
        aff (tuple): Affines.
        flo (tuple): Floating images.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged floating images.

    """

    assert len(aff) == len(flo), "Non-matching number of floating images and transforms"

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -demean1 "

        write_nifti(os.path.join(tmp_folder, "ref.nii"), ref)
        cmd_str += os.path.join(tmp_folder, "ref.nii ")

        for i, x in enumerate(zip(aff, flo)):
            write_txt(os.path.join(tmp_folder, f"avg_aff_{i}.txt"), x[0])
            write_nifti(os.path.join(tmp_folder, f"avg_flo_{i}.nii"), x[1])
            cmd_str += os.path.join(tmp_folder, f"avg_aff_{i}.txt") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_flo_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def demean2(ref, tran, flo, output=None, verbose=False):

    """
    Average images and demean average image that have non-rigid
    transformations to a common space.

    Args:
        ref (array): Reference image.
        tran (tuple): Transforms.
        flo (tuple): Floating images.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged floating images.

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -demean2 "

        write_nifti(os.path.join(tmp_folder, "ref.nii"), ref)
        cmd_str += os.path.join(tmp_folder, "ref.nii ")

        for i, x in enumerate(zip(tran, flo)):
            write_nifti(os.path.join(tmp_folder, f"avg_tran_{i}.nii"), x[0])
            write_nifti(os.path.join(tmp_folder, f"avg_flo_{i}.nii"), x[1])
            cmd_str += os.path.join(tmp_folder, f"avg_tran_{i}.nii") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_flo_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def demean3(ref, aff, tran, flo, output=None, verbose=False):

    """
    Average images and demean average image that have linear and non-rigid
    transformations to a common space.

    Args:
        ref (array): Reference image.
        aff (tuple): Affines.
        tran (tuple): Transforms.
        flo (tuple): Floating images.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged floating images.

    """

    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -demean3 "

        write_nifti(os.path.join(tmp_folder, "ref.nii"), ref)
        cmd_str += os.path.join(tmp_folder, "ref.nii") + " "

        for i, x in enumerate(zip(aff, tran, flo)):
            write_txt(os.path.join(tmp_folder, f"avg_aff_{i}.txt"), x[0])
            write_nifti(os.path.join(tmp_folder, f"avg_tran_{i}.nii"), x[1])
            write_nifti(os.path.join(tmp_folder, f"avg_flo_{i}.nii"), x[2])
            cmd_str += os.path.join(tmp_folder, f"avg_aff_{i}.txt") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_tran_{i}.nii") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_flo_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None


def demean_noaff(ref, aff, tran, flo, output=None, verbose=False):

    """
    Same as the demean expect that the specified affine is removed from the
    non-linear (euclidean) transformation.

    Args:
        ref (array): Reference image.
        aff (tuple): Affines.
        tran (tuple): Transforms.
        flo (tuple): Floating images.
        output (string): Specify output file (optional).
        verbose (bool): Verbose output (default = False).

    Returns:
        array: Averaged floating images.

    """
    with tmp.TemporaryDirectory() as tmp_folder:

        if output is None:
            output = os.path.join(tmp_folder, "output.nii")

        cmd_str = f"reg_average {output}"
        cmd_str += " -demean_noaff "

        write_nifti(os.path.join(tmp_folder, "ref.nii"), ref)
        cmd_str += os.path.join(tmp_folder, "ref.nii") + " "

        for i, x in enumerate(zip(aff, tran, flo)):
            write_txt(os.path.join(tmp_folder, f"avg_aff_{i}.txt"), x[0])
            write_nifti(os.path.join(tmp_folder, f"avg_tran_{i}.nii"), x[1])
            write_nifti(os.path.join(tmp_folder, f"avg_flo_{i}.nii"), x[2])
            cmd_str += os.path.join(tmp_folder, f"avg_aff_{i}.txt") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_tran_{i}.nii") + " "
            cmd_str += os.path.join(tmp_folder, f"avg_flo_{i}.nii") + " "

        return read_nifti(output) if call_niftyreg(cmd_str, verbose) else None

import tempfile as tmp
from os import path

from ..utils import call_niftyreg, read_nifti, read_txt, write_nifti, write_txt


def avg(input):

    """
    If the input are images, the intensities are averaged
    If the input are affine matrices;
        out=expm((logm(M1)+logm(M2)+...+logm(MN))/N)
    """

    if all([a.shape == (4, 4) for a in input]):
        return _avg_txt(input)
    else:
        return _avg_nii(input)


def _avg_txt(input, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.txt  -avg "

        for i, x in enumerate(input):
            write_txt(f"{tmp_folder}avg_{i}.txt", x)
            cmd_str += f"{tmp_folder}avg_{i}.txt "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_txt(f"{tmp_folder}output.txt")
        else:
            return None


def _avg_nii(input, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.nii  -avg "

        for i, x in enumerate(input):
            write_nifti(f"{tmp_folder}avg_{i}.nii", x)
            cmd_str += f"{tmp_folder}avg_{i}.nii "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def avg_lts(aff, verbose=False):

    """
    Estimate the robust average affine matrix by considering outliers.
    """

    assert all([a.shape == (4, 4) for a in aff]), "Not affine matrices"

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.txt  -avg "

        for i, x in enumerate(aff):
            write_txt(f"{tmp_folder}avg_{i}.txt", x)
            cmd_str += f"{tmp_folder}avg_{i}.txt "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_txt(f"{tmp_folder}output.txt")
        else:
            return None


def avg_tran(ref, tran, flo, verbose=False):

    """
    All input images are resampled into the space of <reference image> and
    averaged A cubic spline interpolation scheme is used for resampling.
    """

    assert len(flo) == len(
        tran
    ), "Non-matching number of floating images and transforms"

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.nii  -avg_tran "

        write_nifti(f"{tmp_folder}ref.nii", ref)
        cmd_str += f"{tmp_folder}ref.nii "

        for i, x in enumerate(zip(tran, flo)):
            write_nifti(f"{tmp_folder}avg_tran_{i}.nii", x[0])
            write_nifti(f"{tmp_folder}avg_flo_{i}.nii", x[1])
            cmd_str += f"{tmp_folder}avg_tran_{i}.nii {tmp_folder}avg_flo_{i}.nii "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def demean1(ref, aff, flo, verbose=False):

    """
    The demean1 option enforces the mean of all affine matrices to have a
    Jacobian determinant equal to one. This is done by computing the average
    transformation by considering only the scaling and shearing arguments.
    The inverse of this computed average matrix is then removed to all input
    affine matrix before resampling all floating images to the user-defined
    reference space
    """

    assert len(aff) == len(flo), "Non-matching number of floating images and transforms"

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.nii -demean1 "

        write_nifti(f"{tmp_folder}ref.nii", ref)
        cmd_str += f"{tmp_folder}ref.nii "

        for i, x in enumerate(zip(aff, flo)):
            write_txt(f"{tmp_folder}avg_aff_{i}.txt", x[0])
            write_nifti(f"{tmp_folder}avg_flo_{i}.nii", x[1])
            cmd_str += f"{tmp_folder}avg_aff_{i}.nii {tmp_folder}avg_flo_{i}.nii "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def demean2(ref, tran, flo, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.nii -demean2 "

        write_nifti(f"{tmp_folder}ref.nii", ref)
        cmd_str += f"{tmp_folder}ref.nii "

        for i, x in enumerate(zip(tran, flo)):
            write_nifti(f"{tmp_folder}avg_tran_{i}.nii", x[0])
            write_nifti(f"{tmp_folder}avg_flo_{i}.nii", x[1])
            cmd_str += f"{tmp_folder}avg_tran_{i}.nii {tmp_folder}avg_flo_{i}.nii "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None


def demean3(ref, aff, tran, flo, verbose=False):

    with tmp.TemporaryDirectory() as tmp_folder:

        tmp_folder += path.sep
        cmd_str = "reg_average "
        cmd_str += f"{tmp_folder}output.nii -demean3 "

        write_nifti(f"{tmp_folder}ref.nii", ref)
        cmd_str += f"{tmp_folder}ref.nii "

        for i, x in enumerate(zip(aff, tran, flo)):
            write_txt(f"{tmp_folder}avg_aff_{i}.txt", x[0])
            write_nifti(f"{tmp_folder}avg_tran_{i}.nii", x[1])
            write_nifti(f"{tmp_folder}avg_flo_{i}.nii", x[2])
            cmd_str += f"{tmp_folder}avg_aff_{i}.txt  {tmp_folder}avg_tran_{i}.nii {tmp_folder}avg_flo_{i}.nii "

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(f"{tmp_folder}output.nii")
        else:
            return None

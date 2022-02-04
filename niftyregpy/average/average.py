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


def avg_lts(input, verbose=False):

    """
    It will estimate the robust average affine matrix by considering outliers.
    """

    assert all([a.shape == (4, 4) for a in input]), "Not affine matrices"

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


def avg_tran():
    raise NotImplementedError


def demean1():
    raise NotImplementedError


def demean2():
    raise NotImplementedError


def demean3():
    raise NotImplementedError

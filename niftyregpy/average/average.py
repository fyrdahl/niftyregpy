import tempfile as tmp

from ..utils import call_niftyreg, read_nifti, read_txt, write_nifti, write_txt

DEBUG = False
NAME = tmp.NamedTemporaryFile().name
BASE_STR = "reg_average "


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


def _avg_txt(input):

    cmd_str = BASE_STR
    cmd_str += f"{NAME}output.txt  -avg "

    for i, x in enumerate(input):
        write_txt(f"{NAME}avg_{i}.txt", x)
        cmd_str += f"{NAME}avg_{i}.txt "

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_txt(f"{NAME}output.txt")
    else:
        return None


def _avg_nii(input):

    cmd_str = BASE_STR
    cmd_str += f"{NAME}output.nii  -avg "

    for i, x in enumerate(input):
        write_nifti(f"{NAME}avg_{i}.nii", x)
        cmd_str += f"{NAME}avg_{i}.nii "

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f"{NAME}output.nii")
    else:
        return None


def avg_lts(input):

    """
    It will estimate the robust average affine matrix by considering outliers.
    """

    assert all([a.shape == (4, 4) for a in input]), "Not affine matrices"

    cmd_str = BASE_STR
    cmd_str += f"{NAME}output.txt  -avg "

    for i, x in enumerate(input):
        write_txt(f"{NAME}avg_{i}.txt", x)
        cmd_str += f"{NAME}avg_{i}.txt "

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_txt(f"{NAME}output.txt")
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

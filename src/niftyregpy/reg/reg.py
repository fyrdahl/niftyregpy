import builtins
import tempfile as tmp
from os import path

import numpy as np

from ..average import avg, avg_lts, avg_tran, demean1, demean2, demean3
from ..utils import call_niftyreg, read_nifti, read_txt, write_nifti, write_txt


def aladin(
    ref,
    flo,
    noSym=None,
    rigOnly=None,
    affDirect=None,
    aff=None,
    inaff=None,
    rmask=None,
    fmask=None,
    res=None,
    maxit=None,
    ln=None,
    lp=None,
    smooR=None,
    smooF=None,
    refLowThr=None,
    refUpThr=None,
    floLowThr=None,
    floUpThr=None,
    nac=None,
    cog=None,
    interp=None,
    iso=None,
    pad=None,
    pv=None,
    pi=None,
    speeeeed=None,
    omp=None,
    verbose=False,
):

    """
    Block Matching algorithm for global registration.
    Based on Ourselin et al., "Reconstructing a 3D structure from serial
    histological sections" Image and Vision Computing, 2001
    """
    # usage_string = "reg_aladin -ref <filename> -flo <filename> [OPTIONS]"

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_aladin"

        write_nifti(path.join(tmp_folder, "ref"), ref)
        write_nifti(path.join(tmp_folder, "flo"), flo)

        cmd_str += (
            " -ref "
            + path.join(tmp_folder, "ref.nii")
            + " -flo "
            + path.join(tmp_folder, "flo.nii")
        )

        opts_str = ""

        if res is None:
            opts_str += " -res " + path.join(tmp_folder, "res.nii")

        if aff is None:
            opts_str += " -aff " + path.join(tmp_folder, "aff.txt")

        if noSym is True:
            opts_str += " -noSym"

        if rigOnly is True:
            opts_str += " -rigOnly"

        if affDirect is True:
            opts_str += " -affDirect"

        if inaff is not None:
            write_txt(path.join(tmp_folder, "inaff.txt"), inaff)
            opts_str += " -inaff " + path.join(tmp_folder, "inaff.txt")

        if rmask is not None:
            write_nifti(path.join(tmp_folder, "rmask.nii"), rmask.astype(float))
            opts_str += " -rmask " + path.join(tmp_folder, "rmask.nii")

        if fmask is not None:
            write_nifti(path.join(tmp_folder, "fmask.nii"), fmask.astype(float))
            opts_str += " -fmask " + path.join(tmp_folder, "fmask.nii")

        if maxit is not None:
            opts_str += f" -maxit {maxit}"

        if ln is not None:
            opts_str += f" -ln {ln}"

        if lp is not None:
            opts_str += f" -lp {lp}"

        if smooR is not None:
            opts_str += f" -smooR {smooR}"

        if smooF is not None:
            opts_str += f" -smooF {smooF}"

        if refLowThr is not None:
            opts_str += f" -refLowThr {refLowThr}"

        if refUpThr is not None:
            opts_str += f" -refUpThr {refUpThr}"

        if floLowThr is not None:
            opts_str += f" -floLowThr {floLowThr}"

        if floUpThr is not None:
            opts_str += f" -floUpThr {floUpThr}"

        if nac is True:
            opts_str += " -nac"

        if cog is True:
            opts_str += " -cog"

        if interp is True:
            opts_str += " -interp"

        if iso is True:
            opts_str += " -iso"

        if pad is not None:
            opts_str += f" -pad {int(pad)}"

        if pv is not None:
            opts_str += f" -pv {pv}"

        if pi is not None:
            opts_str += f" -pi {pi}"

        if speeeeed is True:
            opts_str += " -speeeeed"

        if omp is not None:
            opts_str += f" -omp {omp}"

        if not verbose:
            opts_str += " -voff"

        cmd_str += opts_str

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(path.join(tmp_folder, "res.nii")), read_txt(
                path.join(tmp_folder, "aff.txt")
            )
        else:
            return None


def f3d(
    ref=None,
    flo=None,
    aff=None,
    incpp=None,
    cpp=None,
    res=None,
    rmask=None,
    smooR=None,
    smooF=None,
    rLwTh=None,
    rUpTh=None,
    fLwTh=None,
    fUpTh=None,
    sx=None,
    sy=None,
    sz=None,
    be=None,
    le=None,
    l2=None,
    jl=None,
    noAppJL=None,
    nmi=None,
    rbn=None,
    fbn=None,
    lncc=None,
    ssd=None,
    kld=None,
    amc=None,
    maxit=None,
    ln=None,
    lp=None,
    nopy=None,
    noConj=None,
    pert=None,
    vel=None,
    fmask=None,
    omp=None,
    mem=None,
    gpu=None,
    smoothGrad=None,
    pad=None,
    verbose=False,
):

    """
    Fast Free-Form Deformation algorithm for non-rigid registration. This
    implementation is a re-factoring of Daniel Rueckert' 99 TMI work. The code
    is presented in Modat et al., "Fast Free-Form Deformation using graphics
    processing units", CMPB, 2010

    Cubic B-Spline are used to deform a source image in order to optimise a
    objective function based on the Normalised Mutual Information and a penalty
    term. The penalty term could be either the bending energy or the squared
    Jacobian determinant log.
    """

    # usage_string = "reg_f3d -ref <filename> -flo <filename> [OPTIONS]"

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_f3d"

        write_nifti(path.join(tmp_folder, "ref"), ref)
        write_nifti(path.join(tmp_folder, "flo"), flo)

        cmd_str += (
            " -ref "
            + path.join(tmp_folder, "ref.nii")
            + " -flo "
            + path.join(tmp_folder, "flo.nii")
        )

        opts_str = ""

        if res is None:
            res = path.join(tmp_folder, "res.nii")
        if cpp is None:
            cpp = path.join(tmp_folder, "cpp.nii")

        opts_str += f" -res {res}"
        opts_str += f" -cpp {cpp}"

        if aff is not None:
            write_txt(path.join(tmp_folder, "aff.txt"), aff)
            opts_str += " -aff " + path.join(tmp_folder, "aff.txt")

        if incpp is not None:
            opts_str += f" -incpp {incpp}"

        if rmask is not None:
            write_nifti(path.join(tmp_folder, "rmask.nii"), rmask.astype(float))
            opts_str += " -rmask " + path.join(tmp_folder, "rmask.nii")

        if smooR is not None:
            opts_str += f" -smooR {smooR}"

        if smooF is not None:
            opts_str += f" -smooF {smooF}"

        if rLwTh is not None:
            if type(rLwTh) is tuple:
                opts_str += f" -rLwTh {' '.join(str(x) for x in rLwTh)}"
            else:
                opts_str += f" --rLwTh {rLwTh}"

        if rUpTh is not None:
            if type(rUpTh) is tuple:
                opts_str += f" -rUpTh {' '.join(str(x) for x in rUpTh)}"
            else:
                opts_str += f" --rUpTh {rUpTh} "

        if fLwTh is not None:
            if type(fLwTh) is tuple:
                opts_str += f" -fLwTh {' '.join(str(x) for x in fLwTh)} "
            else:
                opts_str += f" --fLwTh {fLwTh} "

        if fUpTh is not None:
            if type(fUpTh) is tuple:
                opts_str += f" -fUpTh {' '.join(str(x) for x in fUpTh)} "
            else:
                opts_str += f" --fUpTh {fUpTh} "

        if sx is not None:
            opts_str += f" -sx {sx}"

        if sy is not None:
            opts_str += f" -sy {sy}"

        if sz is not None:
            opts_str += f" -sz {sz}"

        if be is not None:
            opts_str += f" -be {be}"

        if le is not None:
            opts_str += f" -le {' '.join(str(x) for x in le)}"

        if l2 is not None:
            opts_str += f" -l2 {l2}"

        if jl is not None:
            opts_str += f" -jl {jl}"

        if noAppJL is True:
            opts_str += " -noAppJL"

        if nmi is True:
            opts_str += " --nmi"

        if rbn is not None:
            if type(rbn) is tuple:
                opts_str += f" -rbn {' '.join(str(x) for x in rbn)}"
            else:
                opts_str += f" --rbn {rbn}"

        if fbn is not None:
            if type(fbn) is tuple:
                opts_str += f" -fbn {' '.join(str(x) for x in fbn)}"
            else:
                opts_str += f" --fbn {fbn}"

        if lncc is not None:
            if type(lncc) is tuple:
                opts_str += f" -lncc {' '.join(str(x) for x in lncc)}"
            else:
                opts_str += f" --lncc {lncc}"

        if ssd is not None:
            if type(ssd) is (int, float):
                opts_str += f" -ssd {int(ssd)}"
            else:
                opts_str += " --ssd"

        if kld is not None:
            if type(kld) is (int, float):
                opts_str += f" -kld {int(kld)}"
            else:
                opts_str += " --kld"

        if amc is True:
            opts_str += " -amc"

        if maxit is not None:
            opts_str += f" -maxit {int(maxit)}"

        if ln is not None:
            opts_str += f" -ln {int(ln)}"

        if lp is not None:
            opts_str += f" -lp {int(lp)}"

        if nopy is True:
            opts_str += " -nopy"

        if noConj is True:
            opts_str += " -noConj"

        if pert is not None:
            opts_str += f" -pert {int(pert)}"

        if vel is True:
            opts_str += " -vel"

        if fmask is not None:
            write_nifti(path.join(tmp_folder, "fmask.nii"), fmask.astype(float))
            opts_str += " -fmask " + path.join(tmp_folder, "fmask.nii")

        if omp is not None:
            opts_str += f" -lp {int(omp)}"

        if mem is True:
            opts_str += " -mem"

        if gpu is True:
            opts_str += " -gpu"

        if smoothGrad is not None:
            opts_str += f" -smoothGrad {smoothGrad}"

        if pad is not None:
            opts_str += f" -pad {pad}"

        if not verbose:
            opts_str += " -voff"

        cmd_str += opts_str

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(res), read_nifti(cpp)
        else:
            return None


def resample(
    ref,
    flo,
    trans,
    res=None,
    blank=None,
    inter=None,
    pad=None,
    tensor=None,
    psf=False,
    verbose=False,
):
    """
    Usage:	reg_resample -ref <filename> -flo <filename> [OPTIONS].
    -ref <filename>
        Filename of the reference image (mandatory)
    -flo <filename>
        Filename of the floating image (mandatory)
    """

    # usage_string = "reg_resample -ref <filename> -flo <filename> [OPTIONS]"

    cmd_str = "reg_resample "

    with tmp.TemporaryDirectory() as tmp_folder:

        cmd_str = "reg_resample"

        write_nifti(path.join(tmp_folder, "ref.nii"), ref)
        write_nifti(path.join(tmp_folder, "flo.nii"), flo)

        cmd_str += " -ref " + path.join(tmp_folder, "ref.nii")
        cmd_str += " -flo " + path.join(tmp_folder, "flo.nii")

        if trans.shape == (4, 4):
            write_txt(path.join(tmp_folder, "trans.txt"), trans)
            cmd_str += " -trans " + path.join(tmp_folder, "trans.txt")
        else:
            write_nifti(path.join(tmp_folder, "trans.nii"), trans)
            cmd_str += " -trans " + path.join(tmp_folder, "trans.nii")

        opts_str = ""

        if res is None:
            res = path.join(tmp_folder, "res.nii")

        opts_str += " -res " + res

        if blank is not None:
            write_nifti(path.join(tmp_folder, "blank"), blank)
            opts_str += " -blank " + path.join(tmp_folder, "blank")

        if int(inter) in range(5):
            opts_str += f" -inter {int(inter)}"

        if pad is not None:
            opts_str += f" -pad {int(pad)}"

        if tensor is not None:
            opts_str += " -tensor"

        if psf is True:
            opts_str += " -psf"

        if not verbose:
            opts_str += " -voff"

        cmd_str += opts_str

        if verbose:
            print(cmd_str)

        if call_niftyreg(cmd_str, verbose):
            return read_nifti(res)
        else:
            return None


def jacobian(trans, ref, jac=None, jacM=None, jacL=None):
    raise NotImplementedError


def tools(
    input,
    out=None,
    float=None,
    down=None,
    smoS=None,
    smoG=None,
    smoL=None,
    add=None,
    sub=None,
    mul=None,
    div=None,
    rms=None,
    bin=None,
    thr=None,
    nan=None,
    iso=None,
    noscl=None,
    version=None,
    verbose=False,
):

    cmd_str = "reg_tools "
    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "in.nii"), input)
        cmd_str += " -in " + path.join(tmp_folder, "in.nii")

        if out is None:
            out = path.join(tmp_folder, "out.nii")

        cmd_str += " -out " + out

        if float is not None:
            cmd_str += " -float"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if down is not None:
            cmd_str += " -down"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if smoS is not None:
            smoS = ((smoS,) * 3) if np.isscalar(smoS) else smoS
            cmd_str += f' -smoS {" ".join(str(x) for x in smoS)}'
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if smoG is not None:
            smoG = ((smoG,) * 3) if np.isscalar(smoG) else smoG
            cmd_str += f' -smoG {" ".join(str(x) for x in smoG)}'
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if smoL is not None:
            smoL = ((smoL,) * 3) if np.isscalar(smoL) else smoL
            cmd_str += f' -smoL {" ".join(str(x) for x in smoL)}'
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if add is not None:
            if isinstance(add, np.ndarray):
                write_nifti(path.join(tmp_folder, "add.nii"), add)
                cmd_str += " -add " + path.join(tmp_folder, "add.nii")
            else:
                cmd_str += f" -add {add}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if sub is not None:
            if isinstance(sub, np.ndarray):
                write_nifti(path.join(tmp_folder, "sub.nii"), sub)
                cmd_str += " -sub " + path.join(tmp_folder, "sub.nii")
            else:
                cmd_str += f" -sub {sub}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if mul is not None:
            if isinstance(mul, np.ndarray):
                write_nifti(path.join(tmp_folder, "mul.nii"), mul)
                cmd_str += " -mul " + path.join(tmp_folder, "mul.nii")
            else:
                cmd_str += f" -mul {mul}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if div is not None:
            if isinstance(div, np.ndarray):
                write_nifti(path.join(tmp_folder, "div.nii"), div)
                cmd_str += " -div " + path.join(tmp_folder, "div.nii")
            else:
                cmd_str += f" -div {div}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if rms is not None:
            write_nifti(path.join(tmp_folder, "rms.nii"), rms)
            cmd_str += " -rms " + path.join(tmp_folder, "rms.nii")
            out = call_niftyreg(cmd_str, verbose=verbose, output_stdout=True)
            return None if not out else builtins.float(out)

        if bin is not None:
            cmd_str += " -bin"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if thr is not None:
            cmd_str += f" -thr {thr}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if nan is not None:
            if isinstance(nan, np.ndarray):
                write_nifti(path.join(tmp_folder, "nan.nii"), nan)
                cmd_str += " -nan " + path.join(tmp_folder, "nan.nii")
            else:
                cmd_str += f" -nan {nan}"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if iso is not None:
            cmd_str += " -iso"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if noscl is not None:
            cmd_str += " -noscl"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

        if version is not None:
            cmd_str += " --version"
            if call_niftyreg(cmd_str, verbose):
                return read_nifti(out)
            else:
                return None

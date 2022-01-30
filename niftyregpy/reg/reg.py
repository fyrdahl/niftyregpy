import numpy as np
import shlex as sh
import tempfile as tmp
import subprocess as sp
from ..utils import read_txt, write_txt, read_nifti, write_nifti, call_niftyreg

DEBUG = False
NAME = tmp.NamedTemporaryFile().name


def aladin(ref, flo, noSym=None, regOnly=None, affDirect=None, aff=None,
           inaff=None, rmask=None, fmask=None, res=None, maxit=None, ln=None,
           lp=None, smooR=None, smooF=None, refLowThr=None, refUpThr=None,
           floLowThr=None, floUpThr=None, nac=None, cog=None, interp=None,
           iso=None, pv=None, pi=None, speeeeed=None, omp=None, verbose=True):

    """
    Block Matching algorithm for global registration.
    Based on Ourselin et al., "Reconstructing a 3D structure from serial
    histological sections" Image and Vision Computing, 2001
    """
    usage_string = "reg_aladin -ref <filename> -flo <filename> [OPTIONS]"

    cmd_str = 'reg_aladin '

    write_nifti(f'{NAME}ref', ref)
    write_nifti(f'{NAME}flo', flo)

    cmd_str += f'-ref {NAME}ref.nii -flo {NAME}flo.nii '

    opts_str = ''

    if res is None:
        res = NAME + 'res.nii'
    if aff is None:
        aff = NAME + 'aff.txt'

    opts_str += f'-res {res} '
    opts_str += f'-aff {aff} '

    if noSym is True:
        opts_str += '-noSym '

    if regOnly is True:
        opts_str += '-regOnly '

    if affDirect is True:
        opts_str += '-affDirect '

    if inaff is not None:
        opts_str += f'-inaff {inaff} '

    if rmask is not None:
        opts_str += f'-rmask {rmask} '

    if fmask is not None:
        opts_str += f'-fmask {fmask} '

    if maxit is not None:
        opts_str += f'-maxit {maxit} '

    if ln is not None:
        opts_str += f'-ln {ln} '

    if lp is not None:
        opts_str += f'-lp {lp} '

    if smooR is not None:
        opts_str += f'-smooR {smooR} '

    if smooF is not None:
        opts_str += f'-smooF {smooF} '

    if refLowThr is not None:
        opts_str += f'-refLowThr {refLowThr} '

    if refUpThr is not None:
        opts_str += f'-refUpThr {refUpThr} '

    if floLowThr is not None:
        opts_str += f'-floLowThr {floLowThr} '

    if floUpThr is not None:
        opts_str += f'-floUpThr {floUpThr} '

    if nac is True:
        opts_str += '-nac '

    if cog is True:
        opts_str += '-cog '

    if interp is True:
        opts_str += '-interp '

    if iso is True:
        opts_str += '-iso '

    if pv is not None:
        opts_str += f'-pv {pv} '

    if pi is not None:
        opts_str += f'-pi {pi} '

    if speeeeed is True:
        opts_str += '-speeeeed '

    if omp is not None:
        opts_str += f'-omp {omp} '

    cmd_str += opts_str + '  '

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str, verbose):
        return dict(aff=read_txt(aff), res=read_nifti(res))
    else:
        return None


def f3d(ref=None, flo=None, aff=None, incpp=None, cpp=None, res=None,
        rmask=None, smooR=None, smooF=None, rLwTh=None, rUpTh=None, fLwTh=None,
        fUpTh=None, sx=None, sy=None, sz=None, be=None, le=None, l2=None,
        jl=None, noAppJL=None, nmi=None, rbn=None, fbn=None, lncc=None,
        ssd=None, kld=None, amc=None, maxit=None, ln=None, lp=None, nopy=None,
        noConj=None, pert=None, vel=None, fmask=None, omp=None, mem=None,
        gpu=None, smoothGrad=None, pad=None, verbose=True):

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

    usage_string = "reg_f3d -ref <filename> -flo <filename> [OPTIONS]"

    cmd_str = 'reg_f3d '

    write_nifti(f'{NAME}ref', ref)
    write_nifti(f'{NAME}flo', flo)

    cmd_str += f'-ref {NAME}ref.nii -flo {NAME}flo.nii '

    opts_str = ''

    if res is None:
        res = NAME + 'res.nii'
    if cpp is None:
        cpp = NAME + 'cpp.nii'

    opts_str += f'-res {res} '
    opts_str += f'-cpp {cpp} '

    if aff is not None:
        write_txt(f'{NAME}aff.txt', aff)
        opts_str += f'-aff {NAME}aff.txt '

    if incpp is not None:
        opts_str += f'-incpp {incpp} '

    if rmask is not None:
        write_nifti(f'{NAME}rmask.nii', rmask)
        opts_str += f'-rmask {NAME}rmask.nii '

    if smooR is not None:
        opts_str += f'-smooR {smooR} '

    if smooF is not None:
        opts_str += f'-smooF {smooF} '

    if rLwTh is not None:
        if type(rLwTh) is tuple:
            opts_str += f'-rLwTh {" ".join(str(x) for x in rLwTh)} '
        else:
            opts_str += f'--rLwTh {rLwTh} '

    if rUpTh is not None:
        if type(rUpTh) is tuple:
            opts_str += f'-rUpTh {" ".join(str(x) for x in rUpTh)} '
        else:
            opts_str += f'--rUpTh {rUpTh} '

    if fLwTh is not None:
        if type(fLwTh) is tuple:
            opts_str += f'-fLwTh {" ".join(str(x) for x in fLwTh)} '
        else:
            opts_str += f'--fLwTh {fLwTh} '

    if fUpTh is not None:
        if type(fUpTh) is tuple:
            opts_str += f'-fUpTh {" ".join(str(x) for x in fUpTh)} '
        else:
            opts_str += f'--fUpTh {fUpTh} '

    if sx is not None:
        opts_str += f'-sx {sx} '

    if sy is not None:
        opts_str += f'-sy {sy} '

    if sz is not None:
        opts_str += f'-sz {sz} '

    if be is not None:
        opts_str += f'-be {be} '

    if le is not None:
        opts_str += f'-le {" ".join(str(x) for x in le)} '

    if l2 is not None:
        opts_str += f'-l2 {l2} '

    if jl is not None:
        opts_str += f'-jl {jl} '

    if noAppJL is True:
        opts_str += '-noAppJL '

    if nmi is True:
        opts_str += '--nmi '

    if rbn is not None:
        if type(rbn) is tuple:
            opts_str += f'-rbn {" ".join(str(x) for x in rbn)} '
        else:
            opts_str += f'--rbn {rbn} '

    if fbn is not None:
        if type(fbn) is tuple:
            opts_str += f'-fbn {" ".join(str(x) for x in fbn)} '
        else:
            opts_str += f'--fbn {fbn} '

    if lncc is not None:
        if type(lncc) is tuple:
            opts_str += f'-lncc {" ".join(str(x) for x in lncc)} '
        else:
            opts_str += f'--lncc {lncc} '

    if ssd is not None:
        if type(ssd) is (int, float):
            opts_str += f'-ssd {int(ssd)} '
        else:
            opts_str += '--ssd '

    if kld is not None:
        if type(kld) is (int, float):
            opts_str += f'-kld {int(kld)} '
        else:
            opts_str += '--kld '

    if amc is True:
        opts_str += '-amc '

    if maxit is not None:
        opts_str += f'-maxit {int(maxit)} '

    if ln is not None:
        opts_str += f'-ln {int(ln)} '

    if lp is not None:
        opts_str += f'-lp {int(lp)} '

    if nopy is True:
        opts_str += '-nopy '

    if noConj is True:
        opts_str += '-noConj '

    if pert is not None:
        opts_str += f'-pert {int(pert)} '

    if vel is True:
        opts_str += '-vel '

    if fmask is not None:
        write_nifti(f'{NAME}fmask.nii', rmask)
        opts_str += f'-fmask {NAME}fmask.nii '

    if omp is not None:
        opts_str += f'-lp {int(omp)} '

    if mem is True:
        opts_str += '-mem '

    if gpu is True:
        opts_str += '-gpu '

    if smoothGrad is not None:
        opts_str += f'-smoothGrad {smoothGrad} '

    if pad is not None:
        opts_str += f'-pad {pad} '

    cmd_str += opts_str

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str, verbose):
        return dict(res=read_nifti(res), cpp=read_nifti(cpp))
    else:
        return None


def resample():
    raise NotImplementedError


def transform(ref, flo, trans, res=None, blank=None, inter=None, pad=None,
              tensor=None, psf=False, verbose=False):
    """
    Usage:	reg_resample -ref <filename> -flo <filename> [OPTIONS].
    -ref <filename>
        Filename of the reference image (mandatory)
    -flo <filename>
        Filename of the floating image (mandatory)
    """

    usage_string = "reg_resample -ref <filename> -flo <filename> [OPTIONS]"

    cmd_str = 'reg_resample '

    write_nifti(f'{NAME}ref', ref)
    write_nifti(f'{NAME}flo', flo)
    write_nifti(f'{NAME}trans', trans)

    cmd_str += f'-ref {NAME}ref.nii -flo {NAME}flo.nii -trans {NAME}trans.nii '

    opts_str = ''

    if res is None:
        res = NAME + 'res.nii'

    if blank is not None:
        write_nifti(f'{NAME}blank.nii', blank)
        opts_str += f'-blank {NAME}blank.nii '

    if inter is not None:
        opts_str += f'-inter {int(inter)} '

    if pad is not None:
        opts_str += f'-pad {int(pad)} '

    if tensor is not None:
        opts_str += f'-fbn {" ".join(str(x) for x in tensor)}'

    if pad is True:
        opts_str += '-psf '

    cmd_str += opts_str + '  '

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str, verbose):
        return dict(res=read_nifti(res))
    else:
        return None


def jacobian(trans, ref, jac=None, jacM=None, jacL=None):
    raise NotImplementedError


def average(output, avg=None, avg_lts=None, avg_tran=None, demean1=None,
            demean2=None, demean3=None, version=None):

    usage_string = 'reg_average <outputFileName> [OPTIONS]'

    cmd_str = 'reg_average '

    if output is None:
        cmd_str += f'{NAME}output.nii '
    else:
        cmd_str += f'{output} '

    if avg is not None:
        if not any(isinstance(x, np.ndarray) for x in avg):
            cmd_str += f'-avg {" ".join(str(x) for x in avg)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if avg_lts is not None:
        if not any(isinstance(x, np.ndarray) for x in avg_lts):
            cmd_str += f'-avg_lts {" ".join(str(x) for x in avg_lts)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if avg_tran is not None:
        if not any(isinstance(x, np.ndarray) for x in avg_tran):
            cmd_str += f'-avg_tran {" ".join(str(x) for x in avg_tran)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if demean1 is not None:
        if not any(isinstance(x, np.ndarray) for x in demean1):
            cmd_str += f'-demean1 {" ".join(str(x) for x in demean1)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if demean2 is not None:
        if not any(isinstance(x, np.ndarray) for x in demean2):
            cmd_str += f'-demean2 {" ".join(str(x) for x in demean2)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if demean3 is not None:
        if not any(isinstance(x, np.ndarray) for x in demean3):
            cmd_str += f'-demean3 {" ".join(str(x) for x in demean3)} '
            pipe = sp.Popen(sh.split(cmd_str), stdout=sp.PIPE, stderr=sp.PIPE)
            retcode, err = pipe.communicate()
        else:
            raise NotImplementedError

    if DEBUG:
        print(cmd_str)


def tools(input, output=None, float=None, down=None, smoS=None, smoG=None,
          smoL=None, add=None, sub=None, mul=None, div=None, rms=None,
          bin=None, thr=None, nan=None, iso=None, noscl=None, version=None,
          verbose=False):

    cmd_str = 'reg_tools '

    write_nifti(f'{NAME}input.nii', input)
    cmd_str += f'-in {NAME}input.nii -out {NAME}output.nii '

    if float is not None:
        cmd_str += '-float '

    if down is not None:
        cmd_str += '-down '

    if smoS is not None:
        smoS = ((smoS, ) * 3) if np.isscalar(smoS) else smoS
        cmd_str += f'-smoS {" ".join(str(x) for x in smoS)}'

    if smoG is not None:
        smoG = ((smoG, ) * 3) if np.isscalar(smoG) else smoG
        cmd_str += f'-smoG {" ".join(str(x) for x in smoG)}'

    if smoL is not None:
        smoL = ((smoL, ) * 3) if np.isscalar(smoL) else smoL
        cmd_str += f'-smoL {" ".join(str(x) for x in smoL)}'

    if add is not None:
        if isinstance(add, np.ndarray):
            write_nifti(add, f'{NAME}add')
            cmd_str += f'-add {NAME}add.nii '
        else:
            cmd_str += f'-add {add}'

    if sub is not None:
        if isinstance(sub, np.ndarray):
            write_nifti(sub, f'{NAME}sub')
            cmd_str += f'-sub {NAME}sub.nii '
        else:
            cmd_str += f'-sub {sub}'

    if mul is not None:
        if isinstance(mul, np.ndarray):
            write_nifti(mul, f'{NAME}mul')
            cmd_str += f'-mul {NAME}mul.nii '
        else:
            cmd_str += f'-mul {mul}'

    if div is not None:
        if isinstance(div, np.ndarray):
            write_nifti(div, f'{NAME}div')
            cmd_str += f'-div {NAME}div.nii '
        else:
            cmd_str += f'-div {div}'

    if bin is not None:
        cmd_str += '-bin '

    if thr is not None:
        cmd_str += f'-thr {thr} '

    if nan is not None:
        if isinstance(nan, np.ndarray):
            write_nifti(nan, f'{NAME}nan')
            cmd_str += f'-nan {NAME}nan.nii '
        else:
            cmd_str += f'-nan {nan}'

    if iso is not None:
        cmd_str += '-iso '

    if noscl is not None:
        cmd_str += '-noscl '

    if version is not None:
        cmd_str += '--version '

    if DEBUG:
        print(cmd_str)

    if call_niftyreg(cmd_str):
        return read_nifti(f'{NAME}output.nii')
    else:
        return None

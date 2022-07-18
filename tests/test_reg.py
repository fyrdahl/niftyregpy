import numpy as np
from niftyregpy import reg
from skimage import transform

import test_common as common


class TestReg:
    def setup_method(self, method):
        self.matrix_size = 256
        self.tol = 0.2
        self.non_linearity = 10
        self.verbose = True
        self.object_size = 100
        common.seed_random_generators()

    def test_resample_affine(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = ref = common.create_square(self.matrix_size, size=self.object_size)
        affine = common.random_affine(rigid=True)
        output = reg.resample(ref, flo, trans=affine, inter=0, verbose=self.verbose)
        assert output is not None

    def test_aladin(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.rotate_array(ref, angle=45)
        flo = common.shear_array(flo, angle=10)
        output = reg.aladin(ref, flo, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_aladin_rigonly(self):
        ref = common.create_circle(self.matrix_size, r=self.object_size // 2)
        flo = common.create_circle(
            self.matrix_size, r=self.object_size // 2, c=self.matrix_size // 2 + 16
        )
        output = reg.aladin(ref, flo, rigOnly=True, maxit=10, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_aladin_inaff(self):
        ref = common.create_circle(self.matrix_size, r=self.object_size // 2)
        flo = common.create_circle(
            self.matrix_size, r=self.object_size // 2, c=self.matrix_size // 2 + 16
        )
        _, aff = reg.aladin(ref, flo, verbose=True)
        output = reg.aladin(ref, flo, inaff=aff, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_aladin_rmask(self):
        ref = common.create_circle(self.matrix_size, r=self.object_size // 2)
        flo = common.create_circle(
            self.matrix_size, r=self.object_size // 2, c=self.matrix_size // 2 + 16
        )
        rmask = common.create_square(
            self.matrix_size, size=self.object_size * 1.1, dtype=bool
        )
        output = reg.aladin(ref, flo, rmask=rmask, rigOnly=True, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_aladin_fmask(self):
        ref = common.create_circle(self.matrix_size, r=self.object_size // 2)
        flo = common.create_circle(
            self.matrix_size, r=self.object_size // 2, c=self.matrix_size // 2 + 16
        )
        fmask = common.create_square(
            self.matrix_size, size=self.object_size * 1.1, dtype=bool
        )
        output = reg.aladin(ref, flo, fmask=fmask, rigOnly=True, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_aladin_user_opts(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.rotate_array(ref, angle=45)
        output = reg.aladin(ref, flo, user_opts="-voff", verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_nmi(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        output = reg.f3d(ref, flo, nmi=True, rbn=2, fbn=2, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_lncc(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        output = reg.f3d(ref, flo, lncc=3, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_ssd(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        output = reg.f3d(ref, flo, ssd=True, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_kld(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        output = reg.f3d(ref, flo, kld=True, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_rmask(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        rmask = common.create_circle(
            self.matrix_size, r=self.object_size * 1.2, dtype=bool
        )
        output = reg.f3d(ref, flo, rmask=rmask, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_fmask(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        fmask = common.create_circle(
            self.matrix_size, r=self.object_size * 1.2, dtype=bool
        )
        output = reg.f3d(ref, flo, fmask=fmask, verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_f3d_user_opts(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        flo = common.apply_swirl(
            ref, self.matrix_size // 2, self.non_linearity, self.object_size
        )
        output = reg.f3d(ref, flo, user_opts="-voff", verbose=self.verbose)
        assert 1 - common.dice(ref, output[0]) < self.tol

    def test_reg_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, float=True, verbose=self.verbose)
        assert output is not None

    def test_reg_down(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, down=True, verbose=self.verbose)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_reg_smoS_scalar_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoS=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoS_tuple_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoS=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

    def test_reg_smoG_scalar_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoG=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoG_tuple_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoG=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

    def test_reg_smoL_scalar_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoL=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoL_tuple_input(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, smoL=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

    def test_add_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = reg.tools(input, add=x, verbose=self.verbose)
        assert (output == input + x).all()

    def test_add_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, add=x, verbose=self.verbose)
        assert (output == input + x).all()

    def test_sub_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = reg.tools(input, sub=x, verbose=self.verbose)
        assert (output == input - x).all()

    def test_sub_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, sub=x, verbose=self.verbose)
        assert (output == input - x).all()

    def test_mul_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = reg.tools(input, mul=x, verbose=self.verbose)
        assert (output == input * x).all()

    def test_mul_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, mul=x, verbose=self.verbose)
        assert (output == input * x).all()

    def test_div_float(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_float() + 1
        output = reg.tools(input, div=x, verbose=self.verbose)
        assert (output == input / x).all()

    def test_div_matrix(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        output = reg.tools(input, div=x, verbose=self.verbose)
        assert (output == input / x).all()

    def test_rms(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, rms=input, verbose=self.verbose)
        assert output == 0

    def test_bin(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = reg.tools(input, bin=True, verbose=self.verbose)
        assert (output == (input > 0).astype(np.float32)).all()

    def test_thr(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        thr = common.random_float()
        output = reg.tools(input, thr=thr, verbose=self.verbose)
        assert (output == (input >= thr).astype(np.float32)).all()

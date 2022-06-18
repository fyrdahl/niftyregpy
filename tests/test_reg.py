import numpy as np
from niftyregpy import reg
from skimage import transform

import test_common as common


class TestReg:
    def setup_method(self, method):
        self.length = 256
        self.tol = 0.2
        self.non_linearity = 10
        self.verbose = True
        self.size = common.random_float(self.length / 8, self.length * 7 / 8)
        common.seed_random_generators()

    def test_resample_affine(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        flo = common.create_rect(self.length, w=self.size, h=self.size)
        affine = common.random_rigid()
        output = reg.resample(ref, flo, trans=affine, inter=0, verbose=self.verbose)
        assert output is not None

    def test_aladin_default(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        tform_trans = transform.AffineTransform(translation=-self.length // 2)
        tform_rot = transform.AffineTransform(rotation=45, shear=10)
        flo = transform.warp(ref, tform_trans + tform_rot + tform_trans.inverse)
        output = reg.aladin(ref, flo, pad=0, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_rigonly(self):
        ref = common.create_circle(self.length, r=self.size // 2)
        flo = common.create_circle(
            self.length, r=self.size // 2, c=self.length // 2 + 16
        )
        output = reg.aladin(ref, flo, pad=0, rigOnly=True, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_inaff(self):
        ref = common.create_circle(self.length, r=self.size // 2)
        flo = common.create_circle(
            self.length, r=self.size // 2, c=self.length // 2 + 16
        )
        _, aff = reg.aladin(ref, flo, verbose=True)
        output = reg.aladin(ref, flo, pad=0, inaff=aff, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_rmask(self):
        ref = common.create_circle(self.length, r=self.size // 2)
        flo = common.create_circle(
            self.length, r=self.size // 2, c=self.length // 2 + 16
        )
        rmask = common.create_rect(self.length, w=self.size, h=self.size)
        output = reg.aladin(
            ref, flo, pad=0, rmask=rmask, rigOnly=True, verbose=self.verbose
        )
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_fmask(self):
        ref = common.create_circle(self.length, r=self.size // 2)
        flo = common.create_circle(
            self.length, r=self.size // 2, c=self.length // 2 + 16
        )
        fmask = common.create_rect(
            self.length, c=self.length // 2 + 16, w=self.size, h=self.size
        )
        output = reg.aladin(
            ref, flo, pad=0, fmask=fmask, rigOnly=True, verbose=self.verbose
        )
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_f3d_nmi(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        flo = common.apply_swirl(ref, self.length // 2, self.non_linearity, self.size)
        output = reg.f3d(ref, flo, pad=0, nmi=True, rbn=2, fbn=2, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < 1

    def test_f3d_lncc(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        flo = common.apply_swirl(ref, self.length // 2, self.non_linearity, self.size)
        output = reg.f3d(ref, flo, pad=0, lncc=3, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_f3d_ssd(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        flo = common.apply_swirl(ref, self.length // 2, self.non_linearity, self.size)
        output = reg.f3d(ref, flo, pad=0, ssd=True, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_f3d_kld(self):
        ref = common.create_rect(self.length, w=self.size, h=self.size)
        flo = common.apply_swirl(ref, self.length // 2, self.non_linearity, self.size)
        output = reg.f3d(ref, flo, pad=0, kld=True, verbose=self.verbose)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_reg_float(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, float=True, verbose=self.verbose)
        assert output is not None

    def test_reg_down(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, down=True, verbose=self.verbose)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_reg_smoS_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoS=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoS_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoS=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

    def test_reg_smoG_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoG=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoG_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoG=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

    def test_reg_smoL_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoL=common.random_float(), verbose=self.verbose)
        assert output is not None

    def test_reg_smoL_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoL=common.random_tuple(3), verbose=self.verbose)
        assert output is not None

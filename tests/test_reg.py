import numpy as np
from niftyregpy import reg, utils
from skimage import transform

import test_common as common


class TestReg:
    def setup_method(self, method):
        self.length = 256
        self.tol = 0.1
        self.non_linearity = 10
        common.seed_random_generators()

    def test_resample_affine(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        flo = common.create_rect(self.length, w=size, h=size)
        affine = common.random_rigid()
        output = reg.resample(ref, flo, trans=affine, inter=0, verbose=True)
        assert output is not None

    def test_aladin_default(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        tform1 = transform.AffineTransform(translation=-self.length // 2)
        tform2 = transform.AffineTransform(rotation=45, shear=10)
        tform3 = transform.AffineTransform(translation=self.length // 4)
        flo = transform.warp(ref, tform1 + tform2 + tform3)
        output = reg.aladin(ref, flo, pad=0, verbose=True)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_rigonly(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_circle(self.length, r=size // 2)
        flo = common.create_circle(self.length, r=size // 2, c=self.length // 2 + 16)
        output = reg.aladin(ref, flo, pad=0, rigOnly=True, verbose=True)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_inaff(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_circle(self.length, r=size // 2)
        flo = common.create_circle(self.length, r=size // 2, c=self.length // 2 + 16)
        _, aff = reg.aladin(ref, flo, verbose=True)
        output = reg.aladin(ref, flo, pad=0, inaff=aff, verbose=True)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_rmask(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_circle(self.length, r=size // 2)
        flo = common.create_circle(self.length, r=size // 2, c=self.length // 2 + 16)
        rmask = common.create_rect(self.length, w=size, h=size)
        output = reg.aladin(ref, flo, pad=0, rmask=rmask, rigOnly=True, verbose=True)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_aladin_fmask(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_circle(self.length, r=size // 2)
        flo = common.create_circle(self.length, r=size // 2, c=self.length // 2 + 16)
        fmask = common.create_rect(self.length, c=self.length // 2 + 16, w=size, h=size)
        output = reg.aladin(ref, flo, pad=0, fmask=fmask, rigOnly=True, verbose=True)
        assert 1 - common.jaccard(ref, output[0]) < self.tol

    def test_f3d_nmi(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        np.save("/Users/alefyr/ref.npy", ref)
        flo = transform.swirl(
            ref,
            center=(self.length // 2, self.length // 2),
            strength=10,
            radius=size,
        )
        np.save("/Users/alefyr/flo.npy", flo)
        output = reg.f3d(ref, flo, pad=0, nmi=True, rbn=2, fbn=2, verbose=True)
        np.save("/Users/alefyr/out_nmi.npy", output[0])
        assert 1 - common.jaccard(ref, output[0]) < 1

    def test_f3d_lncc(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        flo = transform.swirl(
            ref,
            center=(self.length // 2, self.length // 2),
            strength=self.non_linearity,
            radius=size,
        )
        output = reg.f3d(ref, flo, pad=0, lncc=3, verbose=True)
        np.save("/Users/alefyr/out_lncc.npy", output[0])
        assert 1 - common.jaccard(ref, output[0]) < 1

    def test_f3d_ssd(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        flo = transform.swirl(
            ref,
            center=(self.length // 2, self.length // 2),
            strength=self.non_linearity,
            radius=size,
        )
        output = reg.f3d(ref, flo, pad=0, ssd=True, verbose=True)
        np.save("/Users/alefyr/out_ssd.npy", output[0])
        assert 1 - common.jaccard(ref, output[0]) < 1

    def test_f3d_kld(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        flo = transform.swirl(
            ref,
            center=(self.length // 2, self.length // 2),
            strength=self.non_linearity,
            radius=size,
        )
        output = reg.f3d(ref, flo, pad=0, kld=True, verbose=True)
        np.save("/Users/alefyr/out_kld.npy", output[0])
        assert 1 - common.jaccard(ref, output[0]) < 1

    def test_reg_float(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, float=True)
        assert output is not None

    def test_reg_down(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, down=True)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_reg_smoS_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoS=common.random_float())
        assert output is not None

    def test_reg_smoS_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoS=common.random_tuple(3))
        assert output is not None

    def test_reg_smoG_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoG=common.random_float())
        assert output is not None

    def test_reg_smoG_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoG=common.random_tuple(3))
        assert output is not None

    def test_reg_smoL_scalar_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoL=common.random_float())
        assert output is not None

    def test_reg_smoL_tuple_input(self):
        input = common.random_array((self.length, self.length), np.float32)
        output = reg.tools(input, smoL=common.random_tuple(3))
        assert output is not None

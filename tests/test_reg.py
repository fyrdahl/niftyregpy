import numpy as np
import test_common as common

from niftyregpy import reg

class TestReg():

    def setup_method(self, method):
        self.length = 256
        common.seed_random_generators()

    def test_aladin_default(self):
        size = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_rect(self.length, w=size, h=size)
        flo = common.create_rect(self.length, w=size, h=size)
        noisy_ref = common.add_noise(ref)
        noisy_flo = common.add_noise(flo)
        output = reg.aladin(noisy_ref, noisy_flo, verbose=False)
        assert output is not None

    def test_f3d_default(self):
        rad_ref = common.random_float(self.length / 8, self.length * 7 / 8)
        ref = common.create_circle(self.length, r=rad_ref)
        noisy_ref = common.add_noise(ref)
        rad_flo = common.random_float(self.length / 8, self.length * 7 / 8)
        flo = common.create_circle(self.length, r=rad_flo)
        noisy_flo = common.add_noise(flo)
        output = reg.f3d(noisy_ref, noisy_flo, pad=0, verbose=False)
        assert output is not None

    def test_reg_float(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, float=True)
        assert output is not None

    def test_reg_down(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, down=True)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_reg_smoS_scalar_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoS=common.random_float())
        assert output is not None

    def test_reg_smoS_tuple_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoS=common.create_random_tuple(3))
        assert output is not None

    def test_reg_smoG_scalar_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoG=common.random_float())
        assert output is not None

    def test_reg_smoG_tuple_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoG=common.create_random_tuple(3))
        assert output is not None

    def test_reg_smoL_scalar_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoL=common.random_float())
        assert output is not None

    def test_reg_smoL_tuple_input(self):
        input = common.create_random_array((256, 256), np.float32)
        output = reg.tools(input, smoL=common.create_random_tuple(3))
        assert output is not None

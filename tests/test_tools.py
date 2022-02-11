import numpy as np
import test_common as common

from niftyregpy import tools


class TestTools():

    def setup_method(self, method):
        self.length = 256
        common.seed_random_generators()

    def test_float(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.float(input)
        assert output is not None

    def test_down(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.down(input)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_smoS(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.smoS(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
        )
        assert output is not None

    def test_smoG(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.smoG(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
        )
        assert output is not None

    def test_smoL(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.smoL(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
        )
        assert output is not None

    def test_add_float(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.random_float()
        output = tools.add(input, x)
        assert (output == input + x).all()

    def test_add_matrix(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.create_random_array((self.length, self.length), np.float32)
        output = tools.add(input, x)
        assert (output == input + x).all()

    def test_sub_float(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.random_float()
        output = tools.sub(input, x)
        assert (output == input - x).all()

    def test_sub_matrix(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.create_random_array((self.length, self.length), np.float32)
        output = tools.sub(input, x)
        assert (output == input - x).all()

    def test_mul_float(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.random_float()
        output = tools.mul(input, x)
        assert (output == input * x).all()

    def test_mul_matrix(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        x = common.create_random_array((self.length, self.length), np.float32)
        output = tools.mul(input, x)
        assert (output == input * x).all()

    def test_div_float(self):
        input = common.create_random_array((self.length, self.length), np.float32) + 1
        x = common.random_float() + 1
        output = tools.div(input, x)
        assert (output == input / x).all()

    def test_div_matrix(self):
        input = common.create_random_array((self.length, self.length), np.float32) + 1
        x = common.create_random_array((self.length, self.length), np.float32) + 1
        output = tools.div(input, x)
        assert (output == input / x).all()

    def test_bin(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        output = tools.bin(input)
        assert (output == (input > 0).astype(np.float32)).all()

    def test_thr(self):
        input = common.create_random_array((self.length, self.length), np.float32)
        thr = common.random_float()
        output = tools.thr(input, thr)
        assert (output == (input >= thr).astype(np.float32)).all()

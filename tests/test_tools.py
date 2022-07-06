import numpy as np
from niftyregpy import tools

import test_common as common


class TestTools:
    def setup_method(self, method):
        self.matrix_size = 256
        self.tol = 1e-6
        common.seed_random_generators()

    def test_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.float(input, verbose=True)
        assert output is not None

    def test_down(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.down(input, verbose=True)
        assert output.shape == tuple(x // 2 for x in input.shape)

    def test_smoS(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.smoS(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
            verbose=True,
        )
        assert output is not None

    def test_smoG(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.smoG(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
            verbose=True,
        )
        assert output is not None

    def test_smoL(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.smoL(
            input,
            sx=common.random_float(),
            sy=common.random_float(),
            sz=common.random_float(),
            verbose=True,
        )
        assert output is not None

    def test_add_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = tools.add(input, x, verbose=True)
        assert (output == input + x).all()

    def test_add_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.add(input, x, verbose=True)
        assert (output == input + x).all()

    def test_sub_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = tools.sub(input, x, verbose=True)
        assert (output == input - x).all()

    def test_sub_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.sub(input, x, verbose=True)
        assert (output == input - x).all()

    def test_mul_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output = tools.mul(input, x, verbose=True)
        assert (output == input * x).all()

    def test_mul_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.mul(input, x, verbose=True)
        assert (output == input * x).all()

    def test_div_float(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_float() + 1
        output = tools.div(input, x, verbose=True)
        assert (output == input / x).all()

    def test_div_matrix(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        output = tools.div(input, x, verbose=True)
        assert (output == input / x).all()

    def test_rms(self):
        input1 = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        input2 = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.rms(input1, input2, verbose=True)
        assert output - np.mean(np.sqrt((input1 - input2) ** 2)) < self.tol

    def test_bin(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.bin(input)
        assert (output == (input > 0).astype(np.float32)).all()

    def test_thr(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        thr = common.random_float()
        output = tools.thr(input, thr, verbose=True)
        assert (output == (input >= thr).astype(np.float32)).all()

    def test_nan(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        mask = common.create_circle(self.matrix_size, r=self.matrix_size // 4)
        output = tools.nan(input, mask, verbose=True)
        assert output is not None

    def test_noslc(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.noscl(input, verbose=True)
        assert output is not None

    def test_iso(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output = tools.iso(input, verbose=True)
        assert output is not None

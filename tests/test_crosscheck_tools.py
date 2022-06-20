import numpy as np
from niftyregpy import reg, tools

import test_common as common


class TestCrosscheckTools:
    def setup_method(self, method):
        self.matrix_size = 256
        self.tol = 1e-6
        self.verbose = True
        common.seed_random_generators()

    def test_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.float(input, verbose=self.verbose)
        output2 = reg.tools(input, float=True, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_down(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.down(input, verbose=True)
        output2 = reg.tools(input, down=True, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_smoS(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        smoS_value = common.random_float()
        output1 = tools.smoS(
            input,
            sx=smoS_value,
            sy=smoS_value,
            sz=smoS_value,
            verbose=True,
        )
        output2 = reg.tools(input, smoS=smoS_value, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_smoG(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        smoG_value = common.random_float()
        output1 = tools.smoG(
            input,
            sx=smoG_value,
            sy=smoG_value,
            sz=smoG_value,
            verbose=True,
        )
        output2 = reg.tools(input, smoG=smoG_value, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_smoL(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        smoL_value = common.random_float()
        output1 = tools.smoL(
            input,
            sx=smoL_value,
            sy=smoL_value,
            sz=smoL_value,
            verbose=True,
        )
        output2 = reg.tools(input, smoL=smoL_value, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_add_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output1 = tools.add(input, x, verbose=True)
        output2 = reg.tools(input, add=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_add_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.add(input, x, verbose=True)
        output2 = reg.tools(input, add=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_sub_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output1 = tools.sub(input, x, verbose=True)
        output2 = reg.tools(input, sub=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_sub_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.sub(input, x, verbose=True)
        output2 = reg.tools(input, sub=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_mul_float(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_float()
        output1 = tools.mul(input, x, verbose=True)
        output2 = reg.tools(input, mul=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_mul_matrix(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.mul(input, x, verbose=True)
        output2 = reg.tools(input, mul=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_div_float(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_float() + 1
        output1 = tools.div(input, x, verbose=True)
        output2 = reg.tools(input, div=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_div_matrix(self):
        input = (
            common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        )
        x = common.random_array((self.matrix_size, self.matrix_size), np.float32) + 1
        output1 = tools.div(input, x, verbose=True)
        output2 = reg.tools(input, div=x, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_bin(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        output1 = tools.bin(input)
        output2 = reg.tools(input, bin=True, verbose=self.verbose)
        assert (output1 == output2).all()

    def test_thr(self):
        input = common.random_array((self.matrix_size, self.matrix_size), np.float32)
        thr = common.random_float()
        output1 = tools.thr(input, thr, verbose=True)
        output2 = reg.tools(input, thr=thr, verbose=self.verbose)
        assert (output1 == output2).all()

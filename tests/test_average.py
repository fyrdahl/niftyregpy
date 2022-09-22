import numpy as np
from niftyregpy import average
from scipy.linalg import expm, logm

import test_common as common


class TestAverage:
    def setup_method(self, method):
        self.matrix_size = 256
        self.tol = 1e-6
        common.seed_random_generators()

    def test_avg_affine(self):
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        output = average.avg((aff1, aff2), verbose=True)
        input = [aff1, aff2]
        output2 = expm(sum(logm(aff) for aff in input) / len(input))
        assert np.allclose(output, output2)

    def test_avg_image(self):
        img1 = common.random_array((self.matrix_size, self.matrix_size))
        img2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.avg((img1, img2), verbose=True)
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_avg_lts(self):
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        output = average.avg_lts((aff1, aff2), verbose=True)
        assert output is not None and output.shape == (4, 4)

    def test_avg_tran_affine(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        tran1 = common.random_affine()
        tran2 = common.random_affine()
        output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2), verbose=True)
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_avg_tran_nii(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        tran1 = common.random_array((self.matrix_size, self.matrix_size))
        tran2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2), verbose=True)
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_demean1(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.avg_tran(ref, (aff1, aff2), (flo1, flo2), verbose=True)
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_demean2(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        tran1 = common.random_array((self.matrix_size, self.matrix_size))
        tran2 = common.random_array((self.matrix_size, self.matrix_size))
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.avg_tran(ref, (tran1, tran2), (flo1, flo2), verbose=True)
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_demean3(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        tran1 = common.random_array((self.matrix_size, self.matrix_size))
        tran2 = common.random_array((self.matrix_size, self.matrix_size))
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.demean3(
            ref, (aff1, aff2), (tran1, tran2), (flo1, flo2), verbose=True
        )
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

    def test_demean_noaff(self):
        ref = common.random_array((self.matrix_size, self.matrix_size))
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        tran1 = common.random_array((self.matrix_size, self.matrix_size))
        tran2 = common.random_array((self.matrix_size, self.matrix_size))
        flo1 = common.random_array((self.matrix_size, self.matrix_size))
        flo2 = common.random_array((self.matrix_size, self.matrix_size))
        output = average.demean_noaff(
            ref, (aff1, aff2), (tran1, tran2), (flo1, flo2), verbose=True
        )
        assert output is not None and output.shape == (
            self.matrix_size,
            self.matrix_size,
        )

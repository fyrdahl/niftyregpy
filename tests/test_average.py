from niftyregpy import average

import test_common as common


class TestAverage:
    def setup_method(self, method):
        self.length = 256
        self.tol = 1e-6
        common.seed_random_generators()

    def test_avg_affine(self):
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        output = average.avg((aff1, aff2), verbose=True)
        assert output is not None

    def test_avg_image(self):
        img1 = common.random_array((self.length, self.length))
        img2 = common.random_array((self.length, self.length))
        output = average.avg((img1, img2), verbose=True)
        assert output is not None

    def test_avg_lts(self):
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        output = average.avg_lts((aff1, aff2), verbose=True)
        assert output is not None

    def test_avg_tran_affine(self):
        ref = common.random_array((self.length, self.length))
        flo1 = common.random_array((self.length, self.length))
        flo2 = common.random_array((self.length, self.length))
        tran1 = common.random_affine()
        tran2 = common.random_affine()
        output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2), verbose=True)
        assert output is not None

    def test_avg_tran_nii(self):
        ref = common.random_array((self.length, self.length))
        flo1 = common.random_array((self.length, self.length))
        flo2 = common.random_array((self.length, self.length))
        tran1 = common.random_array((self.length, self.length))
        tran2 = common.random_array((self.length, self.length))
        output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2), verbose=True)
        assert output is not None

    def test_demean1(self):
        ref = common.random_array((self.length, self.length))
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        flo1 = common.random_array((self.length, self.length))
        flo2 = common.random_array((self.length, self.length))
        output = average.avg_tran(ref, (aff1, aff2), (flo1, flo2), verbose=True)
        assert output is not None

    def test_demean2(self):
        ref = common.random_array((self.length, self.length))
        tran1 = common.random_array((self.length, self.length))
        tran2 = common.random_array((self.length, self.length))
        flo1 = common.random_array((self.length, self.length))
        flo2 = common.random_array((self.length, self.length))
        output = average.avg_tran(ref, (tran1, tran2), (flo1, flo2), verbose=True)
        assert output is not None

    def test_demean3(self):
        ref = common.random_array((self.length, self.length))
        aff1 = common.random_affine()
        aff2 = common.random_affine()
        flo1 = common.random_array((self.length, self.length))
        flo2 = common.random_array((self.length, self.length))
        tran1 = common.random_array((self.length, self.length))
        tran2 = common.random_array((self.length, self.length))
        output = average.demean3(
            ref, (aff1, aff2), (flo1, flo2), (tran1, tran2), verbose=True
        )
        assert output is not None

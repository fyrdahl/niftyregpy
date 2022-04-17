from niftyregpy import average

import test_common as common


class TestAverage:
    def setup_method(self, method):
        self.length = 256
        common.seed_random_generators()

    def test_avg_affine(self):
        aff1 = common.create_random_affine()
        aff2 = common.create_random_affine()
        output = average.avg((aff1, aff2), verbose=True)
        assert output is not None

    def test_avg_image(self):
        img1 = common.create_random_array((self.length, self.length))
        img2 = common.create_random_array((self.length, self.length))
        output = average.avg((img1, img2), verbose=True)
        assert output is not None

    def test_avg_lts(self):
        aff1 = common.create_random_affine()
        aff2 = common.create_random_affine()
        output = average.avg_lts((aff1, aff2), verbose=True)
        assert output is not None

    def test_avg_tran(self):
        ref = common.create_random_array((self.length, self.length))
        flo1 = common.create_random_array((self.length, self.length))
        flo2 = common.create_random_array((self.length, self.length))
        tran1 = common.create_random_array((self.length, self.length))
        tran2 = common.create_random_array((self.length, self.length))
        output = average.avg_tran(ref, (flo1, flo2), (tran1, tran2), verbose=True)
        assert output is not None

    def test_demean1(self):
        ref = common.create_random_array((self.length, self.length))
        aff1 = common.create_random_affine()
        aff2 = common.create_random_affine()
        flo1 = common.create_random_array((self.length, self.length))
        flo2 = common.create_random_array((self.length, self.length))
        output = average.avg_tran(ref, (aff1, aff2), (flo1, flo2), verbose=True)
        assert output is not None

    def test_demean2(self):
        ref = common.create_random_array((self.length, self.length))
        tran1 = common.create_random_array((self.length, self.length))
        tran2 = common.create_random_array((self.length, self.length))
        flo1 = common.create_random_array((self.length, self.length))
        flo2 = common.create_random_array((self.length, self.length))
        output = average.avg_tran(ref, (tran1, tran2), (flo1, flo2), verbose=True)
        assert output is not None

    def test_demean3(self):
        ref = common.create_random_array((self.length, self.length))
        aff1 = common.create_random_affine()
        aff2 = common.create_random_affine()
        flo1 = common.create_random_array((self.length, self.length))
        flo2 = common.create_random_array((self.length, self.length))
        tran1 = common.create_random_array((self.length, self.length))
        tran2 = common.create_random_array((self.length, self.length))
        output = average.demean3(
            ref, (aff1, aff2), (flo1, flo2), (tran1, tran2), verbose=True
        )
        assert output is not None

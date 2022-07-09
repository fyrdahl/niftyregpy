from niftyregpy import apps

import test_common as common


class TestApps:
    def setup_method(self, method):
        self.matrix_size = 256
        self.object_size = 100
        common.seed_random_generators()

    def test_groupwise(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)
        noisy_ref = common.add_noise(ref)

        input_0 = common.create_square(self.matrix_size, size=self.object_size)
        noisy_input_0 = common.add_noise(input_0)
        input_1 = common.create_square(self.matrix_size, size=self.object_size)
        noisy_input_1 = common.add_noise(input_1)
        input_2 = common.create_square(self.matrix_size, size=self.object_size)
        noisy_input_2 = common.add_noise(input_2)
        output = apps.groupwise(
            noisy_ref, (noisy_input_0, noisy_input_1, noisy_input_2), verbose=False
        )
        assert output is not None

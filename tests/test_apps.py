from niftyregpy import apps

import test_common as common


class TestApps:
    def setup_method(self, method):
        self.matrix_size = 256
        self.object_size = 100
        self.tol = 0.2
        common.seed_random_generators()

    def test_groupwise(self):
        ref = common.create_square(self.matrix_size, size=self.object_size)

        input_0 = common.create_square(
            self.matrix_size, size=self.object_size, c=self.matrix_size // 2 - 10
        )
        input_0 = common.shear_array(input_0, angle=10)

        input_1 = common.create_square(
            self.matrix_size, size=self.object_size, c=self.matrix_size // 2 + 10
        )
        input_1 = common.rotate_array(input_1, angle=45)

        output = apps.groupwise(
            ref,
            (input_0, input_1),
            template_mask=common.create_circle(self.matrix_size),
            input_mask=(
                common.create_circle(self.matrix_size),
                common.create_circle(self.matrix_size),
            ),
            affine_args="-maxit 5",
            nrr_args="-maxit 300",
            verbose=False,
        )
        assert 1 - common.dice(ref, output[0]) < self.tol

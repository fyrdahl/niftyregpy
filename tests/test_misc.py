from niftyregpy import utils

import test_common as common


class TestMisc:
    def test_help_string(self):
        output = utils.get_help_string("reg_aladin")
        assert output is not None

    def test_help_string_failure(self):
        output = utils.get_help_string("reg_failure")
        assert output is None

    def test_is_function_available(self):
        output = utils.is_function_available("reg_tools", "iso")
        assert output is True

    def test_is_function_available_failure(self):
        output = utils.is_function_available("reg_aladin", "default")
        assert output is False

    def test_read_nifti_fail(self):
        retval = utils.read_nifti(f"{common.random_float()}_test.txt")
        assert retval is None

    def test_read_txt_fail(self):
        retval = utils.read_txt(f"{common.random_float()}_test.txt")
        assert retval is None
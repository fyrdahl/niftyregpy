import numpy as np
from niftyregpy import utils


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

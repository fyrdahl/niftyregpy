import tempfile as tmp

from ..average import avg, avg_lts
from ..reg import aladin, f3d
from ..utils import call_niftyreg, read_nifti, write_nifti


def groupwise(input, input_mask=None, template_mask=None, aff_it_num=5, nrr_it_num=10):

    raise NotImplementedError

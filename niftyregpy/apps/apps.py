import tempfile as tmp

from ..utils import write_nifti, read_nifti
from ..reg import aladin, f3d, resample, transform, jacobian
from ..average import avg, avg_lts, avg_tran, demean1, demean2, demean3


def groupwise(input, input_mask=None, template_mask=None,
              aff_it_num=5, nrr_it_num=10):

    res_folder = tmp.TemporaryDirectory().name

    assert input.ndim > 2 and input.shape[0] >= 2, \
        "Less than 2 images have been specified"
    assert input_mask is None or input.shape[0] == input_mask.shape[0], \
        "The number of images is different from the number of floating masks"

    raise NotImplementedError

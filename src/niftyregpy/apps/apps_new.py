import os
import tempfile as tmp
from os import path

from ..average import avg, avg_lts
from ..reg import aladin, f3d
from ..utils import call_niftyreg, read_nifti, write_nifti


def groupwise(
    template,
    input,
    input_mask=None,
    template_mask=None,
    aff_it=5,
    nrr_it=10,
    verbose=False,
):

    # with tmp.TemporaryDirectory() as tmp_folder:
    tmp_folder = tmp.mkdtemp()  # DEBUG
    # assert (input.ndim > 2 and input.shape[0] >= 2), "Less than 2 input images have been specified"

    # assert (input_mask is None or input.shape[0] == input_mask.shape[0]), "The number of images is different from the number of floating masks"

    write_nifti(path.join(tmp_folder, "template.nii"), template)

    for i, img in enumerate(input):
        write_nifti(path.join(tmp_folder, f"input_{i}.nii"), img)

    if input_mask is not None:
        for i, mask in enumerate(input_mask):
            write_nifti(path.join(tmp_folder, f"input_mask_{i}.nii"), mask)

    if template_mask is not None:
        write_nifti(path.join(tmp_folder, "template_mask.nii"), template_mask)

    average_image = path.join(tmp_folder, "template.nii")

    # Run the rigid or affine registration
    for cur_it in range(aff_it):

        # Check if the iteration has already been performed
        if not path.isfile(
            path.join(tmp_folder, f"aff_it{cur_it}", f"average_affine_it_{cur_it}.txt")
        ):

            # Create a folder to store the result
            if not path.exists(path.join(tmp_folder, f"aff_it{cur_it}")):
                print(
                    "Folder "
                    + path.join(tmp_folder, f"aff_it{cur_it}")
                    + "does not exist, creating!"
                )
                os.makedirs(path.join(tmp_folder, f"aff_it{cur_it}"))
            else:
                print(
                    "Folder "
                    + path.join(tmp_folder, f"aff_it{cur_it}")
                    + "does already exist!"
                )

            # All registration are performed serially
            for i in range(len(input)):

                # Check if the registration has already been performed
                if not path.isfile(
                    path.join(
                        tmp_folder, f"aff_it{cur_it}", f"aff_mat_img{i}_it{cur_it}.txt"
                    )
                ):

                    aladin_args = ""

                    # Registration is forced to be rigid for the first iteration
                    if cur_it == 0:
                        aladin_args += "-rigOnly "
                    else:
                        # Check if a previous affine can be use for initialization
                        prev_affine_file = path.join(
                            tmp_folder,
                            f"aff_it{cur_it-1}",
                            f"aff_mat_img{i}_it{cur_it-1}.txt",
                        )
                        if path.isfile(prev_affine_file):
                            aladin_args += f"-inaff {prev_affine_file} "

                    # Check if a mask has been specified for the reference image
                    if template_mask is not None:
                        aladin_args += (
                            f"-rmask {tmp_folder}{path.sep}template_mask.nii "
                        )

                    if input_mask is not None:
                        aladin_args += (
                            f"-fmask {tmp_folder}{path.sep}input_mask_{i}.nii "
                        )

                    cur_affine_file = path.join(
                        tmp_folder, f"aff_it{cur_it}", f"aff_mat_img{i}_it{cur_it}.txt"
                    )
                    aladin_args += f"-ref {average_image} "
                    aladin_args += f"-flo {tmp_folder}{path.sep}input_{i}.nii "
                    aladin_args += f"-aff {cur_affine_file} "

                    if cur_it == aff_it:
                        aladin_args += (
                            f"-res {tmp_folder}{path.sep}aff_mat_img{i}_it{cur_it}.nii "
                        )

                    aladin_cmd = "reg_aladin " + aladin_args

                    if True:
                        print(aladin_cmd)

                    assert call_niftyreg(aladin_cmd, verbose), "Aladin command failed!"

        if cur_it < aff_it - 1:
            # The transformations are demeaned to create the average image
            # Note that this is not done for the last iteration step

            average_args = path.join(
                tmp_folder, f"aff_it{cur_it}", f"average_affine_it_{cur_it}.nii"
            )
            average_args += " -demean1 "
            average_args += average_image + " "
            for i in range(len(input)):
                cur_affine_file = path.join(
                    tmp_folder, f"aff_it{cur_it}", f"aff_mat_img{i}_it{cur_it}.txt"
                )
                cur_img = path.join(tmp_folder, f"input_{i}.nii")
                average_args += " " + cur_affine_file + " " + cur_img

            average_cmd = "reg_average " + average_args
            print(average_cmd)
            assert call_niftyreg(average_cmd, True), "Average command failed!"

        else:
            # All the result images are directly averaged during the last step
            average_args = path.join(
                tmp_folder, f"aff_it{cur_it}", f"average_affine_it_{cur_it}.nii"
            )
            average_args += " -avg "
            average_args += average_image + " "
            for i in range(len(input)):
                cur_img = path.join(tmp_folder, f"input_{i}.nii")
                average_args += " " + cur_affine_file + " " + cur_img

    average_image = path.join(
        tmp_folder, f"aff_it{cur_it}", f"average_affine_it_{cur_it}.txt"
    )

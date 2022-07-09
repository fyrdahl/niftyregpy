import os
import re
import tempfile as tmp
from os import path

import numpy as np

from ..utils import call_niftyreg, read_nifti, write_nifti


def groupwise(
    template,
    input,
    input_mask=None,
    template_mask=None,
    aff_it=5,
    nrr_it=10,
    affine_args="",
    nrr_args="",
    verbose=False,
):

    assert len(input) >= 2, "Less than 2 input images have been specified"

    with tmp.TemporaryDirectory() as tmp_folder:
        # tmp_folder = tmp.mkdtemp()  # DEBUG

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
                path.join(
                    tmp_folder, f"aff_{cur_it+1}", f"average_affine_it_{cur_it+1}.txt"
                )
            ):

                # Create a folder to store the result
                if not path.exists(path.join(tmp_folder, f"aff_{cur_it+1}")):
                    print(
                        "Folder "
                        + path.join(tmp_folder, f"aff_{cur_it+1}")
                        + " does not exist, creating!"
                    )
                    os.makedirs(path.join(tmp_folder, f"aff_{cur_it+1}"))
                else:
                    print(
                        "Folder "
                        + path.join(tmp_folder, f"aff_{cur_it+1}")
                        + " already exist!"
                    )

                for i in range(len(input)):

                    # Check if the registration has already been performed
                    if not path.isfile(
                        path.join(
                            tmp_folder,
                            f"aff_{cur_it+1}",
                            f"aff_mat_input_{i}_it{cur_it+1}.txt",
                        )
                    ):

                        aladin_args = ""

                        if cur_it > 0:
                            # Check if a previous affine can be use for initialization
                            prev_affine_file = path.join(
                                tmp_folder,
                                f"aff_{cur_it}",
                                f"aff_mat_input_{i}_it{cur_it}.txt",
                            )
                            if path.isfile(prev_affine_file):
                                aladin_args += f" -inaff {prev_affine_file}"
                            else:
                                print("No previous affine found, probably an error!")
                        else:
                            # Registration is forced to be rigid for the first iteration
                            aladin_args += " -rigOnly"

                        # Check if a mask has been specified for the reference image
                        if template_mask is not None:
                            aladin_args += " -rmask" + path.join(
                                tmp_folder, "template_mask.nii"
                            )

                        if input_mask is not None:
                            aladin_args += " -fmask" + path.join(
                                tmp_folder, f"input_mask_{i}.nii"
                            )

                        cur_affine_file = path.join(
                            tmp_folder,
                            f"aff_{cur_it+1}",
                            f"aff_mat_input_{i}_it{cur_it+1}.txt",
                        )
                        aladin_args += f" -ref {average_image}"
                        aladin_args += " -flo " + path.join(
                            tmp_folder, f"input_{i}.nii"
                        )
                        aladin_args += f" -aff {cur_affine_file}"

                        if cur_it == aff_it - 1:
                            aladin_args += " -res " + path.join(
                                tmp_folder,
                                f"aff_{cur_it+1}",
                                f"aff_res_input_{i}_it{cur_it+1}.nii",
                            )

                        aladin_cmd = "reg_aladin" + aladin_args + " " + affine_args

                        assert call_niftyreg(
                            aladin_cmd, verbose
                        ), "Aladin command failed!"

            if cur_it < aff_it - 1:
                # The transformations are demeaned to create the average image
                # Note that this is not done for the last iteration step

                average_args = path.join(
                    tmp_folder, f"aff_{cur_it+1}", f"average_affine_it_{cur_it+1}.nii"
                )
                average_args += " -demean1 "
                average_args += average_image + " "
                for i, _ in enumerate(input):
                    cur_affine_file = path.join(
                        tmp_folder,
                        f"aff_{cur_it+1}",
                        f"aff_mat_input_{i}_it{cur_it+1}.txt",
                    )
                    cur_img = path.join(tmp_folder, f"input_{i}.nii")
                    average_args += cur_affine_file + " " + cur_img + " "

                average_cmd = "reg_average " + average_args
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"

            else:
                # All the result images are directly averaged during the last step
                average_args = path.join(
                    tmp_folder, f"aff_{cur_it+1}", f"average_affine_it_{cur_it+1}.nii"
                )
                average_args += " -avg"
                for i, _ in enumerate(input):
                    cur_img = path.join(
                        tmp_folder,
                        f"aff_{cur_it+1}",
                        f"aff_res_input_{i}_it{cur_it+1}.nii",
                    )
                    average_args += " " + cur_img

                average_cmd = "reg_average " + average_args
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"

            average_image = path.join(
                tmp_folder, f"aff_{cur_it+1}", f"average_affine_it_{cur_it+1}.nii"
            )

        for cur_it in range(nrr_it):

            # Check if the current average image has already been created
            if not path.isfile(
                path.join(
                    tmp_folder, f"nrr_{cur_it+1}", f"average_nonrigid_it_{cur_it+1}.nii"
                )
            ):
                # Create a folder to store the current results
                if not path.exists(path.join(tmp_folder, f"nrr_{cur_it+1}")):
                    print(
                        "Folder "
                        + path.join(tmp_folder, f"nrr_{cur_it+1}")
                        + " does not exist, creating!"
                    )
                    os.makedirs(path.join(tmp_folder, f"nrr_{cur_it+1}"))
                else:
                    print(
                        "Folder "
                        + path.join(tmp_folder, f"nrr_{cur_it+1}")
                        + " already exist!"
                    )

                for i, _ in enumerate(input):

                    # Check if the registration has already been performed
                    if not path.isfile(
                        path.join(
                            tmp_folder,
                            f"nrr_{cur_it+1}",
                            f"nrr_cpp_input_{i}_it_{cur_it+1}.nii",
                        )
                    ):

                        f3d_args = ""

                        f3d_args += f" -ref {average_image}"
                        f3d_args += " -flo " + path.join(tmp_folder, f"input_{i}.nii")
                        f3d_args += " -cpp " + path.join(
                            tmp_folder,
                            f"nrr_{cur_it+1}",
                            f"nrr_cpp_input_{i}_it{cur_it+1}.nii",
                        )

                        if cur_it == nrr_it - 1:
                            f3d_args += " -res " + path.join(
                                tmp_folder,
                                f"nrr_{cur_it+1}",
                                f"nrr_res_input_{i}_it{cur_it+1}.nii",
                            )

                        # Check if a mask has been specified for the reference image
                        if template_mask is not None:
                            f3d_args += " -rmask" + path.join(
                                tmp_folder, "template_mask.nii"
                            )

                        if input_mask is not None:
                            f3d_args += " -rmask" + path.join(
                                tmp_folder, f"input_mask_{i}.nii"
                            )

                        if aff_it > 0:
                            f3d_args += " -aff " + path.join(
                                tmp_folder,
                                f"aff_{aff_it}",
                                f"aff_mat_input_{i}_it{aff_it}.txt",
                            )

                        f3d_cmd = "reg_f3d " + f3d_args + " " + nrr_args
                        assert call_niftyreg(f3d_cmd, verbose), "f3d command failed!"

            # The transformation are demeaned to create the average image
            # Note that this is not done for the last iteration step
            if cur_it < nrr_it - 1:
                average_args = path.join(
                    tmp_folder,
                    f"nrr_{cur_it+1}",
                    f"average_nonrigid_it_{cur_it+1}.nii",
                )
                average_args += " -demean_noaff "
                average_args += average_image
                for i, _ in enumerate(input):
                    cur_affine_file = path.join(
                        tmp_folder,
                        f"aff_{aff_it}",
                        f"aff_mat_input_{i}_it{aff_it}.txt",
                    )
                    cur_f3d_file = path.join(
                        tmp_folder,
                        f"nrr_{cur_it+1}",
                        f"nrr_cpp_input_{i}_it{cur_it+1}.nii",
                    )
                    cur_img = path.join(tmp_folder, f"input_{i}.nii")
                    average_args += (
                        " " + cur_affine_file + " " + cur_f3d_file + " " + cur_img
                    )

                average_cmd = "reg_average " + average_args
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"
            else:
                # All the result images are directly averaged during the last step
                average_args = path.join(
                    tmp_folder, f"nrr_{cur_it+1}", f"average_nonrigid_it_{cur_it+1}.nii"
                )
                average_args += " -avg"
                for i, _ in enumerate(input):
                    cur_img = path.join(
                        tmp_folder,
                        f"nrr_{cur_it+1}",
                        f"nrr_res_input_{i}_it{cur_it+1}.nii",
                    )
                    average_args += " " + cur_img

                average_cmd = "reg_average " + average_args
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"

            average_image = path.join(
                tmp_folder,
                f"nrr_{cur_it+1}",
                f"average_nonrigid_it_{cur_it+1}.nii",
            )

        average = read_nifti(average_image)

        res = []
        for i, _ in enumerate(input):
            cur_img = path.join(
                tmp_folder, f"nrr_{cur_it+1}", f"nrr_res_input_{i}_it{cur_it+1}.nii"
            )
            res.append(read_nifti(cur_img))

    return average, res

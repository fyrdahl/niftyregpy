import shlex
import tempfile as tmp
from os import path

import numpy as np
from tqdm import tqdm

from ..utils import call_niftyreg, read_nifti, write_nifti


def groupwise(
    input_imgs,
    template=None,
    input_mask=None,
    template_mask=None,
    aff_it=5,
    nrr_it=10,
    affine_args=None,
    nrr_args=None,
    normalize=False,
    nan_out=False,
    verbose=False,
    show_pbar=True,
) -> tuple:
    """
    Groupwise image registration seeks to mitigate bias caused by a single
    template image frame. The groupwise image registration works in two parts,
    first ``aff_it`` number of affine registrations (reg_aladin) are performed,
    the first is a rigid registration. Second, ``nrr_it`` number of non-rigid
    registrations (reg_f3d) are performed. After each full iteration, the
    transforms are averaged, and used to initialize the next iteration.

    Arguments can be passed to both reg_aladin and reg_f3d using the
    ``affine_args`` and ``nrr_args`` arguments.

    If no template image is explicitly provided, the first image in ``input_imgs``
    will be used to initialize the atlas.

    Args:
        input_imgs (tuple): Tuple that contains the images to create the atlas.
        template (array): Template image to use to initialize the atlas (optional).
        input_mask (tuple): Masks for the input images (optional).
        template_mask (array): Mask for the template image (optional).
        aff_it (int): Number of affine iterations to perform (default = 5).
        nrr_it (int): Number of non-rigid iterations to perform (default = 10).
        affine_args (str): Arguments to use for the affine registration (optional).
        nrr_args (str): Arguments to use for the non-rigid registration (optional).
        normalize (bool): Normalize input images to the range [0, 1] (default = False).
        nan_out (bool): If True, output NaN values (default = False).
        verbose (bool): Verbose output (default = False).
        show_pbar (bool): Show progress bars (default = True).

    Returns:
        A tuple containing

        - average (array): Average image
        - reg (tuple): Registered input images as a tuple


    Given two numpy arrays ``input_0`` and ``input_1``, an example usage is:
        >>> avg, res = niftyregpy.apps.groupwise((input_0, input_1))

    """

    assert (
        isinstance(input_imgs, (tuple, list, set)) and len(input_imgs) >= 2
    ), "Less than 2 input images have been specified"

    # If only one input_mask is provided, duplicate it to number of input images
    if isinstance(input_mask, np.ndarray):
        input_mask = [input_mask for _ in input_imgs]

    assert input_mask is None or len(input_imgs) == len(
        input_mask
    ), "The number of input masks are > 1 but different from the number input images"

    assert template is None or not isinstance(
        template, (tuple, list)
    ), "More than 1 template is provided"

    assert template_mask is None or not isinstance(
        template_mask, (tuple, list)
    ), "More than 1 template mask is provided"

    if template is None:
        template = input_imgs[0]

    if normalize:
        max_val = [x.max() for x in input_imgs]
        min_val = [x.min() for x in input_imgs]
        input_imgs = [
            (x - z) / (y - z) for x, y, z in zip(input_imgs, max_val, min_val)
        ]
        template = (template - template.min()) / (template.max() - template.min())

    with tmp.TemporaryDirectory() as tmp_folder:

        write_nifti(path.join(tmp_folder, "template.nii"), template)

        for i, img in enumerate(input_imgs):
            write_nifti(path.join(tmp_folder, f"input_{i}.nii"), img)

        if input_mask is not None:
            for i, mask in enumerate(input_mask):
                write_nifti(path.join(tmp_folder, f"input_mask_{i}.nii"), mask)

        if template_mask is not None:
            write_nifti(path.join(tmp_folder, "template_mask.nii"), template_mask)

        average_image = path.join(tmp_folder, "template.nii")

        # Run the rigid or affine registration
        with tqdm(
            total=aff_it, desc="Affine registration", disable=not show_pbar
        ) as pbar:
            for cur_it in range(aff_it):

                for i, _ in enumerate(input_imgs):

                    aladin_args = ""

                    if cur_it > 0:
                        prev_affine_file = path.join(
                            tmp_folder,
                            f"aff_mat_input_{i}_it{cur_it}.txt",
                        )
                        aladin_args += f" -inaff {prev_affine_file}"
                    else:
                        aladin_args += " -rigOnly"

                    # Check if a mask has been specified for the reference image
                    if template_mask is not None:
                        aladin_args += " -rmask " + path.join(
                            tmp_folder, "template_mask.nii"
                        )

                    if input_mask is not None:
                        aladin_args += " -fmask " + path.join(
                            tmp_folder, f"input_mask_{i}.nii"
                        )

                    cur_affine_file = path.join(
                        tmp_folder,
                        f"aff_mat_input_{i}_it{cur_it+1}.txt",
                    )
                    aladin_args += f" -ref {average_image}"
                    aladin_args += " -flo " + path.join(tmp_folder, f"input_{i}.nii")
                    aladin_args += f" -aff {cur_affine_file}"

                    if cur_it == aff_it - 1:
                        aladin_args += " -res " + path.join(
                            tmp_folder,
                            f"aff_res_input_{i}_it{cur_it+1}.nii",
                        )

                    if affine_args is not None:
                        for x in shlex.split(affine_args):
                            aladin_args += f" {shlex.quote(x)}"

                    aladin_cmd = f"reg_aladin {aladin_args}"

                    assert call_niftyreg(aladin_cmd, verbose), "Aladin command failed!"

                if cur_it < aff_it - 1:
                    # The transformations are demeaned to create the average image
                    # Note that this is not done for the last iteration step

                    average_args = path.join(
                        tmp_folder, f"average_affine_it_{cur_it+1}.nii"
                    )
                    average_args += " -demean1 "
                    average_args += f"{average_image} "
                    for i, _ in enumerate(input_imgs):
                        cur_affine_file = path.join(
                            tmp_folder,
                            f"aff_mat_input_{i}_it{cur_it+1}.txt",
                        )
                        cur_img = path.join(tmp_folder, f"input_{i}.nii")
                        average_args += f"{cur_affine_file} {cur_img} "

                else:
                    # All the result images are directly averaged during the last step
                    average_args = path.join(
                        tmp_folder, f"average_affine_it_{cur_it+1}.nii"
                    )
                    average_args += " -avg"
                    for i, _ in enumerate(input_imgs):
                        cur_img = path.join(
                            tmp_folder,
                            f"aff_res_input_{i}_it{cur_it+1}.nii",
                        )
                        average_args += f" {cur_img}"

                average_cmd = f"reg_average {average_args}"
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"

                average_image = path.join(
                    tmp_folder, f"average_affine_it_{cur_it+1}.nii"
                )
                pbar.update()

        with tqdm(
            total=nrr_it, desc="Non-rigid registration", disable=not show_pbar
        ) as pbar:
            for cur_it in range(nrr_it):

                for i, _ in enumerate(input_imgs):

                    f3d_args = f" -ref {average_image}"
                    f3d_args += " -flo "
                    f3d_args += path.join(tmp_folder, f"input_{i}.nii")
                    f3d_args += " -cpp "
                    f3d_args += path.join(
                        tmp_folder,
                        f"nrr_cpp_input_{i}_it{cur_it+1}.nii",
                    )

                    if cur_it == nrr_it - 1:
                        f3d_args += " -res " + path.join(
                            tmp_folder,
                            f"nrr_res_input_{i}_it{cur_it+1}.nii",
                        )

                    # Check if a mask has been specified for the reference image
                    if template_mask is not None:
                        f3d_args += " -rmask "
                        f3d_args += path.join(tmp_folder, "template_mask.nii")

                    if input_mask is not None:
                        f3d_args += " -fmask "
                        f3d_args += path.join(tmp_folder, f"input_mask_{i}.nii")

                    if aff_it > 0:
                        f3d_args += " -aff "
                        f3d_args += path.join(
                            tmp_folder,
                            f"aff_mat_input_{i}_it{aff_it}.txt",
                        )

                    if nrr_args is not None:
                        for x in shlex.split(nrr_args):
                            f3d_args += f" {shlex.quote(x)}"

                    f3d_cmd = f"reg_f3d {f3d_args}"
                    assert call_niftyreg(f3d_cmd, verbose), "f3d command failed!"

                # The transformation are demeaned to create the average image
                # Note that this is not done for the last iteration step
                if cur_it < nrr_it - 1:
                    average_args = path.join(
                        tmp_folder,
                        f"average_nonrigid_it_{cur_it+1}.nii",
                    )
                    average_args += " -demean_noaff "
                    average_args += average_image
                    for i, _ in enumerate(input_imgs):
                        cur_affine_file = path.join(
                            tmp_folder,
                            f"aff_mat_input_{i}_it{aff_it}.txt",
                        )
                        cur_f3d_file = path.join(
                            tmp_folder,
                            f"nrr_cpp_input_{i}_it{cur_it+1}.nii",
                        )
                        cur_img = path.join(tmp_folder, f"input_{i}.nii")
                        average_args += f" {cur_affine_file} {cur_f3d_file} {cur_img}"

                else:
                    # All the result images are directly averaged during the last step
                    average_args = path.join(
                        tmp_folder, f"average_nonrigid_it_{cur_it+1}.nii"
                    )
                    average_args += " -avg"
                    for i, _ in enumerate(input_imgs):
                        cur_img = path.join(
                            tmp_folder,
                            f"nrr_res_input_{i}_it{cur_it+1}.nii",
                        )
                        average_args += f" {cur_img}"

                average_cmd = f"reg_average {average_args}"
                assert call_niftyreg(average_cmd, verbose), "Average command failed!"
                average_image = path.join(
                    tmp_folder,
                    f"average_nonrigid_it_{cur_it+1}.nii",
                )

                pbar.update()

        average = read_nifti(average_image, output_nan=nan_out)

        res = []
        for i, _ in enumerate(input_imgs):
            cur_img = path.join(tmp_folder, f"nrr_res_input_{i}_it{cur_it+1}.nii")
            res.append(read_nifti(cur_img, output_nan=nan_out))

        if normalize:
            res = [x * (y - z) + z for x, y, z in zip(res, max_val, min_val)]

    return average, res

from logging import Logger
import numpy as np


def cpxMultilook(*, cpx_data, az_look=1, ra_look=1, tar=[0, 1, 2], logger=Logger):
    """
    Multilook complex data.

    Parameters
    ----------
    cpx_data: np.ndarray
        complex data. cpx_data must be a 2D (single image) or 3D (multiple images) array,
        each image is multilooked individually.
    az_look: int
        azimuth look. Default: 1)
    ra_look: int
        range look. Default: 1
    tar: list
        cpx_data dimension order. Default: [0, 1, 2] for (time, az, ra). [2, 0, 1] for (az, ra, time)

    Returns
    -------
    output: np.ndarray
        multilooked data
    """

    if az_look == 1 and ra_look == 1:
        logger.debug(msg="No multilooking applied.")
        return cpx_data
    else:
        logger.debug(msg="Start multilooking data with azimuth look: {} and range look: {} and dimension order{}."
                     .format(az_look, ra_look, tar))

    if cpx_data.ndim == 2:
        cpx_data = cpx_data[None, ...]
    elif cpx_data.ndim != 3:
        msg = "Input must be a 2D or 3D array. Input shape: {}".format(cpx_data.shape)
        logger.error(msg=msg)
        raise ValueError(msg)

    nt, naz, nrg = [cpx_data.shape[i] for i in tar]

    naz_ml = naz // az_look
    nrg_ml = nrg // ra_look

    logger.debug(msg="Output data shape: {} time, {}->{} azimuth, {}->{} range.".format(nt, naz, naz_ml, nrg, nrg_ml))

    if tar == [0, 1, 2]:  # (time, az, ra)
        cpx_cropped = cpx_data[:, :naz_ml * az_look, :nrg_ml * ra_look]
        cpx_blocks = cpx_cropped.reshape((nt, naz_ml, az_look, nrg_ml, ra_look))
        multilooked = np.nanmean(cpx_blocks, axis=(2, 4))
    elif tar == [2, 0, 1]:  # (az, ra, time)
        cpx_cropped = cpx_data[:naz_ml * az_look, :nrg_ml * ra_look, :]
        cpx_blocks = cpx_cropped.reshape((naz_ml, az_look, nrg_ml, ra_look, nt))
        multilooked = np.nanmean(cpx_blocks, axis=(1, 3))
    else:
        logger.error(msg="Invalid dimension order specified in tar {}. Only [0, 1, 2] and [2, 0, 1] are accepted".format(tar))
        raise ValueError("Invalid dimension order specified in tar")

    output = multilooked[0] if nt == 1 else multilooked

    logger.debug(msg="Multilooking done.")

    return output

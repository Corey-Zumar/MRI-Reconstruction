import os
import numpy as np
import nibabel as nib

from .constants import SLICE_WIDTH, SLICE_HEIGHT

ANALYZE_DATA_EXTENSION_IMG = ".img"


def _center_crop(img_data):
    slice_width, slice_height, _ = img_data.shape
    if slice_width < SLICE_WIDTH:
        raise Exception(
            "The width of each MRI image slice must be at least {} pixels!".
            format(SLICE_WIDTH))
    elif slice_height < SLICE_HEIGHT:
        raise Exception(
            "The height of each MRI image slice must be at least {} pixels!".
            format(SLICE_HEIGHT))

    width_crop = (slice_width - SLICE_WIDTH) // 2
    height_crop = (slice_height - SLICE_HEIGHT) // 2

    if width_crop > 0:
        img_data = img_data[width_crop:-width_crop, :, :]
    if height_crop > 0:
        img_data = img_data[:, height_crop:-height_crop, :]

    return img_data


def normalize(img_data):
    img_data -= img_data.min()
    img_data = img_data / img_data.max()
    img_data = img_data * 255.0
    return img_data


def load_image(analyze_img_path):
    """
    Loads an MRI image from a stored representation
    in Analyze 7.5 format.

    Note: In order to load an image with path '<img_name>.img',
    an Analyze header with name '<img_name>.hdr' must be present in the
    same directory as the specified image

    Parameters
    ------------
    analyze_img_path : str
        The path to the Analyze image path (with ".img" extension)

    Returns
    ------------
    nibabel.analyze.AnalyzeImage
        An nibabel image object representing the Analyze 7.5
        MRI image specified by the provided paths
    """
    print("Loading image with path: {}".format(analyze_img_path))
    return nib.load(analyze_img_path)


def load_image_data(analyze_img_path):
    """
    Loads, center-crops, and scales an MRI image from a stored representation
    in Analyze 7.5 format.

    Note: In order to load an image with path '<img_name>.img',
    an Analyze header with name '<img_name>.hdr' must be present in the
    same directory as the specified image

    Parameters
    ------------
    analyze_img_path : str
        The path to the Analyze image path (with ".img" extension)

    Returns
    ------------
    np.ndarray
        A numpy representation of the cropped and scaled
        image data with shape (TARGET_WIDTH, TARGET_HEIGHT, <number_of_slices>).
        The datatype of this array is `np.float32`
    """

    img_data = load_image(analyze_img_path).get_data()
    img_data = np.squeeze(img_data).astype(np.float32)
    img_data = _center_crop(img_data)
    img_data = normalize(img_data)

    return img_data


def get_image_file_paths(dir_path):
    img_paths = []
    dir_walk = os.walk(dir_path)
    for walk_item in dir_walk:
        dir_name, _, file_subpaths = walk_item
        relevant_subpaths = [
            path for path in file_subpaths
            if ANALYZE_DATA_EXTENSION_IMG in path
        ]
        relevant_paths = [
            os.path.join(dir_name, path) for path in relevant_subpaths
        ]
        img_paths += relevant_paths

    return img_paths

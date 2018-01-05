import sys
import os
import argparse
import keras
import numpy as np
import json

from ..utils import load_image_data, get_image_file_paths, normalize, create_output_dir
from ..utils.constants import SLICE_WIDTH, SLICE_HEIGHT

from matplotlib import pyplot as plt

# Data loading
ANALYZE_DATA_EXTENSION_IMG = ".img"

# Result writing
SFX_DIFF_PLOTS = "diffs"

FNAME_LOSS_EVALUATION = "results.json"

def load_net(net_path):
    """
    Loads the serialized deep neural network that
    will be used during the reconstruction process

    Parameters
    ------------
    net_path : str
        The path to the reconstruction network

    Returns
    ------------
    The reconstruction network, represented as 
    a Keras model instance
    """

    return keras.models.load_model(net_path)


def reconstruct_slice(fnet, undersampled_slice):
    """
    Reconstructs an undersampled, Analyze-formatted MR image 

    Parameters
    ------------
    fnet : Keras model
        The reconstruction network
    undersampled_slice : np.ndarray
        The undersampled slice that will be reconstructed,
        represented as a numpy array of datatype `np.float32`
        and shape (SLICE_WIDTH, SLICE_HEIGHT)

    Returns
    ------------
    np.ndarray
        The reconstructed image slice, represented as a numpy array
        of datatype `np.float32` and shape (SLICE_WIDTH, SLICE_HEIGHT)
    """

    # Reshape input to shape (1, SLICE_WIDTH, SLICE_HEIGHT, 1)
    fnet_input = np.expand_dims(undersampled_slice, 0)
    fnet_input = np.expand_dims(fnet_input, -1)

    fnet_output = fnet.predict(fnet_input)
    fnet_output = normalize(fnet_output)
    fnet_output = np.squeeze(fnet_output)

    return fnet_output

def eval_diff_plot(net_path,
                   img_path,
                   results_dir,
                   exp_name=None):
    """
    Given a path to an MR image in Analyze format, performs subsampling
    followed by reconstruction on all image slices and produces plots 
    containing the following grayscale images for each slice:
    
    1. The original (undersampled) slice
    2. The reconstructed slice

    These plots are then saved under the directory specified by `results_dir`

    Parameters
    ------------
    net_path : str
        The path to the serialized deep neural network that
        will be used during the reconstruction process
    img_path : str
        The path to an MR image in Analyze 7.5 format
        with extension `.img`
    results_dir : str
        The directory under which to save the plots for each slice
    exp_name : str
        (optional) The experiment name to include when naming the
        results subdirectory. This subdirectory will contain all
        of the plots that are produced (one plot per slice).
    """

    undersampled_img = load_image_data(analyze_img_path=img_path)
    undersampled_img = np.moveaxis(undersampled_img, -1, 0)

    fnet = load_net(net_path=net_path)

    output_dir_path = create_output_dir(
        base_path=results_dir, suffix=SFX_DIFF_PLOTS, exp_name=exp_name)

    for slice_idx in range(len(undersampled_img)):
        reconstructed_slice = reconstruct_slice(
            fnet=fnet,
            undersampled_slice=undersampled_img[slice_idx])

        plt.figure(figsize=(15, 15))
        plt.subplot(121), plt.imshow(undersampled_img[slice_idx], cmap='gray')
        plt.title('Original Slice'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(
            np.squeeze(reconstructed_slice), cmap='gray')
        plt.title('Reconstructed Slice'), plt.xticks([]), plt.yticks([])

        plot_path = os.path.join(output_dir_path, "{}.png".format(slice_idx))
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        print("Saved diff plot for slice {idx} to {pp}".format(
            idx=slice_idx, pp=plot_path))

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct MR images from undersampled image data')
    parser.add_argument(
        '-i',
        '--img_path',
        type=str,
        help=
        "The path to an MR image to reconstruct and diff-plot"
    )
    parser.add_argument(
        '-n', '--net_path', type=str, help="The path to a trained FNet")
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/tmp',
        help="The base directory to which to write evaluation results")
    parser.add_argument(
        '-e',
        '--experiment_name',
        type=str,
        help="The name of the experiment to use when writing evaluation results"
    )

    args = parser.parse_args()

    if not args.net_path:
        raise Exception("--net_path must be specified!")
    if not args.img_path:
        raise Exception("--img_path must be specified!")

    eval_diff_plot(
        net_path=args.net_path,
        img_path=args.img_path,
        results_dir=args.results_dir,
        exp_name=args.experiment_name)

if __name__ == "__main__":
    main()

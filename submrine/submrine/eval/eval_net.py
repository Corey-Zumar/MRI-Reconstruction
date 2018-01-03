import sys
import os
import argparse
import keras
import numpy as np

from ..utils import subsample, correct_output, load_image_data, get_image_file_paths, normalize
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

# Data loading
ANALYZE_DATA_EXTENSION_IMG = ".img"

# Network evaluation
LOSS_TYPE_MSE = "mse"
LOSS_TYPE_SSIM = "ssim"

# Loss computation
NUM_EVALUATION_SLICES = 35

def load_image(raw_img_path, substep, low_freq_percent):
    original_img = load_image_data(analyze_img_path=raw_img_path)
    subsampled_img, subsampled_K = subsample(analyze_img_data=original_img, 
                                             substep=substep, 
                                             low_freq_percent=low_freq_percent)

    original_img = np.moveaxis(original_img, -1, 0)
    subsampled_img = np.moveaxis(subsampled_img, -1, 0)
    subsampled_K = np.moveaxis(subsampled_K, -1, 0)

    return subsampled_img, subsampled_K, original_img

def load_net(net_path):
    return keras.models.load_model(net_path)

def eval_diff_plot(net_path, img_path, substep, low_freq_percent):
    [
        test_subsampled, 
        test_subsampled_K, 
        test_original
    ] = load_image(raw_img_path=img_path, 
                   substep=substep, 
                   low_freq_percent=low_freq_percent)

    fnet = load_net(net_path=net_path)

    # Reshape input to shape (1, SLICE_WIDTH, SLICE_HEIGHT, 1)
    fnet_input = np.expand_dims(test_subsampled[70], 0)
    fnet_input = np.expand_dims(fnet_input, -1)

    fnet_output = fnet.predict(fnet_input)
    fnet_output = normalize(fnet_output)
    fnet_output = np.squeeze(fnet_output)

    correction_subsampled_input = np.squeeze(test_subsampled_K[70])
    corrected_output = correct_output(subsampled_img_K=correction_subsampled_input,
                                      network_output=fnet_output,
                                      substep=substep,
                                      low_freq_percent=low_freq_percent)

    plt.subplot(121), plt.imshow(test_original[70], cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.squeeze(corrected_output), cmap='gray')
    plt.title('Corrected Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def compute_loss(output, original, loss_type):
    output = np.array(output, dtype=np.float64) / 255.0
    original = np.array(original, dtype=np.float64) / 255.0
    if loss_type == LOSS_TYPE_MSE:
        return np.mean((output - original)**2)
    elif loss_type == LOSS_TYPE_SSIM:
        return ssim(output, original)
    else:
        raise Exception("Attempted to compute an invalid loss!")


def eval_loss(net_path, data_path, size, loss_type, substep, low_freq_percent):
    fnet = load_net(net_path)
    img_paths = get_image_file_paths(data_path)
    losses = []
    aliased_losses = []
    for img_path in img_paths:
        [
            test_subsampled, 
            test_subsampled_K, 
            test_original
        ] = load_image(raw_img_path=img_path, 
                       substep=substep, 
                       low_freq_percent=low_freq_percent)
        num_slices = len(test_subsampled)
        if num_slices > NUM_EVALUATION_SLICES:
            slice_idxs_low = (num_slices - NUM_EVALUATION_SLICES) // 2
            slice_idxs_high = slice_idxs_low + NUM_EVALUATION_SLICES
            slice_idxs = range(slice_idxs_low, slice_idxs_high)
        else:
            slice_idxs = range(num_slices)

        for slice_idx in slice_idxs:
            fnet_input = np.expand_dims(test_subsampled[slice_idx], -1)
            fnet_output = fnet.predict(fnet_input)
            fnet_output = normalize_data(fnet_output)
            fnet_output = np.squeeze(fnet_output)
            corrected_output = correct_output(subsampled_img_K=test_subsampled_k[slice_idx],
                                              network_output=fnet_output,
                                              substep=substep,
                                              low_freq_percent=LOW_FREQ_PERCENT)

            ground_truth = normalize_data(test_original[slice_idx])
            loss = compute_loss(
                output=corrected_output,
                original=ground_truth,
                loss_type=loss_type)
            losses.append(loss)
            aliased_loss = compute_loss(
                output=test_subsampled[slice_idx],
                original=ground_truth,
                loss_type=loss_type)
            aliased_losses.append(aliased_loss)
            print("Evaluated {} images".format(len(losses)))
            if len(losses) >= size:
                break

        else:
            continue

        break

    mean = np.mean(losses)
    std = np.std(losses)

    aliased_mean = np.mean(aliased_losses)
    aliased_std = np.std(aliased_losses)

    print("Aliased MEAN: {}, Aliased STD: {}, MEAN: {}, STD: {}".format(
        aliased_mean, aliased_std, mean, std))

    return losses


def main():
    parser = argparse.ArgumentParser(
        description='Train FNet on MRI image data')
    parser.add_argument(
        '-i',
        '--img_path',
        type=str,
        help="The path to a full-resolution MR image to subsample, reconstruct, and diff-plot")
    parser.add_argument(
        '-n', '--net_path', type=str, help="The path to a trained FNet")
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        help=
        "The path to a test set of full-resolution MR images to evaluate for loss computation"
    )
    parser.add_argument(
        '-s',
        '--substep',
        type=int,
        default=4,
        help="The substep used for subsampling (4 in the paper)")
    parser.add_argument(
        '-f',
        '--lf_percent',
        type=float,
        default=.04,
        help=
        "The percentage of low frequency data to retain when subsampling training images"
    )
    parser.add_argument(
        '-t',
        '--test_size',
        type=str,
        default=400,
        help="The size of the test set (used if --data_path is specified)")
    parser.add_argument(
        '-l',
        '--loss_type',
        type=str,
        default='mse',
        help="The type of evaluation loss. One of: 'mse', 'ssim'")

    args = parser.parse_args()

    if not args.substep:
        raise Exception("--substep must be specified!")
    elif not args.net_path:
        raise Exception("--net_path must be specified!")

    if args.img_path:
        eval_diff_plot(net_path=args.net_path,
                       img_path=args.img_path, 
                       substep=args.substep,
                       low_freq_percent=args.lf_percent)
    elif args.data_path:
        if not args.test_size:
            raise Exception("--test_size must be specified!")

        eval_loss(net_path=args.net_path,
                  data_path=args.data_path,
                  size=int(args.test_size),
                  loss_type=args.loss_type,
                  substep=args.substep,
                  low_freq_percent=args.lf_percent)
    else:
        raise Exception(
            "Either '--img_path' or '--data_path' must be specified!")


if __name__ == "__main__":
    main()

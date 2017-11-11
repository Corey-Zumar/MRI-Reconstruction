import sys
import os
import argparse
import nibabel as nib
import keras
import numpy as np

from utils import Subsample

from matplotlib import pyplot as plt

def load_image(image_path):
    original_img = nib.load(image_path)
    original_data = np.squeeze(original_img.get_data())
    original_data = np.moveaxis(original_data, -1, 0).reshape(128, 256, 256, 1)

    subsampled_img = Subsample.subsample(image_path)
    subsampled_data = np.moveaxis(subsampled_img, -1, 0).reshape(128, 256, 256, 1)

    return subsampled_data, original_data

def load_net(net_path):
    return keras.models.load_model(net_path)

def main():
    parser = argparse.ArgumentParser(description='Train FNet on MRI image data')
    parser.add_argument('-i', '--img_path', type=str, help="The path to an OASIS MRI image to evaluate")
    parser.add_argument('-n', '--net_path', type=str, help="The path to a trained FNet")
    args = parser.parse_args()

    test_subsampled, test_original = load_image(args.img_path)
    fnet = load_net(args.net_path)

    result = fnet.predict(test_subsampled[70].reshape(1, 256, 256, 1))

    magnitude_spectrum = 20*np.log(np.abs(result.reshape(256,256)))
    plt.subplot(121),plt.imshow(test_subsampled[70].reshape(256,256), cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(result.reshape(256,256), cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()

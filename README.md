# MRI-Reconstruction
An open source implementation of the deep learning platform for undersampled MRI reconstruction described by Hyun et. al. (https://arxiv.org/pdf/1709.02576.pdf) 

## Usage

The current implementation is tailored to support the reconstruction of sagittal plane brain images from the free [OASIS](http://www.oasis-brains.org/) dataset.

### Training

1. Download and extract a [disc](http://www.oasis-brains.org/app/template/Tools.vm) of brain images from the OASIS data set.

2. Train the Keras MR image reconstruction network on a subset of the data disc of size `S`:

   ```sh
   $ python train_net.py -d /path/to/disc/root -s S
   ```
   
### Evaluating

1. Obtain a sagittal plane brain test image in [Analyze 7.5](https://rportal.mayo.edu/bir/ANALYZE75.pdf) format.

2. Evaluate the reconstruction pipeline on a sagittal plane brain test image as follows:

   ```sh
   $ python eval_net.py -i /path/to/test/brain/image.img -n /path/to/trained/keras/network.hdf5 
   ```
   
   This will subsample the test image, reconstruct the subsampled copy, and plot both the test and reconstructed images.


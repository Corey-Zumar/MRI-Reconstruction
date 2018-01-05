# Pretrained [OASIS](http://www.oasis-brains.org/) Networks

This directory contains pretrained networks for MR image reconstruction. These networks were trained using the Keras framework on the
[OASIS](http://www.oasis-brains.org/) dataset.

## Network Attributes

* Network names are of the form `fnet-oasis-<substep>.hdf5`, where `<substep>` is the substep used when subsampling MR images for training.

* These networks were trained for 2000 epochs with a batch size of 256 image slices. The training set consisted of 1400 total slices.

* For all three networks, MR images in the training set were subsampled using a `low_frequency_percentage` value of `.04`.

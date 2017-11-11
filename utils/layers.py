from keras.layers import UpSampling2D
from keras import backend as K

"""
This unpooling layer has been taken and renamed
from https://github.com/nanopony/keras-convautoencoder/blob/master/autoencoder_layers.py
"""

class Unpool2D(UpSampling2D):
    '''Simplar to UpSample, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool_input, pool_output, *args, **kwargs):
    	self.pool_input = pool_input
    	self.pool_output = pool_output
       	UpSampling2D.__init__(self, *args, **kwargs)

    def get_config(self):
    	config = {
    		"pool_input" : self.pool_input,
    		"pool_output" : self.pool_output
    	}
    	base_config = super(Unpool2D, self).get_config()
    	return dict(list(base_config.items()) + list(config.items()))

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self.pool_output), wrt=self.pool_input) * output

        return f
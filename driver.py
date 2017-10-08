from Subsample import subsample
from matplotlib import pyplot as plt

imgarr = subsample("img2.img", 4, 0.05)
plt.imshow(imgarr[:,:,70], cmap='gray')
plt.show()

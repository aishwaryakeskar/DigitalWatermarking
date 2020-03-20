import numpy as np
import pywt
from matplotlib import pyplot as plt
import scipy.misc
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import scipy.ndimage
import cv2


#x = pywt.data.camera().astype(np.float32)
x = cv2.imread("lena.jpg",0).astype(np.float32)

shape = x.shape
print(shape)
#plt.imshow(x)

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots


fig, axes = plt.subplots(5, (max_lev + 1), figsize=[10, 6])

for level in range(0, max_lev + 1):
    if level == 0:
        # show the original image before decomposition

        axes[0, 0].set_axis_off()
        axes[2, 0].set_axis_off()
        axes[3, 0].set_axis_off()
        axes[4, 0].set_axis_off()

        axes[1, 0].imshow(x, cmap=plt.cm.gray)
        axes[1, 0].set_title('Image')
        axes[1, 0].set_axis_off()
        continue

    # plot subband boundaries of a standard DWT basis
    draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level], label_levels=label_levels)
    axes[0, level].set_title('{} level\ndecomposition'.format(level))

    # compute the 2D DWT
    c = pywt.wavedec2(x, 'haar', mode='periodization', level=level)

    # normalize each coefficient array independently for better visibility
    c[0] /= np.abs(c[0]).max()


    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]



    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)

#LL
    axes[1, level].imshow(arr[slices[0]], cmap=plt.cm.gray)
    axes[1, level].set_title('Coefficients\n({} level 0)'.format(level))
    axes[1, level].set_axis_off()
    print("Slices[0] shape: ",np.shape(slices[0]))

#HL
    axes[2, level].imshow(arr[slices[1]['da']], cmap=plt.cm.gray)
    axes[2, level].set_title('Coefficients\n({} level 1 da)'.format(level))
    axes[2, level].set_axis_off()
    print("Slices[1][da] shape: ",np.shape(slices[1]['da']))

#LH
    axes[3, level].imshow(arr[slices[1]['ad']], cmap=plt.cm.gray)
    axes[3, level].set_title('Coefficients\n({} level 1 ad)'.format(level))
    axes[3, level].set_axis_off()
    print("Slices[1][ad] shape: ",np.shape(slices[1]['ad']))

#HH
    axes[4, level].imshow(arr[slices[1]['dd']], cmap=plt.cm.gray)
    axes[4, level].set_title('Coefficients\n({} level 1 dd)'.format(level))
    axes[4, level].set_axis_off()
    print("Slices[1][dd] shape: ",np.shape(slices[1]))


plt.tight_layout()
plt.show()


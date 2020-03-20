import cv2
import pywt
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt
import random


class Components():
    Coefficients = []
    U = None
    S = None
    V = None


class watermarking():
    def __init__(self, watermark_path="watermark1.jpg", ratio=0.1, wavelet="haar", level=3, x=[0.1],
                 cover_image="lena.jpg"):
        self.level = level
        self.wavelet = wavelet
        self.ratio = x[0]
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        # print("Shape", self.shape_watermark)

        # W_components is watermark image
        # img_components is cover image
        self.W_components = Components()
        self.img_components = Components()
        self.W_components.Coefficients, self.W_components.U, \
        self.W_components.S, self.W_components.V, self.W_components.LL = self.calculate(watermark_path, 3)
        self.x = x
        self.cover_image_data = cv2.imread(cover_image, 0)
        if self.cover_image_data.shape != (512, 512):
            self.cover_image_data.resize(512, 512)
            cv2.imwrite(cover_image, self.cover_image_data)
            # print(self.cover_image.shape)
        # self.W_ndarr = cv2.imread(watermark_path,0)
        # print("watermark", self.W_ndarr.shape)

    def calculate(self, img, lvl):
        '''
        img is either numpy array or path of Image
        This function returns the SVD components of the image
        '''
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=lvl)
        self.shape_LL = Coefficients[0].shape

        arr, slices = pywt.coeffs_to_array(Coefficients)

        LL = arr[slices[0]]
        HL = arr[slices[1]['da']]
        LH = arr[slices[1]['ad']]
        HH = arr[slices[1]['dd']]

        # plt.imshow(arr[slices[0]], cmap=plt.cm.gray), plt.title('slices[0] LL')            # LL
        # plt.show()
        # plt.imshow(arr[slices[1]['da']], cmap=plt.cm.gray), plt.title('slices[1][da] HL')  # HL
        # plt.show()
        # plt.imshow(arr[slices[1]['ad']], cmap=plt.cm.gray), plt.title('slices[1][ad] LH')  # LH
        # plt.show()
        # plt.imshow(arr[slices[1]['dd']], cmap=plt.cm.gray), plt.title('slices[1][dd] HH')  # HH
        # plt.show()

        U, S, V = np.linalg.svd(LL)
        return Coefficients, U, S, V, LL

    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        s:  1D array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S

    def recover(self, name):
        '''
        To recover the image from the svd components and DWT
        :param name:
        '''
        components = eval("self.{}_components".format(name))
        s = eval("self.S_{}".format(name))
        components.Coefficients[0] = components.U.dot(self.diag(s)).dot(components.V)
        return pywt.waverec2(components.Coefficients, wavelet=self.wavelet)

    def watermark(self, img="lena.jpg", path_save=None):
        '''
        This is the main function for image watermarking.
        :param img: image path or numpy array of the image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        self.img_components.Coefficients, self.img_components.U, \
        self.img_components.S, self.img_components.V, self.img_components.LL = self.calculate(img, 3)
        self.embed()
        img_rec = self.recover("img")  # watermarked image
        cv2.imwrite(path_save, img_rec)
        #self.gaussian_smoothning(X_imgs="watermarked_lena.jpg")

    def extracted(self, image_path="watermarked_lena.jpg", ratio=None,
                  extracted_watermark_path="watermark_extracted.jpg"):
        '''
        Extracted the watermark from the given image.
        '''
        if not extracted_watermark_path:
            extracted_watermark_path = "watermark_extracted.jpg"
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, self.shape_watermark)
        img_components = Components()  # watermarked image
        img_components.Coefficients, img_components.U, img_components.S, img_components.V, img_components.LL = self.calculate(img, 3)
        ratio_ = self.ratio if not self.x[0] else ratio
        self.S_W = (img_components.S - self.img_components.S) / self.x[0]
        watermark_extracted = self.recover("W")
        watermark_extracted = cv2.GaussianBlur(watermark_extracted, (5, 5), 0)
        cv2.imwrite(extracted_watermark_path, watermark_extracted)

    def embed(self):
        # print("Level: ", self.level)
        self.S_img = self.img_components.S + self.x[0] * self.W_components.S * \
                     (self.img_components.S.max() / self.W_components.S.max())

    def gaussian_smoothning(self, X_imgs="watermarked_lena.jpg"):
        input_image = cv2.imread(X_imgs)
        output = cv2.GaussianBlur(input_image, (5, 5), 0)
        cv2.imwrite('watermarked_lena.jpg', output)

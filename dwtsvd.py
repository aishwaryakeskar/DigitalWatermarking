import cv2
import pywt
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt
import random


class Components:
    Coefficients = []
    LL = None
    sub_bands = None


class DigitalWatermark:
    def __init__(self, watermark_path="watermark1.jpg", ratio=0.1, wavelet="haar", level=3,  x=[0.1],
                 cover_image="lena.jpg"):
        self.level = level
        self.wavelet = wavelet
        self.ratio = x[0]
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        self.x = x
        self.wmk_img_components = Components()
        self.cover_img_components = Components()

        self.cover_image_data = cv2.imread(cover_image, 0)
        if self.cover_image_data.shape != (512, 512):
            self.cover_image_data.resize(512, 512)
            cv2.imwrite(cover_image, self.cover_image_data)

    def calculate_dwt(self, img, lvl):
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=lvl)

        coeff_arr, slices = pywt.coeffs_to_array(coefficients)

        LL = coeff_arr[slices[0]]
        HL = coeff_arr[slices[1]['da']]
        LH = coeff_arr[slices[1]['ad']]
        HH = coeff_arr[slices[1]['dd']]

        self.shape_LL = coefficients[0].shape
        self.shape_HH = HH.shape
        self.shape_HL = HL.shape
        self.shape_LH = LH.shape

        sub_bands = LL, HL, LH, HH
        return LL, sub_bands, coeff_arr, slices


    def calculate_svd_LL(self, LL):
        U, S, V = np.linalg.svd(LL)

        return U, S, V


    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        s:  1D array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        # print("S shape: ", S.shape)
        return S


    # def recover(self, name, CLL_S, CLL_V):
    #     '''
    #     To recover the image from the svd components and DWT
    #     :param name:
    #     '''
    #     components = eval("self.{}_components".format(name))
    #     # cll = eval("self.CLL_{}".format(name))
    #     components.LL = components.ULL.dot(CLL_S).dot(CLL_V)
    #     # components.LL = components.ULL.dot(self.diag(CLL_S)).dot(components.VLL)
    #     coeffs_from_arr = pywt.array_to_coeffs(components.coeff_arr, components.slices, output_format='wavedec2')
    #     return pywt.waverec2(coeffs_from_arr, wavelet=self.wavelet)

    def anti_svd(self, name, CLL_S, CLL_V):
        components = eval("self.{}_components".format(name))
        # cll = eval("self.CLL_{}".format(name))
        components.LL = components.ULL.dot(CLL_S).dot(CLL_V)
        # components.LL = components.ULL.dot(self.diag(CLL_S)).dot(components.VLL)


    def inverse_dwt(self, name):
        components = eval("self.{}_components".format(name))

        coeffs_from_arr = pywt.array_to_coeffs(components.coeff_arr, components.slices, output_format='wavedec2')
        return pywt.waverec2(coeffs_from_arr, wavelet=self.wavelet)

    def watermark(self, img="lena.jpg", watermark_path="watermark1.jpg", path_save=None):
        '''
        This is the main function for image watermarking.
        :param cover_img: image path or numpy array of the cover image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        # Cover Image
        self.cover_img_components.LL, self.cover_img_components.sub_bands, \
        self.cover_img_components.coeff_arr, self.cover_img_components.slices = self.calculate_dwt(img, 3)
        self.cover_img_components.ULL, self.cover_img_components.SLL, self.cover_img_components.VLL = self.calculate_svd_LL(
            self.cover_img_components.LL)
        # Watermark Image
        self.wmk_img_components.LL, self.wmk_img_components.sub_bands, \
        self.wmk_img_components.coeff_arr, self.wmk_img_components.slices = self.calculate_dwt(watermark_path, 1)
        # Embed Watermark
        self.embed()
        self.cover_img_components.CLL_U, self.cover_img_components.CLL_S, self.cover_img_components.CLL_V = self.calculate_svd_LL(self.CLL_cover_img)
        self.anti_svd("cover_img", self.cover_img_components.CLL_S, self.cover_img_components.CLL_V)
        recovered_image = self.inverse_dwt("cover_img")  # watermarked image
        cv2.imwrite(path_save, recovered_image)


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
        wmkd_img_components = Components()  # watermarked image

        wmkd_img_components.LL, wmkd_img_components.sub_bands, \
        wmkd_img_components.coeff_arr, wmkd_img_components.slices = self.calculate_dwt(img, 3)

        wmkd_img_components.ULL, wmkd_img_components.SLL, wmkd_img_components.VLL = self.calculate_svd_LL(wmkd_img_components.LL)

        ratio_ = self.ratio if not self.x[0] else ratio

        CWLL = self.cover_img_components.CLL_U.dot(self.diag(wmkd_img_components.SLL)).dot(wmkd_img_components.VLL)
        print("self.cover_img_components.CLL_U shape: ",self.cover_img_components.CLL_U.shape)
        print("wmkd_img_components.SLL shape: ",wmkd_img_components.SLL.shape)
        print("self.wmkd_img_components.VLL shape: ",wmkd_img_components.VLL.shape)

        self.SLL_W = (CWLL - self.cover_img_components.SLL) / self.x[0]
        print("CLL_cover_img shape: ",self.CLL_cover_img.shape)

        coeffs_from_arr = pywt.array_to_coeffs(wmkd_img_components.coeff_arr, wmkd_img_components.slices, output_format='wavedec2')
        wmk_extracted =  pywt.waverec2(coeffs_from_arr, wavelet=self.wavelet)

        # watermark_extracted = self.recover("W")
        # watermark_extracted = cv2.GaussianBlur(watermark_extracted, (5, 5), 0)
        cv2.imwrite(extracted_watermark_path, wmk_extracted)

    def embed(self):
        self.CLL_cover_img = self.cover_img_components.SLL + self.x[0] * self.wmk_img_components.LL

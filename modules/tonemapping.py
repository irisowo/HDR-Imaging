import numpy as np
import cv2
import os


class ToneMapping():

    def __init__(self, tm_method='local', alpha=0.7, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma  # for gamma correction
        self.tm_method = tm_method
        self.delta = 1e-4

    def gamma_correction(self, img, gamma=1.0):
        return np.clip(np.power(img, self.gamma) * 255.0, 0, 255).astype(np.uint8)

    def modify_lightness_saturation(self, img, lightness=-10, saturation=20):
        # BGR > HLS after normalizing img to 0~1 in float32
        hlsImg = cv2.cvtColor(img.astype(np.float32)/255.0, cv2.COLOR_BGR2HLS)

        # lightness
        hlsImg[:, :, 1] = np.clip(
            hlsImg[:, :, 1] * (1 + lightness / 100.0), 0, 1)
        # saturation
        hlsImg[:, :, 2] = np.clip(
            hlsImg[:, :, 2] * (1 + saturation / 100.0), 0, 1)

        # HLS -> BGR
        result_img = (
            (cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255).astype(np.uint8))
        return result_img

    def global_operator(self, Lw):
        Lw_mu = np.exp(np.mean(np.log(self.delta + Lw)))
        Lm = (self.alpha / Lw_mu) * Lw

        # Lwhite is the smallest luminance to be mapped to 1
        Lwhite = np.max(Lm)
        Ld = (Lm * (1 + (Lm / (Lwhite ** 2)))) / (1 + Lm)

        ldr = self.gamma_correction(Ld)
        print(f'[Global Tonemapping] Solved')

        return ldr.astype(np.uint8)

    def gaussian_blurs(self, im, srange=15, a=0.8, alpha1=0.7, alpha2=3.2, epsilon=0.3):
        '''
        param srange: range of scale s for gaussian blur 
        param a: we simplify (2 ** fi) * alpha into a 
        '''
        h, w = im.shape
        num_s = int((srange+1)//2)

        blurs = np.zeros((h, w, num_s))
        s_indices = np.full((h, w), num_s-1, dtype=int)
        for i, s in enumerate(range(1, srange+1, 2)):
            V1 = cv2.GaussianBlur(im, (s, s), alpha1)
            V2 = cv2.GaussianBlur(im, (s, s), alpha2)
            Vs = np.abs((V1 - V2) / (a / (s ** 2) + V1))
            s_indices[np.where(Vs < epsilon)] = i
            blurs[:, :, i] = V1

        h_indices, w_indices = np.indices((h, w))
        blur_smax = blurs[h_indices, w_indices, s_indices]
        return blur_smax

    def local_operator(self, Lw_bgr):
        ldr = np.zeros_like(Lw_bgr, dtype=np.float32)
        Lw_mu = np.exp(np.mean(np.log(self.delta + Lw_bgr)))

        for c in range(3):
            Lw = Lw_bgr[:, :, c]
            Lm = (self.alpha / Lw_mu) * Lw
            Ls_blur = self.gaussian_blurs(Lm)
            Ld = Lm / (1 + Ls_blur)
            ldr[:, :, c] = self.gamma_correction(Ld)
        print(f'[Local Tonemapping] Solved')
        return ldr

    def solve(self, radiance_bgr):
        switcher = {
            'global': self.global_operator,
            'local': self.local_operator
        }

        # Tone mapping : HDR --> LDR
        ldr = switcher.get(self.tm_method)(radiance_bgr)

        # Finetune lightness and saturation
        ldr = self.modify_lightness_saturation(ldr, 5, 20)
        return ldr

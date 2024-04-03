import numpy as np
import cv2


class MTB_Alignment():
    def __init__(self, depth=6):
        self.depth = depth if depth > 0 else 1

    def adpative_threshoding(self, image):
        # Note that we change the median thresholding to adaptive thresholding
        bool_img = cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        return bool_img

    def shift_operation(self, image, pos):
        h, w, _ = image.shape
        M = np.float32([[1, 0, pos[0]],
                        [0, 1, pos[1]]])
        return cv2.warpAffine(image, M, (w, h))

    def compute_xor_error(self, img, ref_img):
        bool_ref_img = self.adpative_threshoding(ref_img)
        bool_img = self.adpative_threshoding(img)
        return np.sum(np.logical_xor(bool_img, bool_ref_img))

    def get_best_shift(self, img, ref_img, layer_idx):
        if layer_idx == 0:
            return np.zeros(2)
        else:
            # accumulate the shift from the previous layer
            accumulated_shift = self.get_best_shift(
                cv2.resize(img, dsize=None, fx=0.5, fy=0.5),
                cv2.resize(ref_img, dsize=None, fx=0.5, fy=0.5),
                layer_idx-1) * 2

            # 9 directions
            movements = np.vstack([[[i, j] for j in [0, -1, 1]]
                                   for i in [0, -1, 1]])

            # Find the optimal movement among 9 possible directions
            min_shift = np.zeros(2)
            min_diff = float('inf')
            for mov in movements:
                shifted_img = self.shift_operation(
                    img, np.add(accumulated_shift, mov))
                diff = self.compute_xor_error(shifted_img, ref_img)
                # update the optimal movement
                if diff < min_diff:
                    min_diff = diff
                    min_shift = mov

            accumulated_shift_layeri = np.add(accumulated_shift, min_shift)
            # print(f'Accumulated shift of layer {layer_idx} = {accumulated_shift_layeri}')

            return accumulated_shift_layeri

    def solve(self, images):
        # select the image with median exposure time as the reference image
        ref_img = images[len(images) // 2]

        aligned_imgs = []
        for img_i, img in enumerate(images):
            print(f'aliging image {img_i}...', end="\r")
            best_shift = self.get_best_shift(img, ref_img, self.depth - 1)
            aligned_imgs.append(self.shift_operation(img, best_shift))
            print('Done!', end='\r')
        print('[MTB] Solved')
        return np.array(aligned_imgs)

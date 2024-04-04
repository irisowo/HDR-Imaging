import numpy as np


def get_weighting_func(wtype='linear', mu=128, std=40):
    if (wtype == 'linear'):
        weights = np.hstack((np.arange(1, 129), np.arange(1, 129)[::-1]))
        return weights.astype(np.float32)

    elif (wtype == 'sin'):
        return 128 * np.sin((np.arange(0, 256)) * np.pi / 255) + 1

    else:  # wtype == 'gaussian'
        x = np.arange(256)
        gaussian = np.exp(-0.5 * ((x - mu) / std) ** 2)
        return 128 * gaussian / np.max(gaussian)


class Debevec():

    def __init__(self, weight_func='linear', lamb=10.0):
        self.wtype = weight_func
        self.lamb = lamb

    def sample_pixels(self, images):
        height, weight, channels = images[0].shape

        steps = 30 + 1
        hd, wd = height // steps, weight // steps
        grid_indices = [(i * hd, j * wd) for j in range(1, steps)
                        for i in range(1, steps)]
        samples_bgr = np.array(
            [[[img[x, y, c] for img in images] for x, y in grid_indices]for c in range(channels)], dtype=np.uint8)

        return samples_bgr

    def debevec_algo(self, images, t):
        # Sample n pixels
        samples_bgr = self.sample_pixels(images)

        n, p = samples_bgr[0].shape
        W = get_weighting_func(self.wtype)
        lnt = np.log(t)

        G_function = np.zeros((3, 256), dtype=np.float32)
        for c, samples in enumerate(samples_bgr):

            # Initialize the system of equations Ax = b
            A = np.zeros((n * p + 1 + 254, 256 + n), dtype=np.float32)
            B = np.zeros((A.shape[0], 1), dtype=np.float32)

            # 1. Include the data-fitting equations
            k = 0
            for i, pixel_i in enumerate(samples):
                for j, z_ij in enumerate(pixel_i):
                    w_ij = W[z_ij]
                    A[k, z_ij] = w_ij
                    A[k, 256 + i] = -w_ij
                    B[k, 0] = w_ij * lnt[j]
                    k += 1

            # 2. Fitting the curve by setting its middle value to 0
            A[k,  127] = 1
            k += 1

            # 3. Include the smoothness equations
            for z_i in range(1, 255):
                w_k = self.lamb * W[z_i]
                A[k, z_i - 1] = w_k
                A[k, z_i] = -2 * w_k
                A[k, z_i + 1] = w_k
                k += 1

            # Solve the system using SVD
            A_inv = np.linalg.pinv(A)
            G_function[c] = (A_inv @ B)[:256].ravel()

        print('[Debevec] Solved')
        return G_function

    def get_radianceMap(self, images, g_bgr, t):
        W = get_weighting_func(self.wtype)
        h, w, channels = images[0].shape
        lnE_bgr = np.zeros((h, w, channels)).astype(np.float32)

        for c in range(channels):
            g = g_bgr[c]
            sum_w = np.zeros((h, w), dtype=np.float32) + 1e-6
            sum_w_lnE = np.zeros((h, w), dtype=np.float32)
            lnt = np.log(t).astype(np.float32)

            for j, img in enumerate(images):
                zj = img[:, :, c].ravel()
                w_zj = W[zj].reshape(h, w)
                sum_w += w_zj
                sum_w_lnE += w_zj * (g[zj] - lnt[j]).reshape(h, w)

            lnE_bgr[:, :, c] = sum_w_lnE / sum_w

        radiance_bgr = np.exp(lnE_bgr).astype(np.float32)

        return radiance_bgr

    def solve(self, images, t):
        g_function = self.debevec_algo(images, t)
        radinace_bgr = self.get_radianceMap(images, g_function, t)
        return g_function, radinace_bgr

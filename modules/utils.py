import os
import cv2
import errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules.hdr import get_weighting_func


def read_images_and_exposure(img_dir):
    # check path
    if not os.path.exists(img_dir) or not os.path.isdir(img_dir):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), img_dir)

    # read the meta data
    df = pd.read_csv(os.path.join(img_dir, 'exposure_times.csv'), sep=',')

    # read the images
    images = [cv2.imread(os.path.join(img_dir, fn)) for fn in df.filename]

    # resize images
    # h, w, _ = images[0].shape
    # images = [cv2.resize(img, (w // 2 , h // 2)) for img in images]

    print(f'[Read] {len(images)} images with shape: {images[0].shape}')

    return np.array(images), np.array(df.shutter_time, dtype=np.float32)


def plot_response_curve(G_bgr, savedir):
    channels = ['blue', 'green', 'red']

    fig = plt.figure(figsize=(10, 10))
    plt.xlabel('lnX : Log Exposure X')
    plt.ylabel('Z: Pixel Value')

    for c in range(len(channels)):
        plt.plot(G_bgr[c],  np.arange(256), c=channels[c])

    fig.savefig(os.path.join(savedir, 'response_curve.png'),
                bbox_inches='tight', dpi=256)
    plt.close('all')
    print('[Plot g curve] Solved')


def plot_weighting_functions(savedir):
    import matplotlib.pyplot as plt
    x = np.arange(256)
    weights_linear = get_weighting_func('linear')
    weights_uniform = get_weighting_func('sin')
    weights_gaussian = get_weighting_func('gaussian')

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, weights_linear, label='Linear')
    plt.plot(x, weights_uniform, label='Sin')
    plt.plot(x, weights_gaussian, label='Gaussian')
    plt.title('Weighting Functions')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    fig.savefig(os.path.join(savedir, 'weighting_func.jpg'))

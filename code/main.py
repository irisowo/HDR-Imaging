import os
import cv2
import argparse

import modules
from modules.align import MTB_Alignment
from modules.tonemapping import ToneMapping
from modules.hdr import Debevec


parser = argparse.ArgumentParser('HDR Imaging')
parser.add_argument('--indir', default='../data/team26_3',
                    type=str, help='Input directory containing images and exposure_times.csv')
parser.add_argument('--outdir', default='../data/',
                    type=str, help='Output directory.')
parser.add_argument('-d', default=6, type=int,
                    help='Depth for image alignment.')
parser.add_argument('-w', default='gaussian',
                    choices=['linear', 'sin', 'gaussian'], type=str,
                    help='Weighting function for for Debevec method.')
parser.add_argument('--lamb', default=25.0, type=float,
                    help='Lambda for Debevec method.')
parser.add_argument('--alpha', default=0.7, type=float,
                    help='Alpha for tonemapping.')
parser.add_argument('--gamma', default=1/1.1, type=float,
                    help='Gamma for gamma correction.')
parser.add_argument('--tm', default='local',
                    choices=['global', 'local'], type=str, help='Tone mapping method.')
parser.add_argument('--seed', default=8763, type=int, help='Random seed.')


class HDR_Solution():

    def __init__(self, args):
        self.image_alignment = MTB_Alignment(args.d)
        self.debevec = Debevec(args.w, args.lamb)
        self.tone_mapping = ToneMapping(args.tm, args.alpha, args.gamma)
        self.args = args

    def write_img(self, fname, img):
        cv2.imwrite(os.path.join(self.args.outdir, fname), img)

    def solve(self, images, t):
        # [MTB] image alignment
        images = self.image_alignment.solve(images)

        # [DebeVec] Reconstruct response function
        g_bgr, radiance_bgr = self.debevec.solve(images, t)
        modules.plot_response_curve(g_bgr, self.args.outdir)
        # Save HDR image
        self.write_img('recovered_hdr_image.hdr', radiance_bgr)

        # [Local/Global TM] Convert HDR to LDR by tonemapping
        ldr = self.tone_mapping.solve(radiance_bgr)
        self.write_img(f'tone-mapped.png', ldr)


if __name__ == '__main__':
    args = parser.parse_args()

    # check path
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # read immages and exposure times
    images, times = modules.read_images_and_exposure(args.indir)

    # HDR
    hdr = HDR_Solution(args)
    hdr.solve(images, times)

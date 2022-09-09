'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import __init_paths
from face_enhancement import FaceEnhancement
from segmentation2face import Segmentation2Face
from utils import get_args

"""
hyper-parameters:
    --in_size: input size of GPEN
    
"""


if __name__ == '__main__':
    # 1. get args
    args = get_args()

    # model = {'name':'GPEN-BFR-512', 'size':512, 'channel_multiplier':2, 'narrow':1}
    # model = {'name':'GPEN-BFR-256', 'size':256, 'channel_multiplier':1, 'narrow':0.5}

    # 2. make output directory
    os.makedirs(args.outdir, exist_ok=True)

    # 3. load model
    processor = None
    if args.task == 'FaceEnhancement':
        processor = FaceEnhancement(args, in_size=args.in_size, model=args.model, use_sr=args.use_sr,
                                    device='cuda' if args.use_cuda else 'cpu')
    elif args.task == 'Segmentation2Face':
        processor = Segmentation2Face(in_size=args.in_size, model=args.model, is_norm=False,
                                      device='cuda' if args.use_cuda else 'cpu')
    # 4. load images and process
    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)

        img = cv2.imread(file, cv2.IMREAD_COLOR)  # HWC-BGR-uint8
        if not isinstance(img, np.ndarray):     # error: not an image
            print(filename, 'error')
            continue
        # img = cv2.resize(img, (0,0), fx=2, fy=2) # optional

        img_out, orig_faces, enhanced_faces = processor.process(img, aligned=args.aligned)

        # save results images
        img = cv2.resize(img, img_out.shape[:2][::-1])
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + f'_COMP{args.ext}'),
                    np.hstack((img, img_out)))
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + f'_GPEN{args.ext}'), img_out)

        if args.save_face:
            for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                of = cv2.resize(of, ef.shape[:2])
                cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + '_face%02d' % m + args.ext),
                            np.hstack((of, ef)))

        if n % 10 == 0: print(n, filename)

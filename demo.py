"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os
import cv2
import glob
import numpy as np
from face_enhancement import FaceEnhancement
from segmentation2face import Segmentation2Face
from utils import get_args

"""
hyper-parameters:
    face_enhancement:
        --threshold: 
    detector:
        --confidence_threshold: confidence threshold of face detection
        --nms_threshold: nms threshold of face detection
        --top_k: top k faces to be detected
        --keep_top_k: keep top k faces after nms
    face_gan:
        --in_size: input size of GPEN
        --out_size: output size of GPEN
        --key: key of GPEN
    bg_sr:
        --sr_model: super-resolution model
        --sr_scale: scale factor of super-resolution
        --tile_size: tile size of super-resolution
    face_parser:
        --
"""

if __name__ == '__main__':
    # 1. get args
    args = get_args()

    # 2. make output directory
    os.makedirs(args.outdir, exist_ok=True)

    # 3. load model
    processor = None
    if args.task == 'FaceEnhancement':
        processor = FaceEnhancement(args, in_size=args.in_size,
                                    model=args.model,
                                    use_sr=args.use_sr,
                                    device='cuda' if args.use_cuda else 'cpu')

    # 4. load images and process
    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)

        img = cv2.imread(file, cv2.IMREAD_COLOR)  # HWC-BGR-uint8
        if not isinstance(img, np.ndarray):  # error: not an image
            print(filename, 'error')
            continue
        # img = cv2.resize(img, (0,0), fx=2, fy=2) # optional
        # 5. enhance face
        img_out, orig_faces, enhanced_faces = processor.process(img, aligned=args.aligned)

        # 6. save results images -----
        img = cv2.resize(img, img_out.shape[:2][::-1])
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + f'_COMP{args.ext}'),
                    np.hstack((img, img_out)))
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + f'_GPEN{args.ext}'), img_out)

        if args.save_face:
            for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                of = cv2.resize(of, ef.shape[:2])
                cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1]) + '_face%02d' % m + args.ext),
                            np.hstack((of, ef)))

        if n % 10 == 0:
            print(n, filename)

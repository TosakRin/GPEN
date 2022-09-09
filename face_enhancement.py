"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import cv2
import numpy as np
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points


class FaceEnhancement(object):
    def __init__(self, args, base_dir='./', in_size=512, out_size=None, model=None, use_sr=True, device='cuda'):
        # ----- init models -----
        self.face_detector = RetinaFaceDetection(base_dir, device)
        self.face_gan = FaceGAN(base_dir, in_size, out_size, model, args.channel_multiplier, args.narrow, args.key,
                                device=device)
        self.bgsr_model = RealESRNet(base_dir, args.sr_model, args.sr_scale, args.tile_size, device=device)
        self.face_parser = FaceParse(base_dir, device=device)
        # -----------------------

        self.use_sr = use_sr  # use background super-resolution
        self.in_size = in_size  # input size of face enhancement model
        self.out_size = in_size if out_size is None else out_size  # output size of face enhancement model
        self.threshold = 0.9  # threshold of face detection
        self.alpha = args.alpha  # alpha of face enhancement

        # the kernel for smoothing small faces
        self.kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)  # (0, 0) or (0.1, 0.1)
        # reference_5pts (5, 2):[left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]
        self.reference_5pts = get_reference_facial_points(
            (self.in_size, self.in_size),
            inner_padding_factor,
            outer_padding,
            default_square)

    @staticmethod
    def mask_postprocess(mask, thres=26):
        """
        post-process the mask:

        Args:
            mask:
            thres:

        Returns:

        """
        mask[:thres, :] = 0  # top
        mask[-thres:, :] = 0  # bottom
        mask[:, :thres] = 0  # left
        mask[:, -thres:] = 0  # right
        mask = cv2.GaussianBlur(mask, (101, 101), 4)  # smooth mask edges
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def process(self, img, aligned=False):
        """

        Args:
            img:
            aligned:

        Returns:

        """
        orig_faces, enhanced_faces = [], []
        if aligned:
            ef = self.face_gan.process(img)
            orig_faces.append(img)
            enhanced_faces.append(ef)

            if self.use_sr:
                ef = self.bgsr_model.process(ef)

            return ef, orig_faces, enhanced_faces

        # ----- super-resolution background -----
        if self.use_sr:
            img_sr = self.bgsr_model.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])

        # ----- face detection -----
        # facebs (face_num, 5): [x1, y1, x2, y2, score]
        # landmarks (face_num, 10): [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
        facebs, landms = self.face_detector.detect(img)

        height, width = img.shape[:2]
        # full_mask: the mask for pasting restored faces back
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:  # skip low score faces
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            # facial5points: [[x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5]]
            facial5points = np.reshape(facial5points, (2, 5))

            # -------- get the cropped aligned face image and tfm --------
            # of: output face | tfm_inv: inverse transform matrix
            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts,
                                             crop_size=(self.in_size, self.in_size))

            # ------- enhance the face -------
            ef = self.face_gan.process(of)  # ef shape: (512, 512, 3)

            # ------- collect origin_face and enhanced_face for saving -------
            orig_faces.append(of)
            enhanced_faces.append(ef)

            # ------- parse face and get face mask -------
            """
            several varibles:
            - tmp_mask: the mask of parsed face, unaligned(affine transform back)) 
            - tmp_img: the image of enhanced face, unaligned(affine transform back))
            - full_mask: the mask for pasting restored faces back, size is the same as sr image
            - full_img: the image for pasting restored faces back, size is the same as sr image
            """
            tmp_mask = self.mask_postprocess(self.face_parser.process(ef)[0] / 255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)  #

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            # give ef a weight of self.alpha
            ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

            # in_size is
            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            # affine transform back to unaligned face
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)  # width, height is size of original image

            # ------- prepare sr size mask and image -------
            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        full_mask = full_mask[:, :, np.newaxis]  # (h, w, 1)

        # --------- paste the enhanced face back to the sr image -----------
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
        else:
            img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

        return img, orig_faces, enhanced_faces

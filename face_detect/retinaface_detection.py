"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from facemodels.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import torch.nn.functional as F


class RetinaFaceDetection(object):
    def __init__(self, base_dir, device='cuda', network='RetinaFace-R50'):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True

        self.pretrained_path = os.path.join(base_dir, 'weights', network + '.pth')
        self.device = device  # torch.cuda.current_device()
        self.cfg = cfg_re50  # detector cfg: RetinaFace-R50
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.load_model()
        self.net = self.net.to(device)
        self.mean = torch.tensor([[[[104]], [[117]], [[123]]]]).to(device)  # mean

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.net.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    @staticmethod
    def remove_prefix(state_dict, prefix):
        """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, load_to_cpu=False):
        # if load_to_cpu:
        #    pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage)
        # else:
        #    pretrained_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage.cuda())
        pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device('cpu'))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def detect(self, img_raw, resize=1, confidence_threshold=0.9, nms_threshold=0.4, top_k=5000, keep_top_k=750,
               save_image=False):
        """

        Args:
            img_raw:
            resize:
            confidence_threshold:
            nms_threshold:
            top_k:
            keep_top_k:
            save_image:

        Returns:

        """
        # --------- image processing --------
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        ss = 1.0
        # tricky: resize large image to less than 1000x1000
        if max(im_height, im_width) > 1500:  #
            ss = 1000.0 / max(im_height, im_width)
            img = cv2.resize(img, (0, 0), fx=ss, fy=ss)
            im_height, im_width = img.shape[:2]

        # shape[1] = height, shape[0] = width
        box_scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)  # mean
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).unsqueeze(0)  # add batch dim, HWC -> BCHW

        # detector input image: BCHW-BGR
        # ---------------------------------- #

        # ------- raw face detection using RetinaFace ------- #
        img = img.to(self.device)  # to GPU
        box_scale = box_scale.to(self.device)
        loc, conf, landms = self.net(img)  # forward pass
        del img

        # ------- post-processing ------- #
        # priorbox: # fixme: use priorbox to generate prior boxes
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        # get boxes, scores, landmarks
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * box_scale / resize  # scale to original image size
        boxes = boxes.cpu().numpy()     # boxes: [x1, y1, x2, y2]

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]   # scores: [score]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        landms_scale = torch.Tensor([im_width, im_height, im_width, im_height,
                                     im_width, im_height, im_width, im_height,
                                     im_width, im_height])
        landms_scale = landms_scale.to(self.device)
        landms = landms * landms_scale / resize     # scale to original image size
        landms = landms.cpu().numpy()   # landms: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]

        # ignore low scores face
        idx = np.where(scores > confidence_threshold)[0]
        boxes = boxes[idx]
        landms = landms[idx]
        scores = scores[idx]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)  # keep: [index]
        # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]     # dets shape: [keep_top_k, 5]
        landms = landms[:keep_top_k, :]     # landms shape: [keep_top_k, 10]

        # sort faces(delete)
        '''
        fscores = [det[4] for det in dets]
        sorted_idx = sorted(range(len(fscores)), key=lambda k:fscores[k], reverse=False) # sort index
        tmp = [landms[idx] for idx in sorted_idx]
        landms = np.asarray(tmp)
        '''

        # change landmarks from [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] to [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
        landms = landms.reshape((-1, 5, 2))     # landms shape: [keep_top_k, 5, 2]
        landms = landms.transpose((0, 2, 1))    # landms shape: [keep_top_k, 2, 5]
        landms = landms.reshape(-1, 10, )       # landms shape: [keep_top_k, 10]
        return dets / ss, landms / ss

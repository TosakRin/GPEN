#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project_name   ：GPEN 
# @file_name      : utils.py.py
# @Create_time    : 2022/9/9 009 PM 13:34
# @Author         : TosakRin
# @Email          : TosakRin@outlook.com
# @IDE            : PyCharm
# @Describe       : {描述内容}

import argparse


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned', action='store_true',
                        help='input are aligned faces or not')
    parser.add_argument('--alpha', type=float, default=1,
                        help='blending the results')
    parser.add_argument('--channel_multiplier', type=int, default=2,
                        help='channel multiplier of GPEN')
    parser.add_argument('--ext', type=str, default='.jpg',
                        help='extension of output')
    parser.add_argument('--indir', type=str, default='examples/imgs',
                        help='input folder')
    parser.add_argument('--in_size', type=int, default=512,
                        help='in resolution of GPEN')
    parser.add_argument('--key', type=str, default=None,
                        help='key of GPEN model')
    parser.add_argument('--model', type=str, default='GPEN-BFR-512',
                        help='GPEN model')
    parser.add_argument('--narrow', type=float, default=1,
                        help='channel narrow scale')
    parser.add_argument('--outdir', type=str, default='results/outs-BFR',
                        help='output folder')
    parser.add_argument('--out_size', type=int, default=None,
                        help='out resolution of GPEN')
    parser.add_argument('--save_face', action='store_true',
                        help='save face or not')
    parser.add_argument('--sr_model', type=str, default='realesrnet',
                        help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2,
                        help='SR scale')
    parser.add_argument('--task', type=str, default='FaceEnhancement',
                        help='task of GPEN model')
    parser.add_argument('--tile_size', type=int, default=0,
                        help='tile size for SR to avoid OOM')
    parser.add_argument('--use_sr', action='store_true',
                        help='use sr or not')
    parser.add_argument('--use_cuda', action='store_true',
                        help='use cuda or not')
    args = parser.parse_args()
    return args

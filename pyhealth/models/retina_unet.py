#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
NMS (Non-Maximum Suppression) functions for Retina U-Net.
Embedded here to remove external dependencies on cuda_functions.
"""

import numpy as np
import torch
import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.utils


############################################################
#  NMS Functions (from cuda_functions.nms_2D.python_nms)
############################################################

def gpu_nms(keep, num_out, boxes, nms_overlap_thresh):
    """
    Pure-Python GPU-compatible NMS fallback.
    - `boxes` is expected to be a tensor of shape (N, 5) and already sorted by score descending.
    - `keep` is a LongTensor (preallocated) where kept indices (into the sorted boxes) will be written.
    - `num_out` is a LongTensor of size 1 where the number of kept indices will be written.
    Returns 1 on success (match C extension behavior).
    """
    if boxes.numel() == 0:
        num_out[0] = 0
        return 1

    if boxes.is_cuda:
        boxes = boxes.cpu()
    boxes = boxes.contiguous().detach()

    x1 = boxes[:, 0].numpy()
    y1 = boxes[:, 1].numpy()
    x2 = boxes[:, 2].numpy()
    y2 = boxes[:, 3].numpy()

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    N = boxes.size(0)
    suppressed = np.zeros((N,), dtype=np.uint8)

    keep_list = []
    for i in range(N):
        if suppressed[i]:
            continue
        keep_list.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for j in range(i + 1, N):
            if suppressed[j]:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1.0)
            h = max(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_overlap_thresh:
                suppressed[j] = 1

    num = len(keep_list)
    for idx in range(num):
        keep[idx] = int(keep_list[idx])
    num_out[0] = num
    return 1


def cpu_nms(keep_out, num_out, boxes, order, areas, nms_overlap_thresh):
    """
    CPU NMS compatible with the original C signature.
    - `boxes` shape (N, 4)
    - `order` is a LongTensor of sorted indices (by score desc)
    - `areas` is a FloatTensor of areas per box
    Writes original box indices into `keep_out` and the count into `num_out[0]`.
    """
    if boxes.numel() == 0:
        num_out[0] = 0
        return 1

    if boxes.is_cuda:
        boxes = boxes.cpu()
    if order.is_cuda:
        order = order.cpu()
    if areas.is_cuda:
        areas = areas.cpu()

    boxes = boxes.contiguous().detach()
    order = order.contiguous().detach()
    areas = areas.contiguous().detach()

    boxes_np = boxes.numpy()
    order_np = order.numpy()
    areas_np = areas.numpy()

    N = boxes_np.shape[0]
    suppressed = np.zeros((N,), dtype=np.uint8)

    num_to_keep = 0
    for _i in range(N):
        i = int(order_np[_i])
        if suppressed[i]:
            continue
        keep_out[num_to_keep] = i
        num_to_keep += 1

        ix1 = boxes_np[i, 0]
        iy1 = boxes_np[i, 1]
        ix2 = boxes_np[i, 2]
        iy2 = boxes_np[i, 3]
        iarea = areas_np[i]

        for _j in range(_i + 1, N):
            j = int(order_np[_j])
            if suppressed[j]:
                continue
            xx1 = max(ix1, boxes_np[j, 0])
            yy1 = max(iy1, boxes_np[j, 1])
            xx2 = min(ix2, boxes_np[j, 2])
            yy2 = min(iy2, boxes_np[j, 3])
            w = max(0.0, xx2 - xx1 + 1.0)
            h = max(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            ovr = inter / (iarea + areas_np[j] - inter)
            if ovr >= nms_overlap_thresh:
                suppressed[j] = 1

    num_out[0] = num_to_keep
    return 1


def nms_gpu(dets, thresh):
    """High-level wrapper compatible with pth_nms.nms_gpu.
    Accepts `dets` of shape (N,5) and returns indices of kept boxes (tensor).
    """
    scores = dets[:, 4]
    order = scores.sort(0, descending=True)[1]
    dets_sorted = dets[order].contiguous()

    keep = torch.LongTensor(dets_sorted.size(0))
    num_out = torch.LongTensor(1)
    # call low-level implementation
    gpu_nms(keep, num_out, dets_sorted, thresh)

    # map kept indices back to original ordering
    if num_out[0] == 0:
        return torch.LongTensor([])
    kept = keep[:num_out[0]]
    return order[kept.cuda()].contiguous() if dets.is_cuda else order[kept].contiguous()


def nms_cpu(dets, thresh):
    """High-level wrapper compatible with pth_nms.nms_cpu.
    Returns `keep` tensor containing kept indices (in sorted-order reference).
    """
    dets = dets.cpu().detach()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]


# 2D NMS alias used by refine_detections.
nms_2D = nms_gpu


@dataclass
class RetinaUNetParams:
    """In-file parameter container so no external config file is required."""
    n_channels: int = 1
    start_filts: int = 18
    end_filts: int = 64
    res_architecture: str = "resnet50"
    norm: Optional[str] = None
    sixth_pooling: bool = False
    n_latent_dims: int = 0
    dim: int = 2
    head_classes: int = 2
    n_rpn_features: int = 64
    n_anchors_per_pos: int = 3
    rpn_anchor_stride: int = 1
    relu: str = "relu"
    rpn_anchor_ratios: tuple = (0.5, 1.0, 2.0)
    rpn_anchor_scales: Optional[Dict[str, Any]] = None
    backbone_shapes: Optional[Dict[str, Any]] = None
    backbone_strides: Optional[Dict[str, Any]] = None
    rpn_train_anchors_per_image: int = 6
    anchor_matching_iou: float = 0.7
    pre_nms_limit: int = 6000
    rpn_bbox_std_dev: tuple = (0.1, 0.1, 0.2, 0.2)
    scale: tuple = (1.0, 1.0, 1.0, 1.0)
    window: tuple = (0, 0, 511, 511)
    detection_nms_threshold: float = 0.1
    model_max_instances_per_batch_element: int = 100
    model_min_confidence: float = 0.0
    weight_init: Optional[str] = None
    patch_size: tuple = (512, 512)
    backbone_path: str = "models/backbone.py"
    operate_stride1: bool = False
    num_seg_classes: int = 2
    pyramid_levels: tuple = (0, 1, 2)
    backbone_class: Any = None

    @classmethod
    def from_kwargs(cls, **kwargs):
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**values)

    def __post_init__(self):
        if self.rpn_anchor_scales is None:
            self.rpn_anchor_scales = {
                'xy': [[8], [16], [32], [64]],
                'z': [[2], [2], [2], [2]],
            }

        if self.backbone_strides is None:
            self.backbone_strides = {
                'xy': [4, 8, 16, 32],
                'z': [1, 1, 1, 1],
            }

        if self.backbone_shapes is None:
            h, w = self.patch_size[:2]
            self.backbone_shapes = [
                np.array([max(1, int(np.ceil(float(h) / 4.0))), max(1, int(np.ceil(float(w) / 4.0)))]),
                np.array([max(1, int(np.ceil(float(h) / 8.0))), max(1, int(np.ceil(float(w) / 8.0)))]),
                np.array([max(1, int(np.ceil(float(h) / 16.0))), max(1, int(np.ceil(float(w) / 16.0)))]),
                np.array([max(1, int(np.ceil(float(h) / 32.0))), max(1, int(np.ceil(float(w) / 32.0)))]),
            ]


def compute_iou_2D(box, boxes, box_area, boxes_area):
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    return intersection / union


def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    return intersection / union


def compute_overlaps(boxes1, boxes2):
    if boxes1.shape[1] == 4:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            overlaps[:, i] = compute_iou_2D(boxes2[i], boxes1, area2[i], area1)
        return overlaps

    volume1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 4])
    volume2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 4])
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        overlaps[:, i] = compute_iou_3D(boxes2[i], boxes1, volume2[i], volume1)
    return overlaps


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    return np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)


def generate_pyramid_anchors(logger, cf):
    scales = cf.rpn_anchor_scales
    ratios = cf.rpn_anchor_ratios
    feature_shapes = cf.backbone_shapes
    anchor_stride = cf.rpn_anchor_stride
    pyramid_levels = cf.pyramid_levels
    feature_strides = cf.backbone_strides

    anchors = []
    logger.info("feature map shapes: {}".format(feature_shapes))
    logger.info("anchor scales: {}".format(scales))
    for level in pyramid_levels:
        anchors.append(generate_anchors(scales['xy'][level], ratios, feature_shapes[level], feature_strides['xy'][level], anchor_stride))
    return np.concatenate(anchors, axis=0)


def apply_box_deltas_2D(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return torch.stack([y1, x1, y2, x2], dim=1)


def apply_box_deltas_3D(boxes, deltas):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 4]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 4] + 0.5 * depth
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= torch.exp(deltas[:, 3])
    width *= torch.exp(deltas[:, 4])
    depth *= torch.exp(deltas[:, 5])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    return torch.stack([y1, x1, y2, x2, z1, z2], dim=1)


def clip_boxes_numpy(boxes, window):
    if boxes.shape[1] == 4:
        return np.concatenate(
            (
                np.clip(boxes[:, 0], 0, window[0])[:, None],
                np.clip(boxes[:, 1], 0, window[0])[:, None],
                np.clip(boxes[:, 2], 0, window[1])[:, None],
                np.clip(boxes[:, 3], 0, window[1])[:, None],
            ),
            1,
        )

    return np.concatenate(
        (
            np.clip(boxes[:, 0], 0, window[0])[:, None],
            np.clip(boxes[:, 1], 0, window[0])[:, None],
            np.clip(boxes[:, 2], 0, window[1])[:, None],
            np.clip(boxes[:, 3], 0, window[1])[:, None],
            np.clip(boxes[:, 4], 0, window[2])[:, None],
            np.clip(boxes[:, 5], 0, window[2])[:, None],
        ),
        1,
    )


def gt_anchor_matching(cf, anchors, gt_boxes, gt_class_ids=None):
    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2 * cf.dim))

    if gt_boxes is None:
        return np.full(anchor_class_matches.shape, fill_value=-1), anchor_delta_targets

    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    overlaps = compute_overlaps(anchors, gt_boxes)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]

    if anchors.shape[1] == 4:
        anchor_class_matches[(anchor_iou_max < 0.1)] = -1
    else:
        anchor_class_matches[(anchor_iou_max < 0.01)] = -1

    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    above_ix = np.argwhere(anchor_iou_max >= cf.anchor_matching_iou)
    anchor_class_matches[above_ix] = gt_class_ids[anchor_iou_argmax[above_ix]]

    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (cf.rpn_train_anchors_per_image // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        anchor_delta_targets[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        anchor_delta_targets[ix] /= cf.rpn_bbox_std_dev
        ix += 1

    return anchor_class_matches, anchor_delta_targets


def clip_to_window(window, boxes):
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))
    if boxes.shape[1] > 5:
        boxes[:, 4] = boxes[:, 4].clamp(float(window[4]), float(window[5]))
        boxes[:, 5] = boxes[:, 5].clamp(float(window[4]), float(window[5]))
    return boxes


def unique1d(tensor):
    if tensor.size()[0] <= 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = torch.tensor([True], dtype=torch.bool, device=tensor.device)
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool]


def shem(roi_probs_neg, negative_count, ohem_poolsize):
    _, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = min(ohem_poolsize * int(negative_count), order.size()[0])
    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0], device=pool_indices.device)
    return pool_indices[rand_idx[:negative_count]]


def initialize_weights(net):
    init_type = net.cf.weight_init
    for m in [
        module
        for module in net.modules()
        if type(module) in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear]
    ]:
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)


class NDConvGenerator(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                norm_layer = nn.InstanceNorm2d(c_out) if norm == 'instance_norm' else nn.BatchNorm2d(c_out)
                conv = nn.Sequential(conv, norm_layer)
        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                norm_layer = nn.InstanceNorm3d(c_out) if norm == 'instance_norm' else nn.BatchNorm3d(c_out)
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            relu_layer = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
            conv = nn.Sequential(conv, relu_layer)
        return conv


def get_one_hot_encoding(y, n_classes):
    dim = len(y.shape) - 2
    if dim == 2:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3]), dtype='int32')
    else:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3], y.shape[4]), dtype='int32')
    for cl in range(n_classes):
        y_ohe[:, cl][y[:, 0] == cl] = 1
    return y_ohe


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input


def batch_dice(pred, y, false_positive_weight=1.0, smooth=1e-6):
    if len(pred.size()) == 4:
        axes = (0, 2, 3)
    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
    else:
        raise ValueError('wrong input dimension in dice loss')

    intersect = sum_tensor(pred * y, axes, keepdim=False)
    denom = sum_tensor(false_positive_weight * pred + y, axes, keepdim=False)
    return torch.mean(((2 * intersect + smooth) / (denom + smooth))[1:])


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None) if downsample is not None else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class FPN(nn.Module):
    def __init__(self, cf, conv, operate_stride1=False):
        super(FPN, self).__init__()
        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling
        self.dim = conv.dim

        if operate_stride1:
            self.C0 = nn.Sequential(
                conv(cf.n_channels, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
                conv(start_filts, start_filts, ks=3, pad=1, norm=cf.norm, relu=cf.relu),
            )
            self.C1 = conv(start_filts, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)
        else:
            self.C1 = conv(cf.n_channels, start_filts, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm, relu=cf.relu)

        start_filts_exp = start_filts * self.block_expansion
        C2_layers = [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1),
            self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu, downsample=(start_filts, self.block_expansion, 1)),
        ]
        for _ in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = [self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp, 2, 2))]
        for _ in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = [self.block(start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2))]
        for _ in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = [self.block(start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2))]
        for _ in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = [self.block(start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 8, 2, 2))]
            for _ in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=cf.norm, relu=cf.relu))
            self.C6 = nn.Sequential(*C6_layers)

        self.P1_upsample = Interpolate(scale_factor=2 if conv.dim == 2 else (2, 2, 1), mode='bilinear' if conv.dim == 2 else 'trilinear')
        self.P2_upsample = Interpolate(scale_factor=2 if conv.dim == 2 else (2, 2, 1), mode='bilinear' if conv.dim == 2 else 'trilinear')

        self.out_channels = cf.end_filts
        self.P5_conv1 = conv(start_filts * 32 + cf.n_latent_dims, self.out_channels, ks=1, stride=1, relu=None)
        self.P4_conv1 = conv(start_filts * 16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts * 8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts * 4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

    def forward(self, x):
        c0_out = self.C0(x) if self.operate_stride1 else x
        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)

        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            out_list.append(self.P6_conv2(p6_pre_out))

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list


def nms_3D(dets, thresh):
    if dets.numel() == 0:
        return torch.LongTensor([]).to(dets.device) if dets.is_cuda else torch.LongTensor([])

    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    z1 = dets[:, 4]
    z2 = dets[:, 5]
    scores = dets[:, 6]

    volumes = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    order = scores.sort(0, descending=True)[1]
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[rest], x1[i])
        yy1 = torch.maximum(y1[rest], y1[i])
        zz1 = torch.maximum(z1[rest], z1[i])
        xx2 = torch.minimum(x2[rest], x2[i])
        yy2 = torch.minimum(y2[rest], y2[i])
        zz2 = torch.minimum(z2[rest], z2[i])

        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        d = torch.clamp(zz2 - zz1 + 1, min=0)
        inter = w * h * d
        iou = inter / (volumes[i] + volumes[rest] - inter)
        order = rest[iou <= thresh]

    out = torch.LongTensor(keep)
    return out.to(dets.device) if dets.is_cuda else out


# 3D NMS alias for refine_detections.
nms_3D = nms_3D


############################################################
#  Network Heads
############################################################

class Classifier(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = cf.head_classes
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * cf.head_classes
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]



class BBRegressor(nn.Module):


    def __init__(self, cf, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = cf.end_filts
        n_features = cf.n_rpn_features
        n_output_channels = cf.n_anchors_per_pos * self.dim * 2
        anchor_stride = cf.rpn_anchor_stride

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=cf.relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]


############################################################
#  Loss Functions
############################################################

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).to(anchor_matches.device)

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).to(anchor_matches.device))
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).to(anchor_matches.device)
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        device = target_deltas.device if isinstance(target_deltas, torch.Tensor) else torch.device('cpu')
        loss = torch.FloatTensor([0]).to(device)

    return loss


############################################################
#  Output Handler
############################################################

def refine_detections(anchors, probs, deltas, batch_ixs, cf):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    anchors = anchors.repeat(len(np.unique(batch_ixs)), 1)

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:cf.pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    keep_arr = keep_arr.long()
    pre_nms_scores = flat_probs[:cf.pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    device = probs.device if isinstance(probs, torch.Tensor) else torch.device('cpu')
    keep = torch.arange(pre_nms_scores.size()[0]).long().to(device)

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().to(device)
    scale = torch.from_numpy(np.asarray(cf.scale)).float().to(device)
    refined_rois = apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale \
        if cf.dim == 2 else apply_box_deltas_3D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = clip_to_window(cf.window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores

            if cf.dim == 2:
                class_keep = nms_2D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)
            else:
                class_keep = nms_3D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result



def get_results(cf, img_shape, detections, seg_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, cf.dim*2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
            class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
            scores = detections[ix][:, 2 * cf.dim + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if cf.dim == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= cf.model_min_confidence:
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]})

    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')

    return results_dict


############################################################
#  Retina (U-)Net Class
############################################################


class net(nn.Module):


    def __init__(self, cf=None, logger=None, **kwargs):

        super(net, self).__init__()
        if cf is None:
            self.cf = RetinaUNetParams.from_kwargs(**kwargs)
        else:
            self.cf = cf

        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        self.logger = logger

        self.build()
        if self.cf.weight_init is not None:
            logger.info("using pytorch weight init of type {}".format(self.cf.weight_init))
            initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")

    def build(self):
        """
        Build Retina Net architecture.
        """

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = NDConvGenerator(self.cf.dim)
        backbone_class = getattr(self.cf, 'backbone_class', None) or FPN

        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = generate_pyramid_anchors(self.logger, self.cf)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.anchors = torch.from_numpy(self.np_anchors).float().to(device)
        self.Fpn = backbone_class(self.cf, conv, operate_stride1=self.cf.operate_stride1)
        self.Classifier = Classifier(self.cf, conv)
        self.BBRegressor = BBRegressor(self.cf, conv)
        self.final_conv = conv(self.cf.end_filts, self.cf.num_seg_classes, ks=1, pad=0, norm=None, relu=None)


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        gt_class_ids = batch['roi_labels']
        gt_boxes = batch['bb_target']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        var_seg_ohe = torch.FloatTensor(get_one_hot_encoding(batch['seg'], self.cf.num_seg_classes)).to(device)
        var_seg = torch.LongTensor(batch['seg']).to(device)

        img = torch.from_numpy(img).float().to(device)
        batch_class_loss = torch.FloatTensor([0]).to(device)
        batch_bbox_loss = torch.FloatTensor([0]).to(device)

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas, seg_logits = self.forward(img)

        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['roi_labels'][b][ix], 'box_type': 'gt'})

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = gt_anchor_matching(
                    self.cf, self.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).to(device)
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().to(device)

            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            # add negative anchors used for loss to results_dict for monitoring.
            neg_anchors = clip_boxes_numpy(
                self.np_anchors[np.argwhere(anchor_class_match == -1)][0, neg_anchor_ix], img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]

        results_dict = get_results(self.cf, img.shape, detections, seg_logits, box_results_list)
        seg_loss_dice = 1 - batch_dice(F.softmax(seg_logits, dim=1), var_seg_ohe)
        seg_loss_ce = F.cross_entropy(seg_logits, var_seg[:, 0])
        loss = batch_class_loss + batch_bbox_loss + (seg_loss_dice + seg_loss_ce) / 2
        results_dict['torch_loss'] = loss
        results_dict['monitor_values'] = {'loss': loss.item(), 'class_loss': batch_class_loss.item()}
        results_dict['logger_string'] = \
            "loss: {0:.2f}, class: {1:.2f}, bbox: {2:.2f}, seg dice: {3:.3f}, seg ce: {4:.3f}, mean pix. pr.: {5:.5f}"\
            .format(loss.item(), batch_class_loss.item(), batch_bbox_loss.item(), seg_loss_dice.item(),
                    seg_loss_ce.item(), np.mean(results_dict['seg_preds']))

        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = torch.from_numpy(img).float().to(device)
        detections, _, _, seg_logits = self.forward(img)
        results_dict = get_results(self.cf, img.shape, detections, seg_logits)
        return results_dict


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.Fpn(img)
        seg_logits = self.final_conv(fpn_outs[0])
        selected_fmaps = [fpn_outs[i] for i in self.cf.pyramid_levels]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.Classifier(p))
            bb_reg_layer_outputs.append(self.BBRegressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

        # merge batch_dimension and store info in batch_ixs for re-allocation.
        device = class_logits.device
        batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).to(device)
        flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, self.cf)

        return detections, class_logits, bb_outputs, seg_logits


def smoke_test(batch_size=1, channels=1, height=512, width=512):
    """Run a minimal forward pass to verify this module works standalone."""
    model = net(
        n_channels=channels,
        patch_size=(height, width),
        num_seg_classes=2,
        head_classes=2,
        dim=2,
    )
    model.eval()

    dummy = np.random.rand(batch_size, channels, height, width).astype("float32")
    with torch.no_grad():
        out = model.test_forward({"data": dummy})

    print("Smoke test OK")
    print("boxes entries:", len(out["boxes"]))
    print("seg_preds shape:", out["seg_preds"].shape)


def train_smoke_test(batch_size=1, channels=1, height=512, width=512):
    """Run a minimal training forward pass to verify train path works standalone."""
    model = net(
        n_channels=channels,
        patch_size=(height, width),
        num_seg_classes=2,
        head_classes=2,
        dim=2,
    )
    model.train()

    data = np.random.rand(batch_size, channels, height, width).astype("float32")
    seg_h, seg_w = height // 4, width // 4
    seg = np.zeros((batch_size, 1, seg_h, seg_w), dtype="int64")

    # Add one simple foreground square per image for synthetic supervision.
    y1, x1, y2, x2 = height // 4, width // 4, height // 2, width // 2
    sy1, sx1, sy2, sx2 = seg_h // 4, seg_w // 4, seg_h // 2, seg_w // 2
    seg[:, 0, sy1:sy2, sx1:sx2] = 1

    bb_target = [np.array([[y1, x1, y2, x2]], dtype="float32") for _ in range(batch_size)]
    roi_labels = [np.array([1], dtype="int64") for _ in range(batch_size)]

    batch = {
        "data": data,
        "seg": seg,
        "bb_target": bb_target,
        "roi_labels": roi_labels,
    }

    out = model.train_forward(batch)
    print("Train smoke test OK")
    print("loss:", float(out["torch_loss"].item()))
    print("logger:", out["logger_string"])


def example_train_method(
    steps=5,
    batch_size=1,
    channels=1,
    height=512,
    width=512,
    lr=1e-4,
    data=None,
    seg=None,
    bb_target=None,
    roi_labels=None,
):
    """Example end-to-end training loop on dummy or user-provided arrays.

    Expected optional user-provided inputs:
    - data: np.ndarray [B, C, H, W], float32
    - seg: np.ndarray [B, 1, H/4, W/4] (or [B, 1, H, W], auto-resized), int64
    - bb_target: list length B, each element np.ndarray [N, 4] in y1, x1, y2, x2
    - roi_labels: list length B, each element np.ndarray [N] class ids
    """
    model = net(
        n_channels=channels,
        patch_size=(height, width),
        num_seg_classes=2,
        head_classes=2,
        dim=2,
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if data is None:
        data = np.random.rand(batch_size, channels, height, width).astype("float32")

    seg_h, seg_w = height // 4, width // 4
    if seg is None:
        seg = np.zeros((batch_size, 1, seg_h, seg_w), dtype="int64")
        sy1, sx1, sy2, sx2 = seg_h // 4, seg_w // 4, seg_h // 2, seg_w // 2
        seg[:, 0, sy1:sy2, sx1:sx2] = 1
    elif seg.shape[-2:] == (height, width):
        # If full-res labels are provided, resize to segmentation head resolution.
        seg_resized = []
        for b in range(seg.shape[0]):
            s = torch.from_numpy(seg[b:b + 1].astype("float32"))
            s = F.interpolate(s, size=(seg_h, seg_w), mode="nearest")
            seg_resized.append(s.numpy())
        seg = np.concatenate(seg_resized, axis=0).astype("int64")

    y1, x1, y2, x2 = height // 4, width // 4, height // 2, width // 2
    if bb_target is None:
        bb_target = [np.array([[y1, x1, y2, x2]], dtype="float32") for _ in range(batch_size)]
    if roi_labels is None:
        roi_labels = [np.array([1], dtype="int64") for _ in range(batch_size)]

    losses = []
    for step in range(steps):
        batch = {
            "data": data,
            "seg": seg,
            "bb_target": bb_target,
            "roi_labels": roi_labels,
        }

        optimizer.zero_grad(set_to_none=True)
        out = model.train_forward(batch)
        loss = out["torch_loss"]
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        print("step {}/{} | loss {:.6f}".format(step + 1, steps, loss_val))

    return {
        "model": model,
        "optimizer": optimizer,
        "losses": losses,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Retina U-Net module smoke test")
    parser.add_argument("--mode", choices=["test", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "test":
        smoke_test(
            batch_size=args.batch_size,
            channels=args.channels,
            height=args.height,
            width=args.width,
        )
    else:
        train_smoke_test(
            batch_size=args.batch_size,
            channels=args.channels,
            height=args.height,
            width=args.width,
        )
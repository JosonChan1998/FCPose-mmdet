# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import numpy as np
from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module
class FCPose(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCPose, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for
                each image in [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x,
                p^{K}_y, p^{K}_v] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints,
                              img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        bbox_inputs = outs + (img, img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            self.bbox_kpt2result(
                det_bboxes, det_labels, det_keypoints, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_keypoints in bbox_list
        ]
        return bbox_results
    
    def bbox_kpt2result(self, bboxes, labels, kpts, num_classes):
        """Convert detection results to a list of numpy arrays.
        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            kpts (torch.Tensor | np.ndarray): shape (n, K*3)
            num_classes (int): class number, including background class
        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)], \
                [np.zeros((0, kpts.size(1), 3), dtype=np.float32)
                    for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                kpts = kpts.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)], \
                   [kpts[labels == i, :, :] for i in range(num_classes)]
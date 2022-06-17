# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import result

import mmcv
import cv2
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_kpt=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_kpt = with_kpt
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results
    
    def _load_kpts(self, results):
        results['gt_keypoints'] = results['ann_info']['keypoints'].copy()
        results['gt_keypoints_ignore'] = results['ann_info']['keypoints_ignore'].copy()
        results['gt_areas'] = results['ann_info']['areas'].copy()

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_kpt:
            results = self._load_kpts(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations,
              self).__init__(with_bbox, with_label, with_mask, with_seg, True,
                             file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True
    """

    def __init__(self, min_gt_bbox_wh, keep_empty=True):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
               f'always_keep={self.always_keep})'


@PIPELINES.register_module()
class HeatmapGenerator:
    def __init__(
        self,
        num_keypoints,
        gt_heatmap_stride=2,
        aux_sigma=1.8,
        gt_p3_heatmap_stride=8,
        aux_p3_sigma=0.9,
        head_sigma=0.01):

        self.num_keypoints = num_keypoints
        self.gt_heatmap_stride = gt_heatmap_stride
        self.gt_p3_heatmap_stride = gt_p3_heatmap_stride

        # softmax heatmap: 2x down sampling aug_p1_g
        self.head_sigma = head_sigma
        size = 2 * np.round(3 * head_sigma) + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.head_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * head_sigma ** 2))

        # 2x down sampling
        self.aux_sigma = aux_sigma
        size = 2 * np.round(3 * aux_sigma) + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) / 2, (size - 1) / 2
        self.aux_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * aux_sigma ** 2))

        # 8x down sampling
        self.aux_p3_sigma = aux_p3_sigma
        size = 2 * np.round(3 * self.aux_p3_sigma) + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) / 2, (size - 1) / 2
        self.aug_p3_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * aux_p3_sigma ** 2))
    
    def show_skelenton(self, img, kpts, color=(255,128,128), thr=0.01):
        kpts = np.array(kpts).reshape(-1, 3)
        skelenton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                     [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                     [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                     [3, 5], [4, 6]]
        for sk in skelenton:
            pos1 = (int(np.round(kpts[sk[0], 0])), int(np.round(kpts[sk[0], 1])))
            pos2 = (int(np.round(kpts[sk[1], 0])), int(np.round(kpts[sk[1], 1])))
            if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0], 2] > thr and kpts[sk[1], 2] > thr:
                cv2.line(img, pos1, pos2, color, 2, 8)
                cv2.circle(img, pos1, radius=1, color=(0, 255, 0))
                cv2.circle(img, pos2, radius=1, color=(0, 255, 0))
        return img

    def __call__(self, results):
        gt_keypoints = results['gt_keypoints'].copy()
        if gt_keypoints.shape[0] == 0:
            return None

        aux_sigma = self.aux_sigma
        head_sigma = self.head_sigma
        aux_p3_sigma = self.aux_p3_sigma

        gt_keypoints[:, :, :2] = gt_keypoints[:, :, :2] / self.gt_heatmap_stride
        img_shape = results['pad_shape'][:2]
        h, w = [img_shape[0] / self.gt_heatmap_stride,
                img_shape[1] / self.gt_heatmap_stride]
        h, w = int(h), int(w)

        sem_heatmap_list = []
        inst_heatmaps_list = []
        kpt_labels_list = []
        for p in gt_keypoints:
            hms = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
            head_hms = np.zeros((self.num_keypoints, h, w), dtype=np.float32)
            kpt_label = np.zeros((self.num_keypoints, ), dtype=np.int64)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    kpt_label[idx] = 1
                    x, y = int(pt[0]), int(pt[1]) # use round better
                    if x < 0 or y < 0 or x >= w or y >= h:
                        continue
                    ul = int(np.round(x - 3 * aux_sigma - 1)), int(np.round(y - 3 * aux_sigma - 1))
                    br = int(np.round(x + 3 * aux_sigma + 2)), int(np.round(y + 3 * aux_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)

                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.aux_g[a:b, c:d])

                    ul = int(np.round(x - 3 * head_sigma - 1)), int(np.round(y - 3 * head_sigma - 1))
                    br = int(np.round(x + 3 * head_sigma + 2)), int(np.round(y + 3 * head_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    head_hms[idx, aa:bb, cc:dd] = np.maximum(
                        head_hms[idx, aa:bb, cc:dd], self.head_g[a:b, c:d])

            sem_heatmap_list.append(hms)
            inst_heatmaps_list.append(head_hms)
            kpt_labels_list.append(kpt_label)
        
        sem_heatmap = np.stack(sem_heatmap_list, axis=0).max(axis=0)
        inst_heatmaps = np.stack(inst_heatmaps_list, axis=0)
        kpt_labels = np.stack(kpt_labels_list, axis=0)

        # img = cv2.imread(results['filename'])
        # img = mmcv.imresize(
            # img, (w, h), return_scale=False, backend='cv2')
        # img = results['img'].transpose(2, 1, 0)
        # mean=[123.675, 116.28, 103.53]
        # std=[58.395, 57.12, 57.375]
        # img = (img * std + mean).astype(np.uint8)
        # img = mmcv.imresize(
        #     img, (w, h), return_scale=False, backend='cv2')
        # for pt in gt_keypoints:
        #     img = self.show_skelenton(img, pt)
        # cv2.imwrite("joints.jpg", img)

        p3_h, p3_w = [img_shape[0] / self.gt_p3_heatmap_stride,
                      img_shape[1] / self.gt_p3_heatmap_stride]
        p3_h, p3_w = int(p3_h), int(p3_w)
        p3_sem_heatmap_list = []
        keypoints = results['gt_keypoints'].copy()
        keypoints[:, :, :2] = keypoints[:, :, :2] / self.gt_p3_heatmap_stride
        for p in keypoints:
            p3_hms = np.zeros((self.num_keypoints, p3_h, p3_w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1]) # use round better
                    if x < 0 or y < 0 or \
                       x >= p3_w or y >= p3_h:
                        continue

                    ul = int(np.round(x - 3 * aux_p3_sigma - 1)), int(np.round(y - 3 * aux_p3_sigma - 1))
                    br = int(np.round(x + 3 * aux_p3_sigma + 2)), int(np.round(y + 3 * aux_p3_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], p3_w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], p3_h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], p3_w)
                    aa, bb = max(0, ul[1]), min(br[1], p3_h)
                    p3_hms[idx, aa:bb, cc:dd] = np.maximum(
                        p3_hms[idx, aa:bb, cc:dd], self.aug_p3_g[a:b, c:d])
            p3_sem_heatmap_list.append(p3_hms)
        p3_sem_heatmap = np.stack(p3_sem_heatmap_list, axis=0).max(axis=0)

        # img = cv2.imread(results['filename'])
        # for bbox in results['gt_bboxes']:
        #     x1, y1, x2, y2 = [int(i) for i in bbox]
        #     img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0))
        # img = mmcv.imresize(
        #     img, (p3_w, p3_h), return_scale=False, backend='cv2')
        # for pt in keypoints:
        #     img = self.show_skelenton(img, pt)
        # cv2.imwrite("p3_joints.jpg", img)
        
        mask = (results['gt_keypoints_ignore']*255).astype(np.uint8)
        mask = mmcv.imresize(
            mask, (w, h), return_scale=False, backend='cv2')
        mask = (mask / 255).astype(np.float32)

        p3_mask = (results['gt_keypoints_ignore']*255).astype(np.uint8)
        p3_mask = mmcv.imresize(
            p3_mask, (p3_w, p3_h), return_scale=False, backend='cv2')
        p3_mask = (p3_mask / 255).astype(np.float32)

        results['gt_p3_kpt_heatmap'] = p3_sem_heatmap
        results['gt_p3_kpt_ignore'] = p3_mask
        results['gt_kpt_heatmap'] = sem_heatmap
        results['gt_kpt_ignore'] = mask
        results['gt_inst_heatmaps'] = inst_heatmaps
        results['gt_kpt_labels'] = kpt_labels

        return results

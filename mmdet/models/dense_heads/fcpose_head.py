# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.core.post_processing import multiclass_nms
from mmcv.cnn import Scale, ConvModule, normal_init, bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


def reduce_sum(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class Folder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feature_map):
        N,_,H,W = feature_map.size()
        feature_map = F.unfold(feature_map,kernel_size=3,padding=1)
        feature_map = feature_map.view(N,-1,H,W)
        return feature_map


@HEADS.register_module()
class FCPoseHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 num_keypoints=17,
                 p3_hm_feat_stride=8,
                 p1_hm_feat_stride=2,
                 refine_levels=[0, 1, 2],
                 stacked_convs_share=4, 
                 feat_channels_share=128,
                 stacked_convs_kpt_head=3,
                 feat_channels_kpt_head=32,
                 stacked_convs_hm=2,
                 feat_channels_hm=128,
                 max_proposal_per_img=70,
                 with_hm_loss=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_mse=dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bn_norm_cfg=dict(type='SyncBN'),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        # fcpose head
        self.num_keypoints = num_keypoints
        self.p3_hm_feat_stride = p3_hm_feat_stride
        self.p1_hm_feat_stride = p1_hm_feat_stride
        self.refine_levels = refine_levels
        self.stacked_convs_share = stacked_convs_share
        self.feat_channels_share = feat_channels_share
        self.stacked_convs_kpt_head = stacked_convs_kpt_head
        self.feat_channels_kpt_head = feat_channels_kpt_head
        self.stacked_convs_hm = stacked_convs_hm
        self.feat_channels_hm = feat_channels_hm
        self.with_hm_loss = with_hm_loss
        self.max_proposal_per_img = max_proposal_per_img
        self.bn_norm_cfg = bn_norm_cfg

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_aux_mse = build_loss(loss_mse)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        # add refine convs
        self.refine_convs = nn.ModuleList()
        for level in self.refine_levels:
            self.refine_convs.append(
                ConvModule(
                    self.in_channels,
                    self.feat_channels_share,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.bn_norm_cfg,
                    bias=self.bn_norm_cfg is None))

        # add base convs
        self.shared_convs = nn.ModuleList()
        for i in range(self.stacked_convs_share):
            self.shared_convs.append(
                ConvModule(
                    self.feat_channels_share,
                    self.feat_channels_share,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.bn_norm_cfg,
                    bias=self.bn_norm_cfg is None))
        self.shared_convs.append(
            ConvModule(
                    self.feat_channels_share,
                    self.feat_channels_kpt_head + (2 * self.num_keypoints),
                    1,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None))

        # kpt fcn head params
        self.weight_nums = []
        self.bias_nums = []
        for i in range(self.stacked_convs_kpt_head):
            if i == 0:
                # for rel_coords
                self.weight_nums.append((self.feat_channels_kpt_head + 2) * self.feat_channels_kpt_head)
                self.bias_nums.append(self.feat_channels_kpt_head)
            elif i == self.stacked_convs_kpt_head - 1:
                self.weight_nums.append(self.feat_channels_kpt_head * self.num_keypoints)
                self.bias_nums.append(self.num_keypoints)
            else:
                self.weight_nums.append(self.feat_channels_kpt_head * self.feat_channels_kpt_head)
                self.bias_nums.append(self.feat_channels_kpt_head)
        self.total_params = 0
        self.total_params += sum(self.weight_nums)
        self.total_params += sum(self.bias_nums)

        # kpt head controller
        self.top_module = Folder()
        self.controller = nn.Linear(9 * self.feat_channels, self.total_params)
        self.kpt_upsampler = nn.Upsample(scale_factor=4, mode='bilinear')
        
        if self.with_hm_loss:
            self.hm_convs = nn.ModuleList()
            for i in range(self.stacked_convs_hm):
                chn = self.feat_channels_share if i == 0 else self.feat_channels_hm
                self.hm_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels_hm,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.bn_norm_cfg,
                        bias=self.bn_norm_cfg is None))
            self.p3_hm_pred = nn.Conv2d(
                self.feat_channels_hm, self.num_keypoints, 1)

            self.upsampler = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.feat_channels_hm + self.num_keypoints,
                    out_channels=self.feat_channels_hm,
                    kernel_size=8,
                    stride=4,
                    padding=6 // 2 - 1),
                nn.ReLU())
            self.p1_hm_pred = nn.Conv2d(
                self.feat_channels_hm, self.num_keypoints, kernel_size=3, padding=1)
    
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01, bias=0)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01, bias=0)
        for m in self.refine_convs:
            normal_init(m.conv, std=0.01)
        for m in self.shared_convs:
            normal_init(m.conv, std=0.01)
        for m in self.hm_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01, bias=0)
        normal_init(self.conv_centerness, std=0.01, bias=0)
        if self.with_hm_loss:
            torch.nn.init.constant_(self.p1_hm_pred.bias, 0.0)
            torch.nn.init.normal_(self.p1_hm_pred.weight, std=0.0001)
            torch.nn.init.constant_(self.p3_hm_pred.bias, 0.0)
            torch.nn.init.normal_(self.p3_hm_pred.weight, std=0.0001)

    def forward(self, feats):
        # refine convs
        for i, l in enumerate(self.refine_levels):
            if i == 0:
                x = self.refine_convs[i](feats[l])
            else:
                x_p = self.refine_convs[i](feats[l])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = self.aligned_bilinear(x_p, factor_h)
                x = x + x_p

        shared_feat = x
        hm_shared_feat = x

        # base convs
        for shared_layer in self.shared_convs:
            shared_feat = shared_layer(shared_feat)
        shared_feat, hm_offset_feat = torch.split_with_sizes(
            shared_feat, [self.feat_channels_kpt_head,  2*self.num_keypoints], dim=1)

        # aux mse preds
        if self.with_hm_loss:
            for hm_layer in self.hm_convs:
                hm_shared_feat = hm_layer(hm_shared_feat)
            p3_hm_feat = self.p3_hm_pred(hm_shared_feat)
            hm_shared_feat = torch.cat([hm_shared_feat, p3_hm_feat], dim=1)
            hm_shared_feat = self.upsampler(hm_shared_feat)
            p1_hm_feat = self.p1_hm_pred(hm_shared_feat)
        else:
            p3_hm_feat = None
            p1_hm_feat = None
        return multi_apply(self.forward_single, feats, self.scales, self.strides) + \
                          (shared_feat, hm_offset_feat, p1_hm_feat, p3_hm_feat)

    def forward_single(self, x, scale, stride):
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        controller = self.top_module(reg_feat).float()

        return cls_score, bbox_pred, centerness, controller

    def get_rel_coord_map(self,
                          feats,
                          instance_locations,
                          strides,
                          batch_inds):
        num_instance = len(instance_locations)
        H, W = feats.shape[2:]
        locations = self.compute_locations(H,
                                           W,
                                           stride=self.p3_hm_feat_stride,
                                           device=feats.device)
        relative_coordinates = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1))
        relative_coordinates = relative_coordinates.to(dtype=feats.dtype)
        pos_mask_feats = feats[batch_inds]
        coordinates_feat = torch.cat([
            relative_coordinates.view(num_instance, 2, H, W),
            pos_mask_feats], dim=1)
        coordinates_feat = coordinates_feat.view(1, -1, H, W)
        return coordinates_feat
    
    def compute_locations(self, h, w, stride, device):
        shifts_x = torch.arange(
            0,
            w * stride,
            step=stride,
            dtype=torch.float32,
            device=device)
        shifts_y = torch.arange(0,
            h * stride,
            step=stride,
            dtype=torch.float32,
            device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        num_instances = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(
            torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(
                    num_instances * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(
                    num_instances * self.num_keypoints, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances * self.num_keypoints)
        return weight_splits, bias_splits

    def kpt_heads_forward(self, features, weights, biases, num_instances):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                         w,
                         bias=b,
                         stride=1,
                         padding=0,
                         groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                          'controllers',))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             controllers,
             shared_feat,
             hm_offset_feat,
             p1_hm_feat,
             p3_hm_feat,
             gt_bboxes,
             gt_labels,
             gt_keypoints,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             **kwargs):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, ins_keypoints,\
        ins_bboxes, ins_inds, img_inds = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, gt_keypoints)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_controller = [
            controller.permute(0, 2, 3, 1).reshape(-1, 9 * self.feat_channels)
            for controller in controllers
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_controller = torch.cat(flatten_controller)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_ins_keypoints = torch.cat(ins_keypoints)
        flatten_ins_bboxes = torch.cat(ins_bboxes)
        flatten_ins_inds = torch.cat(ins_inds)
        flatten_img_inds = torch.cat(img_inds)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_strides = torch.cat(
            [points.new_tensor(stride)[None].repeat(num_imgs * points.shape[0])
            for stride, points in zip(self.strides, all_level_points)])
        flatten_coord_normalize = torch.cat(
            [points.new_tensor(2 ** i * 64).repeat(num_imgs * points.shape[0])
            for i, points in enumerate(all_level_points)])
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        # bbox & centerness target
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        # keypoint head forward
        # num_proposals = len(pos_inds)
        # if num_proposals > (num_imgs * self.max_proposal_per_img):
        #     with torch.no_grad():
        #         pos_cls_scores = flatten_cls_scores[pos_inds]
        #         scores = pos_cls_scores.sigmoid().reshape(-1, ) * pos_centerness.sigmoid().reshape(-1, )
        #         value, index = scores.topk(num_imgs * self.max_proposal_per_img, largest=True, sorted=True)
        #         index = index.sort()[0]
        #     pos_inds = pos_inds[index]
        
        filter_pos_inds = []
        ins_weights = []

        pos_ins_inds = flatten_ins_inds[pos_inds]
        pos_img_inds = flatten_img_inds[pos_inds]

        gt_num_inst = 0
        for bbox in gt_bboxes:
            gt_num_inst += bbox.shape[0]

        max_num_instances_per_gt = self.max_proposal_per_img // gt_num_inst
        max_num_instances_per_gt = max(max_num_instances_per_gt, 1)

        for img_id in range(num_imgs):
            inds_per_img = (pos_img_inds == img_id).bool()
            pos_ins_inds_per_img = pos_ins_inds[inds_per_img]
            if pos_ins_inds_per_img.shape[0] == 0:
                continue
            unique_gt_inds = pos_ins_inds_per_img.unique()

            for gt_ind in unique_gt_inds:
                inds_per_gt = (pos_ins_inds_per_img == gt_ind).bool()
                pos_ins_inds_per_img_per_inst = pos_ins_inds_per_img[inds_per_gt]
                if pos_ins_inds_per_img_per_inst.shape[0] > max_num_instances_per_gt:
                    ins_cls_scores = flatten_cls_scores[pos_inds][inds_per_img][inds_per_gt]
                    ins_centerness = pos_centerness[inds_per_img][inds_per_gt]
                    with torch.no_grad():
                        scores = ins_cls_scores.sigmoid().reshape(-1, ) *\
                                 ins_centerness.sigmoid().reshape(-1, )
                        value, index = scores.topk(
                            max_num_instances_per_gt, largest=True, sorted=True)
                    filter_pos_inds.append(pos_inds[inds_per_img][inds_per_gt][index])
                    ins_weights.append(unique_gt_inds.new_full(
                        [max_num_instances_per_gt], max_num_instances_per_gt))
                else:
                    filter_pos_inds.append(pos_inds[inds_per_img][inds_per_gt])
                    ins_weights.append(unique_gt_inds.new_full(
                        [len(pos_ins_inds_per_img_per_inst)],
                        len(pos_ins_inds_per_img_per_inst)))
        
        pos_inds = torch.cat(filter_pos_inds, dim=0)
        ins_weights = torch.cat(ins_weights, dim=0)

        pos_controller = flatten_controller[pos_inds]
        pos_points = flatten_points[pos_inds]
        pos_coord_normalize = flatten_coord_normalize[pos_inds]
        pos_coord_normalize = pos_coord_normalize.clamp(0, 1333)
        pos_img_inds = flatten_img_inds[pos_inds]
        pos_strides = flatten_strides[pos_inds]
        pos_ins_inds = flatten_ins_inds[pos_inds]
        pos_keypoints = flatten_ins_keypoints[pos_inds][:, :, :2]
        pos_bboxes = flatten_ins_bboxes[pos_inds]

        pos_controller = self.controller(pos_controller)
        if len(pos_inds) > 0:
            kpt_head_inputs = self.get_rel_coord_map(
                shared_feat,
                pos_points,
                pos_coord_normalize,
                pos_img_inds)
            
            weights, biases = self.parse_dynamic_params(
                pos_controller,
                self.feat_channels_kpt_head,
                self.weight_nums,
                self.bias_nums)
            
            kpt_logits = self.kpt_heads_forward(
                kpt_head_inputs,
                weights,
                biases,
                len(pos_inds))
            kpt_logits = kpt_logits.reshape(
                -1, self.num_keypoints, shared_feat[0].size(1), shared_feat[0].size(2))
            larger_kpt_logits = self.kpt_upsampler(kpt_logits)

            # get inst heatmap targets
            N, C, H, W = larger_kpt_logits.shape
            gt_inst_heatmaps = kwargs['gt_inst_heatmaps']
            gt_bitmasks_list = []
            for i in range(len(pos_inds)):
                cur_img_ind = pos_img_inds[i]
                cur_ins_ind = pos_ins_inds[i]
                kpt_target = gt_inst_heatmaps[cur_img_ind][cur_ins_ind]
                h, w = kpt_target.size()[1:]
                kpt_target = F.pad(kpt_target, (0, W - w, 0, H - h), "constant", 0)
                kpt_target = kpt_target.to(kpt_logits.device)
                gt_bitmasks_list.append(kpt_target)
            gt_bitmasks = torch.cat(gt_bitmasks_list, dim=0)
            gt_bitmasks = gt_bitmasks.reshape(N*C, H, W).bool()
            gt_bitmasks = gt_bitmasks.reshape(N*C, H*W)
            gt_bitmasks_visible_mask = gt_bitmasks.sum(dim=1).bool()

            # get inst keypoints and bboxes
            gt_box_x = pos_bboxes[:, 2] - pos_bboxes[:, 0]
            gt_box_y = pos_bboxes[:, 3] - pos_bboxes[:, 1]
            max_ranges = (gt_box_x + gt_box_y) / 2

            # ins weight
            # ins_weights = max_ranges.new_ones(max_ranges.shape)
            # ids = max_ranges.unique()
            # for id in ids:
            #     inds = (max_ranges == id).bool()
            #     ins_weights[inds] = inds.sum().float()
            ins_weights = 1 / ins_weights
            num_inst = ins_weights.sum()

            total_inst = reduce_sum(num_inst)
            gpu_balance_factor = num_inst / total_inst
            ins_weights = ins_weights[:, None].repeat(1, 17).flatten()[gt_bitmasks_visible_mask]
            max_ranges = max_ranges[:,None].repeat(1, 17).flatten()[gt_bitmasks_visible_mask]

            # get keypoints loss
            if gt_bitmasks_visible_mask.sum() != 0:
                larger_kpt_logits = larger_kpt_logits.reshape(N*C, H*W)
                larger_kpt_logits = larger_kpt_logits[gt_bitmasks_visible_mask]
                larger_kpt_logits = F.log_softmax(larger_kpt_logits, dim=1)

                gt_bitmasks = gt_bitmasks[gt_bitmasks_visible_mask]
                loss_keypoints = (-larger_kpt_logits[gt_bitmasks])
                loss_keypoints = (loss_keypoints * ins_weights * gpu_balance_factor).sum() / (self.num_keypoints * num_inst)
                loss_keypoints *= 2.5

                # get offset preds
                n_inst, C, H, W = kpt_logits.shape
                kpt_logits = kpt_logits.flatten(start_dim=2).softmax(dim=2).reshape(-1, 17, H, W)
                kpt_logits = kpt_logits.permute(0,2,3,1)[:,:,:,:,None]
                kpt_logits = kpt_logits.permute(0,3,1,2,4).reshape(n_inst*17,H,W,1)
                kpt_logits = kpt_logits[gt_bitmasks_visible_mask]

                N, C, H, W = hm_offset_feat.shape
                base_locations = self.compute_locations(
                    H, W, stride=self.p3_hm_feat_stride, device=hm_offset_feat.device)
                base_locations = base_locations.reshape(H, W, 2)
                base_locations = base_locations.permute(2, 0, 1)[None].repeat(N, 17, 1, 1) # N, 17*2, H ,W
                hm_offset_feat = hm_offset_feat + base_locations
                hm_offset_feat = hm_offset_feat[pos_img_inds]

                hm_offset_feat = hm_offset_feat[:,:,:,:,None].permute(0,2,3,4,1).reshape(n_inst,H,W,17,2)
                hm_offset_feat = hm_offset_feat - pos_keypoints[:, None, None, :, :]
                hm_offset_feat = hm_offset_feat.permute(0,3,1,2,4).reshape(n_inst*17,H,W,2)
                hm_offset_feat = hm_offset_feat[gt_bitmasks_visible_mask]
                hm_offset_feat = (hm_offset_feat[:,:,:,0] ** 2 + hm_offset_feat[:,:,:,1] ** 2).sqrt()[:,:,:,None]
                hm_offset_feat = hm_offset_feat / max_ranges[:,None,None,None]
                hm_offset_feat = hm_offset_feat * 12
                hm_offset_feat = (hm_offset_feat.sigmoid()-0.5) * 2
                
                hm_offset_feat = hm_offset_feat * kpt_logits
                hm_offset_feat = hm_offset_feat.flatten(start_dim=1).sum(dim=1)
                hm_offset_feat = hm_offset_feat * ins_weights
                loss_offset = (hm_offset_feat * gpu_balance_factor) / (num_inst * 12.0)
                loss_offset = loss_offset.sum()
                loss_offset *= 9.0
            else:
                loss_keypoints = larger_kpt_logits.sum() * 0.
                loss_offset = hm_offset_feat.sum() * 0. + kpt_logits.sum() * 0.
        else:
            loss_keypoints = pos_controller.sum() * 0.
            loss_offset = hm_offset_feat.sum() * 0.

        if self.with_hm_loss:
            gt_kpt_heatmap = kwargs['gt_kpt_heatmap']
            gt_kpt_ignore = kwargs['gt_kpt_ignore']
            gt_p3_kpt_heatmap = kwargs['gt_p3_kpt_heatmap']
            gt_p3_kpt_ignore = kwargs['gt_p3_kpt_ignore']

            num_dice = (gt_kpt_heatmap**2).sum()
            num_dice = max(reduce_mean(num_dice), 1.0)
            gt_kpt_ignore = gt_kpt_ignore.repeat(1, self.num_keypoints, 1, 1)
            loss_p1_mse = self.loss_aux_mse(
                p1_hm_feat, gt_kpt_heatmap, weight=gt_kpt_ignore, avg_factor=num_dice)
            p3_num_dice = (gt_p3_kpt_heatmap**2).sum()
            p3_num_dice = max(reduce_mean(p3_num_dice), 1.0)
            gt_p3_kpt_ignore = gt_p3_kpt_ignore.repeat(1, self.num_keypoints, 1, 1)
            loss_p3_mse = self.loss_aux_mse(
                p3_hm_feat, gt_p3_kpt_heatmap, weight=gt_p3_kpt_ignore, avg_factor=p3_num_dice)
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness,
                loss_keypoints=loss_keypoints,
                loss_offset=loss_offset,
                loss_p1_mse=loss_p1_mse,
                loss_p3_mse=loss_p3_mse)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_keypoints=loss_keypoints,
            loss_offset=loss_offset)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_keypoints_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list,\
        inst_keypoints_list, inst_bboxes_list, inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_keypoints_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        inst_keypoints_list = [
            inst_keypoints.split(num_points, 0) for inst_keypoints in inst_keypoints_list
        ]
        inst_bboxes_list = [
            inst_bboxes.split(num_points, 0) for inst_bboxes in inst_bboxes_list
        ]
        inds_list = [inds.split(num_points, 0) for inds in inds_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_inst_keypoints = []
        concat_lvl_inst_bboxes = []
        concat_lvl_inds = []
        concat_lvl_imgs = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_inst_keypoints.append(
                torch.cat([inst_keypoints[i] for inst_keypoints in inst_keypoints_list]))
            concat_lvl_inst_bboxes.append(
                torch.cat([inst_bboxes[i] for inst_bboxes in inst_bboxes_list]))
            concat_lvl_inds.append(torch.cat([inds[i] for inds in inds_list]))
            img_ind = []
            for j, labels in enumerate(labels_list):
                img_ind.extend([j for _ in range(labels[i].size(0))])
            img_ind = concat_lvl_inds[i].new_tensor(img_ind)
            concat_lvl_imgs.append(img_ind)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_inst_keypoints,\
               concat_lvl_inst_bboxes, concat_lvl_inds, concat_lvl_imgs

    def _get_target_single(self, gt_bboxes, gt_labels, gt_keypoints, points,
                           regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)),\
                   gt_labels.new_zeros(num_points,)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        inst_keypoints = gt_keypoints[min_area_inds]
        inst_bboxes = gt_bboxes[range(num_points), min_area_inds]

        return labels, bbox_targets, inst_keypoints, inst_bboxes, min_area_inds

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'controllers'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   controllers,
                   shared_feats,
                   hm_offset_feats,
                   p1_hm_feats,
                   p3_hm_feats,
                   imgs,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, cls_scores[0].dtype,
                                      cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            centerness_pred_list = select_single_mlvl(centernesses, img_id)
            controller_pred_list = select_single_mlvl(controllers, img_id)
            img = imgs[img_id]
            img_meta = img_metas[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            shared_feat = shared_feats[img_id][None]
            hm_offset_feat = hm_offset_feats[img_id][None]
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 controller_pred_list,
                                                 shared_feat,
                                                 hm_offset_feat,
                                                 mlvl_points, img, img_meta,
                                                 img_shape, scale_factor, cfg,
                                                 rescale)
            result_list.append(det_bboxes)
        return result_list
    
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           controller_pred_list,
                           shared_feats,
                           hm_offset_feats,
                           points_list,
                           img,
                           img_meta,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        assert len(cls_score_list) == len(bbox_pred_list)
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0.15)

        mlvl_scores = []
        mlvl_bboxes = []
        mlvl_labels = []
        mlvl_score_factors = []
        mlvl_controller = []
        mlvl_points = []
        mlvl_coord_normalizes = []

        for cls_score, bbox_pred, score_factor, controller, point, stride in zip(
            cls_score_list, bbox_pred_list, score_factor_list, controller_pred_list,
            points_list, self.strides):
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            controller = controller.permute(1, 2, 0).reshape(-1, 9 * self.feat_channels)
            coord_normalize = point.new_ones(point.size(0), 1) * stride * 8
            point = point.reshape(-1, 2)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred,
                     score_factor=score_factor,
                     controller=controller,
                     point=point,
                     coord_normalize=coord_normalize))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            score_factor = filtered_results['score_factor']
            controller = filtered_results['controller']
            coord_normalize = filtered_results['coord_normalize']
            point = filtered_results['point']

            bboxes = self.bbox_coder.decode(
                point, bbox_pred, max_shape=img_shape)

            mlvl_scores.append(scores)
            mlvl_bboxes.append(bboxes)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(score_factor)
            mlvl_controller.append(controller)
            mlvl_points.append(point)
            mlvl_coord_normalizes.append(coord_normalize)

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_score_factors = torch.cat(mlvl_score_factors)
        mlvl_controller = torch.cat(mlvl_controller)
        mlvl_points = torch.cat(mlvl_points)
        mlvl_coord_normalizes = torch.cat(mlvl_coord_normalizes)


        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        # nms process
        mlvl_scores = (mlvl_scores * mlvl_score_factors)[:, None]
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        det_bboxes, det_labels, nms_inds = multiclass_nms(mlvl_bboxes,
                                                          mlvl_scores,
                                                          score_thr,
                                                          cfg.nms,
                                                          cfg.max_per_img,
                                                          return_inds=True)
        det_controllers = mlvl_controller[nms_inds]
        det_points = mlvl_points[nms_inds]
        det_coord_normalizes = mlvl_coord_normalizes[nms_inds]
        det_coord_normalizes = det_coord_normalizes.clamp(0, 1333)
        det_img_inds = torch.zeros((det_points.shape[0]), dtype=torch.long)

        if det_bboxes.size(0) > 0:
            kpt_head_inputs = self.get_rel_coord_map(
                shared_feats,
                det_points,
                det_coord_normalizes,
                det_img_inds)
            
            det_controllers = self.controller(det_controllers)
            weights, biases = self.parse_dynamic_params(
                det_controllers,
                self.feat_channels_kpt_head,
                self.weight_nums,
                self.bias_nums)
            
            kpt_logits = self.kpt_heads_forward(
                kpt_head_inputs,
                weights,
                biases,
                len(det_img_inds))
            kpt_logits = kpt_logits.reshape(
                -1, self.num_keypoints, shared_feats[0].size(1), shared_feats[0].size(2))
            N, C, H, W = kpt_logits.shape
            kpt_logits = kpt_logits.permute(0, 2, 3, 1)[:, :, :, :, None]
            num_inst = len(det_img_inds)
            kpt_logits = kpt_logits.reshape(num_inst, H*W, 17).permute(0,2,1)
            kpt_logits = kpt_logits.reshape(num_inst*17, H*W)
            max_value, max_index = kpt_logits.max(dim = 1)
            arr = torch.arange(num_inst*17, device=kpt_logits.device)

            N, C, H, W = hm_offset_feats.shape
            base_locations = self.compute_locations(
                H, W, stride=self.p3_hm_feat_stride, device=hm_offset_feats.device)
            base_locations = base_locations.reshape(H, W, 2)
            base_locations = base_locations.permute(2, 0, 1)[None].repeat(N,17,1,1) # N, 17*2, H ,W
            hm_offset_feats = hm_offset_feats + base_locations
            hm_offset_feats = hm_offset_feats.repeat(num_inst, 1, 1, 1)
            hm_offset_feats = hm_offset_feats[:,:,:,:,None].permute(0,2,3,4,1).reshape(num_inst,H,W,17,2)
            hm_offset_feats = hm_offset_feats.permute(0,3,1,2,4).reshape(num_inst*17,H*W,2)
            hm_offset_feats = hm_offset_feats[arr,max_index]
            pred_keypoints = hm_offset_feats.reshape(num_inst, 17, 2)

            if rescale:
                pred_keypoints /= pred_keypoints.new_tensor(scale_factor[:2])
            vis = max_value.reshape(pred_keypoints.size(0),pred_keypoints.size(1),1)
            det_keypoints = torch.cat([pred_keypoints, vis], dim = 2)

        else:
            det_keypoints = torch.zeros((0, 17, 3), dtype=torch.float32)

        return det_bboxes, det_labels, det_keypoints

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
    
    def aligned_bilinear(self, tensor, factor):
        assert tensor.dim() == 4
        assert factor >= 1
        assert int(factor) == factor
        if factor == 1:
            return tensor

        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(tensor,
                            size=(oh, ow),
                            mode='bilinear',
                            align_corners=True)
        tensor = F.pad(tensor,
                    pad=(factor // 2, 0, factor // 2, 0),
                    mode="replicate")
        return tensor[:, :, :oh - 1, :ow - 1]

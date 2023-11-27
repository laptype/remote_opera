# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, bias_init_with_prob, constant_init, normal_init,
                      build_activation_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads import AnchorFreeHead

from opera.core.bbox import build_assigner, build_sampler
from opera.core.keypoint import gaussian_radius, draw_umich_gaussian
from opera.models.utils import build_positional_encoding, build_transformer
from opera.models.utils.smpl_utils import SMPL, rot6D_to_angular
from ..builder import HEADS, build_loss
      
@HEADS.register_module()
class PETRHead(AnchorFreeHead):
    """Head of `End-to-End Multi-Person Pose Estimation with Transformers`.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_kpt_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the keypoint regression head.
            Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): ConfigDict is used for
            building the Encoder and Decoder. Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_kpt (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_oks (obj:`mmcv.ConfigDict`|dict): Config of the
            regression oks loss. Default `OKSLoss`.
        loss_hm (obj:`mmcv.ConfigDict`|dict): Config of the
            regression heatmap loss. Default `NegLoss`.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        with_kpt_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to True.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_kpt_fcs=2,
                 num_keypoints=17,
                 smpl_path=None,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_kpt=dict(type='L1Loss', loss_weight=70.0),
                 as_two_stage=True,
                 with_kpt_refine=True,
                 train_cfg=dict(
                     assigner=dict(
                         type='PoseHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         kpt_cost=dict(type='KptL1Cost', weight=70.0),
                         oks_cost=dict(type='OksCost', weight=7.0))),
                 loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_joint=dict(type='L1Loss', loss_weight=70.0),
                 loss_joint_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_joint_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_pose=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_pose_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_pose_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_trans=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_trans_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_trans_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_shape=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_shape_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_shape_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                #  loss_prior=dict(type='opera.ParamLoss', path='/home/qianbo/git/hmr_best_bf/parameters', loss_weight=0.0),
                #  loss_prior_rpn=dict(type='opera.ParamLoss', path='/home/qianbo/git/hmr_best_bf/parameters', loss_weight=0.0),
                #  loss_prior_refine=dict(type='opera.ParamLoss', path='/home/qianbo/git/hmr_best_bf/parameters', loss_weight=0.0),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_kpt['loss_weight'] == assigner['kpt_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_kpt_fcs = num_kpt_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.with_kpt_refine = with_kpt_refine
        self.num_keypoints = num_keypoints
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')
        self.loss_cls = build_loss(loss_cls)
        self.loss_kpt = build_loss(loss_kpt)
        self.loss_kpt_rpn = build_loss(loss_kpt_rpn)
        self.loss_kpt_refine = build_loss(loss_kpt_refine)
        self.loss_joint = build_loss(loss_joint)
        self.loss_joint_rpn = build_loss(loss_joint_rpn)
        self.loss_joint_refine = build_loss(loss_joint_refine)
        self.loss_pose = build_loss(loss_pose)
        self.loss_pose_rpn = build_loss(loss_pose_rpn)
        self.loss_pose_refine = build_loss(loss_pose_refine)
        self.loss_trans = build_loss(loss_trans)
        self.loss_trans_rpn = build_loss(loss_trans_rpn)
        self.loss_trans_refine = build_loss(loss_trans_refine)
        self.loss_shape = build_loss(loss_shape)
        self.loss_shape_rpn = build_loss(loss_shape_rpn)
        self.loss_shape_refine = build_loss(loss_shape_refine)
        # self.loss_prior = build_loss(loss_prior)
        # self.loss_prior_rpn = build_loss(loss_prior_rpn)
        # self.loss_prior_refine = build_loss(loss_prior_refine)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.smpl = SMPL(smpl_path)
        self.embed_dims = self.transformer.embed_dims

        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and keypoint branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        kpt_branch = []
        kpt_branch.append(Linear(self.embed_dims, 512))
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            kpt_branch.append(Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(512, 3 * self.num_keypoints))
        kpt_branch = nn.Sequential(*kpt_branch)

        pose_branch = []
        pose_branch.append(Linear(self.embed_dims, 512))
        pose_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            pose_branch.append(Linear(512, 512))
            pose_branch.append(nn.ReLU())
        pose_branch.append(Linear(512, 22*6))
        pose_branch = nn.Sequential(*pose_branch)

        trans_branch = []
        trans_branch.append(Linear(self.embed_dims, 512))
        trans_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            trans_branch.append(Linear(512, 512))
            trans_branch.append(nn.ReLU())
        trans_branch.append(Linear(512, 3))
        trans_branch = nn.Sequential(*trans_branch)

        shape_branch = Linear(self.embed_dims, 10)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last kpt_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_kpt_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.kpt_branches = _get_clones(kpt_branch, num_pred)
            self.pose_branches = _get_clones(pose_branch, num_pred)
            self.trans_branches = _get_clones(trans_branch, num_pred)
            self.shape_branches = _get_clones(shape_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.kpt_branches = nn.ModuleList(
                [kpt_branch for _ in range(num_pred)])
            self.pose_branches = nn.ModuleList(
                [pose_branch for _ in range(num_pred)])
            self.trans_branches = nn.ModuleList(
                [trans_branch for _ in range(num_pred)])
            self.shape_branches = nn.ModuleList(
                [shape_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)

        refine_pose_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_pose_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_pose_branch.append(nn.ReLU())
        refine_pose_branch.append(Linear(self.embed_dims, 6))
        refine_pose_branch = nn.Sequential(*refine_pose_branch)

        refine_trans_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_trans_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_trans_branch.append(nn.ReLU())
        refine_trans_branch.append(Linear(self.embed_dims, 3))
        refine_trans_branch = nn.Sequential(*refine_trans_branch)

        refine_shape_branch = Linear(self.embed_dims, 10)

        if self.with_kpt_refine:
            num_pred = self.transformer.refine_decoder.num_layers
            self.refine_pose_branches = _get_clones(refine_pose_branch, num_pred)
            self.refine_shape_branches = _get_clones(refine_shape_branch, num_pred)
            self.refine_trans_branches = _get_clones(refine_trans_branch, num_pred)

    def init_weights(self):
        """Initialize weights of the PETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.kpt_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.pose_branches:
            constant_init(m[-1], 0, bias=0)
        # initialization of keypoint refinement branch
        if self.with_kpt_refine:
            for m in self.refine_pose_branches:
                constant_init(m[-1], 0, bias=0)
            for m in self.refine_trans_branches:
                constant_init(m[-1], 0, bias=0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            outputs_classes (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            outputs_kpts (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, K*2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, h*w, num_class). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
            enc_outputs_kpt (Tensor): The proposal generate from the
                encode feature map, has shape (N, h*w, K*2). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
        """

        # batch_size = mlvl_feats[0].size(0)

        query_embeds = self.query_embedding.weight
        """
            encoder --> 对应论文中的 csi Feature Encoder ？
            decoder --> 对应论文中的 Coarse Decoder
            lanbo: hs decoder的结果？
        """

        hs, init_reference_points, init_reference_poses, init_reference_trans, \
            inter_references_points, inter_references_poses, inter_references_trans, \
            enc_outputs_class, enc_outputs_kpt, enc_outputs_pose, enc_outputs_trans, enc_outputs_shape, memory = \
                self.transformer(
                    mlvl_feats,
                    query_embeds,
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    kpt_branches=self.kpt_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    pose_branches=self.pose_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    trans_branches=self.trans_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    shape_branches=self.shape_branches \
                        if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_kpts = []
        outputs_poses = []
        outputs_trans = []
        outputs_shapes = []
        # lanbo: hs.shape[0]: 猜 7
        """
            TODO: 
                hs 的维度
            lvl 是第几层的意思， hs.shape[0] = 7
        """
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference_points = init_reference_points
                reference_poses = init_reference_poses
                reference_trans = init_reference_trans
            else:
                reference_points = inter_references_points[lvl - 1]
                reference_poses = inter_references_poses[lvl - 1]
                reference_trans = inter_references_trans[lvl - 1]
            # reference = inverse_sigmoid(reference)
            """
                lanbo:
                self.cls_branches 是 Linear(self.embed_dims, self.cls_out_channels) 深拷贝了7份
            
            """
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp_kpt = self.kpt_branches[lvl](hs[lvl])
            tmp_pose = self.pose_branches[lvl](hs[lvl])
            tmp_trans = self.trans_branches[lvl](hs[lvl])
            # magnitude_test = torch.sqrt((tmp_pose**2).squeeze(0).sum(-1).mean())
            # print('pose magn.',lvl, ':', magnitude_test)
            """
                self.num_keypoints = 54
            """
            assert reference_points.shape[-1] == self.num_keypoints * 3
            """ 
                tmp_kpt: 当前层的 offsets predict by the dth layer.
                reference_points: 前一层的结果
            """
            tmp_kpt += reference_points
            tmp_pose += reference_poses
            tmp_trans += reference_trans
            outputs_shape = self.shape_branches[lvl](hs[lvl])
            outputs_kpt = tmp_kpt
            outputs_pose = tmp_pose
            outputs_tran = tmp_trans
            outputs_classes.append(outputs_class)
            outputs_kpts.append(outputs_kpt)
            outputs_poses.append(outputs_pose)
            outputs_trans.append(outputs_tran)
            outputs_shapes.append(outputs_shape)

        outputs_classes = torch.stack(outputs_classes)
        outputs_kpts = torch.stack(outputs_kpts)
        outputs_poses = torch.stack(outputs_poses)
        outputs_trans = torch.stack(outputs_trans)
        outputs_shapes = torch.stack(outputs_shapes)
        """
            hs[-1] 表示最后一层的输出特征
        """
        if self.as_two_stage:
            return outputs_classes, outputs_kpts, outputs_poses, outputs_trans, outputs_shapes, \
                enc_outputs_class, enc_outputs_kpt, enc_outputs_pose, enc_outputs_trans, enc_outputs_shape, memory, hs[-1]
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')

    def forward_refine(self, memory, human_queries, refine_targets, losses,
                       img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        """
        lanbo: 
            猜测： refine_targets是Groundtruth
        
        """
        """
        lanbo: 
            TODO:
                猜测：
                    3200 = 32 * 100 (32 是 batch size, 100是100个querys)
            SMPL 模型，每个人体模型有10个体型参数+24x3=72个姿态参数，我们用10+72=82个数就可以表示一个SMPL人体
            pose_targets: (3200, 72): 里面72是smpl里面的参数，表示72个关节点？
            shape_targets: (3200, 10): 
            trans_targets: (3200, 3): x, y, z ? 位移？
            valid_target: (3200) TODO: 

            human_queries: (32, 100, 256) -> (3200, 56)
        """
        beta_preds, pose_targets, shape_preds, shape_targets, trans_preds, trans_targets, valid_target, kpt_targets, kpt_weights = refine_targets
        # human_queries: (32, 100, 256) -> (3200, 56)
        human_queries = human_queries.reshape(-1, human_queries.shape[-1])
        pos_inds = valid_target > 0
        if pos_inds.sum() == 0:
            pos_pose_preds = torch.zeros_like(beta_preds[:1])
            trans_preds = torch.zeros_like(trans_preds[:1])
            pos_img_inds = beta_preds.new_zeros([1], dtype=torch.int64)
            human_queries = human_queries.new_zeros(human_queries[:1])
        else:
            """
                pos_pose_preds: (65, 132)  65 取决于valid_target中大于0的个数
            """
            pos_pose_preds = beta_preds[pos_inds]  
            trans_preds = trans_preds[pos_inds]
            pos_img_inds = (pos_inds.nonzero() / self.num_query).squeeze(1).to(
                torch.int64)
            human_queries = human_queries[pos_inds]
        """
            transformer.forward_refine：
                输入 
                memory:
                    前面encoder的结果: (180, 16, 256)   16 应该是batchsize
                human_queries:
                    前面 hs[-1]: 前面encoder+decoder的结果
                输出：
                hs：inter_states

        """
        hs, init_reference_pose, inter_references_pose,\
             init_reference_trans, inter_references_trans = self.transformer.forward_refine(
            memory,
            human_queries,
            pos_pose_preds.detach(),
            trans_preds.detach(),
            pos_img_inds,
            pose_branches=self.refine_pose_branches if self.with_kpt_refine else None, # noqa:E501
            trans_branches=self.refine_trans_branches if self.with_kpt_refine else None
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_poses = []
        outputs_shapes = []
        outputs_trans = []
        """
            hs.shape[0] 应该也是7层

            refine_pose_branches里面复制7个 Linear(self.embed_dims, 6) 输出是 6维
            refine_trans_branches里面复制7个Linear(self.embed_dims, 3) 输出是 3维
            refine_shape_branches里面复制7个Linear(self.embed_dims, 10) 输出是10维

        """
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference_pose = init_reference_pose
                reference_trans = init_reference_trans
            else:
                reference_pose = inter_references_pose[lvl - 1]
                reference_trans = inter_references_trans[lvl - 1]
            tmp_pose = self.refine_pose_branches[lvl](hs[lvl, :, 1:])
            tmp_trans = self.refine_trans_branches[lvl](hs[lvl, :, 0])
            tmp_pose += reference_pose
            tmp_trans += reference_trans
            outputs_pose = tmp_pose
            outputs_tran = tmp_trans
            outputs_shape = self.refine_shape_branches[lvl](hs[lvl, :, 0])
            outputs_poses.append(outputs_pose)
            outputs_trans.append(outputs_tran)
            outputs_shapes.append(outputs_shape)

        """
            这个就是最终模型的输出 
                outputs_poses       : 
                outputs_shapes
                outputs_trans
        """
        outputs_poses = torch.stack(outputs_poses)
        outputs_shapes = torch.stack(outputs_shapes)
        outputs_trans = torch.stack(outputs_trans)

        """
            [双人] outputs_poses: (2, 32, 22, 6)
                num_layer:  2
                num_gt:     32
                num_keypoints: 22
                dim:        6
            [单人] outputs_poses: (2, 16, 22, 6)   
        """

        num_layer, num_gt, num_keypoints, dim = outputs_poses.shape
        outputs_poses = outputs_poses.reshape(-1, num_keypoints*dim)

        global_orient_pred = outputs_poses[..., :6]
        body_pose_pred = outputs_poses[..., 6:]
        global_orient_pred = rot6D_to_angular(global_orient_pred)
        body_pose_pred = rot6D_to_angular(body_pose_pred)
        outputs_poses = torch.cat((global_orient_pred, body_pose_pred, torch.zeros(num_layer * num_gt, 6).to(outputs_poses.device)), 1)
        
        outputs_poses = outputs_poses.reshape(num_layer, num_gt, -1)

        """
            测试的输出结果 ----------------------------------------------------------------------------
            [单人]
                outputs_poses:  (2, 16, 72)
                outputs_shapes: (2, 16, 10)
                outputs_trans:  (2, 16, 3)
            [双人]
                outputs_poses:  (2, 32, 72)
                outputs_shapes: (2, 32, 10)
                outputs_trans:  (2, 32, 3)
        """
        if not self.training:
            return outputs_poses, outputs_shapes, outputs_trans
        
        """
            下面的应该都是在计算loss吧 -----------------------------------------------------------------
            [双人]
                pose_targets: (1600, 72) := 16 * 100, 72
                    在里面选了pos_inds: valid_target>0的下标
                    -> pose_targets: [x, 72] x是pos_inds里面ture的下标
                
        """
        # 获取groundtruth
        pose_targets = pose_targets[pos_inds]
        shape_targets = shape_targets[pos_inds]
        trans_targets = trans_targets[pos_inds]
        kpt_targets = kpt_targets[pos_inds]
        kpt_weights = kpt_weights[pos_inds]

        param_weights = torch.ones(num_gt).to(beta_preds.device)
        param_weights = param_weights.unsqueeze(-1)

        num_dec_layer = 0
        for pose_refine_preds, shape_preds in zip(outputs_poses, outputs_shapes):
            if pos_inds.sum() == 0:
                loss_pose = loss_shape = pose_refine_preds.sum() * 0
                losses[f'd{num_dec_layer}.loss_poses_refine'] = loss_pose
                losses[f'd{num_dec_layer}.loss_shapes_refine'] = loss_shape
                continue
            # kpt L1 Loss
            pose_weights = torch.ones((pose_refine_preds.shape[0], 24)).to(pose_refine_preds.device)
            num_valid_pose = torch.clamp(
                reduce_mean(pose_weights.sum()), min=1).item()
            pose_refine_preds = pose_refine_preds.reshape(
                pose_refine_preds.size(0), -1)
            loss_pose = self.loss_pose_refine(
                pose_refine_preds,
                pose_targets,
                pose_weights)
            
            
            losses[f'd{num_dec_layer}.loss_pose_refine'] = loss_pose
            shape_weights = torch.ones_like(shape_preds).to(pose_refine_preds.device)
            num_valid_shape = torch.clamp(
                reduce_mean(shape_weights.sum()), min=1).item()
            loss_shape = self.loss_shape_refine(shape_preds, shape_targets, shape_weights, avg_factor=num_valid_shape)
            losses[f'd{num_dec_layer}.loss_shape_refine'] = loss_shape

            _, joints_preds, _ = self.smpl(shape_preds, pose_refine_preds)
            joints_preds = joints_preds[:, :54]
            joints_preds = joints_preds.reshape(joints_preds.shape[0], -1)
            num_valid_kpt = torch.clamp(
                reduce_mean(kpt_weights.sum()), min=1).item()
            # assert num_valid_kpt == (kpt_targets>0).sum().item()
            loss_kpt = self.loss_joint_refine(
                joints_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)
            losses[f'd{num_dec_layer}.loss_kpt_refine'] = loss_kpt

            # lanbo shan
            # loss_prior = self.loss_prior_refine(pose_refine_preds, shape_preds, pose_weights)
            # losses[f'd{num_dec_layer}.loss_prior_refine'] = loss_prior

            trans_weights = param_weights.repeat(1, 3)
            num_valid_trans = torch.clamp(
                reduce_mean(trans_weights.sum()), min=1).item()
            trans_preds = trans_preds.reshape(-1, trans_preds.shape[-1])
            loss_trans = self.loss_trans(trans_preds, trans_targets, param_weights, avg_factor=num_valid_trans)
            losses[f'd{num_dec_layer}.loss_trans_refine'] = loss_trans

            num_dec_layer += 1

        return losses

    # over-write because img_metas are needed as inputs for bbox_head.
    """
        lanbo:
        模型的输入：
            x: (B, 180, 256)    wifi 数据
            [单人]
                gt_poses: list  B * (1, 72)   groundtruth pose 单人是 (1, 72)
                gt_shapes:      B * (1, 10)
                gt_keypoints:   B * (1, 54, 3)
                cam_trans:      B * (1, 3)
    """
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None, 
                      gt_poses=None, 
                      gt_shapes=None,
                      gt_keypoints=None,
                      gt_areas=None,
                      cam_trans=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_keypoints (list[Tensor]): Ground truth keypoints of the image,
                shape (num_gts, K*3).
            gt_areas (list[Tensor]): Ground truth mask areas of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        """
            human_queries: hs[-1]: 表示前面encoder+decoder最后一层输出的特征
            memory: 是前面encoder的输出

            outs: len = 12
            memory, human_queries = outs[-2:]
                memory:         (180, 16, 256): TODO 16是batchsize吗
                human_queries:  (16, 100, 256)


            outs = outs[:-2]: -> len = 10
        """
        memory, human_queries = outs[-2:]
        outs = outs[:-2]
        """
            loss_inputs = outs (10) + (8) = 18
        """
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, gt_poses, gt_shapes, gt_areas,
                                  cam_trans, img_metas)
        losses_and_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses, refine_targets = losses_and_targets
        # get pose refinement loss
        """
            得到最后的loss。训练时候最后一步输出
        """
        if self.with_kpt_refine:
            losses = self.forward_refine(memory, human_queries, refine_targets,
                                        losses, img_metas)
        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds', 'all_pose_preds', 'all_trans_preds', 'all_shape_preds'))
    def loss(self,
             all_cls_scores,
             all_kpt_preds,
             all_pose_preds,
             all_trans_preds,
             all_shape_preds,
             enc_cls_scores,
             enc_kpt_preds,
             enc_pose_preds,
             enc_trans_preds,
             enc_shape_preds,
             gt_bboxes_list,
             gt_labels_list,
             gt_keypoints_list,
             gt_poses_list,
             gt_shapes_list,
             gt_areas_list,
             gt_trans_list,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map, has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                    ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_keypoints_list = [
            gt_keypoints_list for _ in range(num_dec_layers)
        ]
        all_gt_poses_list = [
            gt_poses_list for _ in range(num_dec_layers)
        ]
        all_gt_trans_list = [
            gt_trans_list for _ in range(num_dec_layers)
        ]
        all_gt_shapes_lsit = [
            gt_shapes_list for _ in range(num_dec_layers)
        ]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_kpt, losses_pose, losses_trans, losses_shape, losses_joints, losses_prior, beta_preds_list, pose_targets_list, \
             shape_preds_list, shape_targets_list, trans_preds_list, trans_targets_list, kpt_preds_list, kpt_targets_list, kpt_weights_list, valid_target_list = multi_apply(
                self.loss_single, all_cls_scores, all_kpt_preds, all_pose_preds, all_trans_preds, all_shape_preds,
                all_gt_labels_list, all_gt_keypoints_list, all_gt_poses_list, 
                all_gt_shapes_lsit, all_gt_areas_list, all_gt_trans_list, img_metas_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt, enc_losses_pose, enc_losses_tran, enc_losses_shape, enc_losses_joint, enc_loss_prior = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, enc_pose_preds, enc_trans_preds, enc_shape_preds, binary_labels_list,
                    gt_keypoints_list, gt_poses_list, gt_shapes_list, gt_areas_list, gt_trans_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_kpt'] = enc_losses_kpt
            loss_dict['enc_loss_pose'] = enc_losses_pose
            loss_dict['enc_loss_tran'] = enc_losses_tran
            loss_dict['enc_loss_shape'] = enc_losses_shape
            loss_dict['enc_loss_joint'] = enc_losses_joint
            # lanbo shan
            # loss_dict['enc_loss_prior'] = enc_loss_prior

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        loss_dict['loss_pose'] = losses_pose[-1]
        loss_dict['loss_trans'] = losses_trans[-1]
        loss_dict['loss_shape'] = losses_shape[-1]
        loss_dict['losses_joints'] = losses_joints[-1]
        # lanbo shan
        # loss_dict['loss_prior'] = losses_prior[-1]
        # loss from other decoder layers
        num_dec_layer = 0

        # lanbo shan
        # for loss_cls_i, loss_kpt_i, loss_pose_i, loss_trans_i, loss_shape_i, loss_joints_i, loss_prior_i in zip(
        #         losses_cls[:-1], losses_kpt[:-1], losses_pose[:-1], losses_trans[:-1], losses_shape[:-1], losses_joints[:-1], losses_prior[:-1]):
        #     loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        #     loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
        #     loss_dict[f'd{num_dec_layer}.loss_pose'] = loss_pose_i
        #     loss_dict[f'd{num_dec_layer}.loss_trans'] = loss_trans_i
        #     loss_dict[f'd{num_dec_layer}.loss_shape'] = loss_shape_i
        #     loss_dict[f'd{num_dec_layer}.loss_joint'] = loss_joints_i
        #     loss_dict[f'd{num_dec_layer}.loss_prior'] = loss_prior_i
        #     num_dec_layer += 1
        for loss_cls_i, loss_kpt_i, loss_pose_i, loss_trans_i, loss_shape_i, loss_joints_i in zip(
                losses_cls[:-1], losses_kpt[:-1], losses_pose[:-1], losses_trans[:-1], losses_shape[:-1], losses_joints[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_pose'] = loss_pose_i
            loss_dict[f'd{num_dec_layer}.loss_trans'] = loss_trans_i
            loss_dict[f'd{num_dec_layer}.loss_shape'] = loss_shape_i
            loss_dict[f'd{num_dec_layer}.loss_joint'] = loss_joints_i
            num_dec_layer += 1

        return loss_dict, (beta_preds_list[-1], pose_targets_list[-1],
                        shape_preds_list[-1], shape_targets_list[-1], 
                        trans_preds_list[-1], trans_targets_list[-1],
                        valid_target_list[-1], kpt_targets_list[-1], kpt_weights_list[-1])

    def loss_single(self,
                    cls_scores,
                    kpt_preds,
                    pose_preds,
                    trans_preds,
                    shape_preds,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_poses_list,
                    gt_shapes_list,
                    gt_areas_list,
                    gt_trans_list,
                    img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs, num_query, _ = pose_preds.shape
        pose_preds = pose_preds.reshape(num_imgs * num_query, -1)
        beta_preds = pose_preds
        global_orient_pred = pose_preds[..., :6]
        body_pose_pred = pose_preds[..., 6:]
        global_orient_pred = rot6D_to_angular(global_orient_pred)
        body_pose_pred = rot6D_to_angular(body_pose_pred)
        pose_preds = torch.cat((global_orient_pred, body_pose_pred, torch.zeros(num_imgs * num_query, 6).to(pose_preds.device)), 1)
        _, joints_preds, _ = self.smpl(shape_preds.reshape(num_imgs * num_query, -1), pose_preds)
        joints_preds = joints_preds[:, :54]
        pose_preds = pose_preds.reshape(num_imgs, num_query, -1)
        joints_preds = joints_preds.reshape(num_imgs, num_query, -1)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        trans_preds_list = [trans_preds[i] for i in range(num_imgs)]
        pose_preds_list = [pose_preds[i] for i in range(num_imgs)]
        shape_preds_list = [shape_preds[i] for i in range(num_imgs)]
        joints_preds_list = [joints_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list, pose_preds_list, trans_preds_list, shape_preds_list, joints_preds_list,
                                           gt_labels_list, gt_keypoints_list, gt_poses_list, gt_trans_list, gt_shapes_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         pose_targets_list, trans_targets_list, shape_targets_list, param_weights_list, valid_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)
        pose_targets = torch.cat(pose_targets_list, 0)
        trans_targets = torch.cat(trans_targets_list, 0)
        shape_targets = torch.cat(shape_targets_list, 0)
        param_weights = torch.cat(param_weights_list, 0)
        valid_targets = torch.cat(valid_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)
        
        #smpl loss
        pose_weight = param_weights.repeat(1, 24)
        num_valid_pose = torch.clamp(
            reduce_mean(pose_weight.sum()), min=1).item()
        pose_preds = pose_preds.reshape(-1, pose_preds.shape[-1])    
        loss_pose = self.loss_pose(pose_preds, pose_targets, pose_weight)

        shape_weights = param_weights.repeat(1, 10)
        num_valid_shape = torch.clamp(
            reduce_mean(shape_weights.sum()), min=1).item()
        shape_preds = shape_preds.reshape(-1, shape_preds.shape[-1])
        loss_shape = self.loss_shape(shape_preds, shape_targets, param_weights, avg_factor=num_valid_shape)

        trans_weights = param_weights.repeat(1, 3)
        num_valid_trans = torch.clamp(
            reduce_mean(trans_weights.sum()), min=1).item()
        trans_preds = trans_preds.reshape(-1, trans_preds.shape[-1])
        loss_trans = self.loss_trans(trans_preds, trans_targets, param_weights, avg_factor=num_valid_trans)
        

        joints_preds = joints_preds.reshape(-1, joints_preds.shape[-1])
        loss_joints = self.loss_joint(
            joints_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        # lanbo shan
        # loss_prior = self.loss_prior(pose_preds, shape_preds, pose_weight)
        # return loss_cls, loss_kpt, loss_pose, loss_trans, loss_shape, loss_joints, \
        #     loss_prior, beta_preds, pose_targets, shape_preds, shape_targets, \
        #     trans_preds, trans_targets, kpt_preds, kpt_targets, kpt_weights, valid_targets

        return loss_cls, loss_kpt, loss_pose, loss_trans, loss_shape, loss_joints, \
               None, beta_preds, pose_targets, shape_preds, shape_targets, \
               trans_preds, trans_targets, kpt_preds, kpt_targets, kpt_weights, valid_targets

    def get_targets(self,
                    cls_scores_list,
                    kpt_preds_list,
                    pose_preds_list,
                    trans_preds_list,
                    shape_preds_list,
                    joint_preds_list,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_poses_list,
                    gt_trans_list,
                    gt_shape_list,
                    gt_areas_list,
                    img_metas):
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (x_{i}, y_{i}) and shape [num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3).
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all
                    images.
                - kpt_targets_list (list[Tensor]): Keypoint targets for all
                    images.
                - kpt_weights_list (list[Tensor]): Keypoint weights for all
                    images.
                - area_targets_list (list[Tensor]): area targets for all
                    images.
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         pose_targets_list, trans_targets_list, shape_targets_list, param_weights_list, valid_target_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, kpt_preds_list, pose_preds_list, trans_preds_list, shape_preds_list, joint_preds_list,
             gt_labels_list, gt_keypoints_list, gt_poses_list, gt_trans_list, gt_shape_list, gt_areas_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list, 
                pose_targets_list, trans_targets_list, shape_targets_list, param_weights_list,
                valid_target_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           kpt_pred,
                           pose_pred,
                           trans_pred,
                           shape_pred,
                           joints_pred,
                           gt_labels,
                           gt_keypoints,
                           gt_poses,
                           gt_trans,
                           gt_shapes,
                           gt_areas,
                           img_meta):
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (x_{i}, y_{i}) and
                shape [num_query, K*2].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_keypoints (Tensor): Ground truth keypoints for one image with
                shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas (Tensor): Ground truth mask areas for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - kpt_targets (Tensor): Keypoint targets of each image.
                - kpt_weights (Tensor): Keypoint weights of each image.
                - area_targets (Tensor): Area targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = kpt_pred.size(0)
       
        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, kpt_pred, pose_pred, trans_pred, shape_pred, joints_pred,
                                             gt_labels, gt_keypoints, gt_poses, gt_trans, gt_shapes, gt_areas, img_meta)
        # assign_result = self.assigner.assign(cls_score, kpt_pred, joints_pred, pose_pred, shape_pred,
        #                                      gt_labels, gt_keypoints, gt_poses, gt_shapes, gt_areas, img_meta)
        sampling_result = self.sampler.sample(assign_result, kpt_pred,
                                              gt_keypoints)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_labels.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_bboxes)

        # keypoint targets
        kpt_targets = torch.zeros_like(kpt_pred)
        kpt_weights = torch.zeros_like(kpt_pred)
        pos_gt_kpts = gt_keypoints[sampling_result.pos_assigned_gt_inds]
        valid_idx = pos_gt_kpts.new_ones((pos_gt_kpts.shape[0],
                                          pos_gt_kpts.shape[1]), dtype = torch.int64)
        valid_idx = valid_idx > 0
        pos_kpt_weights = kpt_weights[pos_inds].reshape(
            pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 3, 3)

        pos_kpt_weights[valid_idx] = 1.0
        kpt_weights[pos_inds] = pos_kpt_weights.reshape(
            pos_kpt_weights.shape[0], kpt_pred.shape[-1])

        pos_gt_kpts_normalized = pos_gt_kpts

        kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
            pos_gt_kpts.shape[0], kpt_pred.shape[-1])
        
        #smpl target
        pose_targets = torch.zeros_like(pose_pred)
        pos_gt_poses = gt_poses[sampling_result.pos_assigned_gt_inds]
        pose_targets[pos_inds] = pos_gt_poses

        trans_targets = torch.zeros_like(trans_pred)
        pos_gt_trans = gt_trans[sampling_result.pos_assigned_gt_inds]
        trans_targets[pos_inds] = pos_gt_trans

        shape_targets = torch.zeros_like(shape_pred)
        pos_gt_shapes = gt_shapes[sampling_result.pos_assigned_gt_inds]
        shape_targets[pos_inds] = pos_gt_shapes

        param_weights = torch.zeros(num_bboxes).to(pose_pred.device)
        param_weights[pos_inds] = 1.0
        param_weights = param_weights.unsqueeze(-1)

        #smpl kpt & smpl vert
        valid_targets = torch.zeros(num_bboxes, device=kpt_pred.device)
        valid_targets[pos_inds] = 1.0

        return (labels, label_weights, kpt_targets, kpt_weights,
                pose_targets, trans_targets, shape_targets, param_weights, 
                valid_targets, pos_inds, neg_inds)

    def loss_single_rpn(self,
                        cls_scores,
                        kpt_preds,
                        pose_preds,
                        trans_preds,
                        shape_preds,
                        gt_labels_list,
                        gt_keypoints_list,
                        gt_poses_list,
                        gt_shapes_list,
                        gt_areas_list,
                        gt_trans_list,
                        img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs, num_query, _ = pose_preds.shape
        pose_preds = pose_preds.reshape(num_imgs * num_query, -1)
        global_orient_pred = pose_preds[..., :6]
        body_pose_pred = pose_preds[..., 6:]
        global_orient_pred = rot6D_to_angular(global_orient_pred)
        body_pose_pred = rot6D_to_angular(body_pose_pred)
        pose_preds = torch.cat((global_orient_pred, body_pose_pred, torch.zeros(num_imgs * num_query, 6).to(pose_preds.device)), 1)
        _, joints_pred, _ = self.smpl(shape_preds.reshape(num_imgs * num_query, -1), pose_preds)
        joints_pred = joints_pred[:, :54]
        pose_preds = pose_preds.reshape(num_imgs, num_query, -1)
        joints_pred = joints_pred.reshape(num_imgs, num_query, -1)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        pose_preds_list = [pose_preds[i] for i in range(num_imgs)]
        trans_preds_list = [trans_preds[i] for i in range(num_imgs)]
        shape_preds_list = [shape_preds[i] for i in range(num_imgs)]
        joints_pred_list = [joints_pred[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list, pose_preds_list,
                                           trans_preds_list, shape_preds_list, joints_pred_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_poses_list,  gt_trans_list, gt_shapes_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list, 
         pose_targets_list, trans_targets_list, shape_targets_list, param_weights_list, valid_target_list,  num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)
        pose_targets = torch.cat(pose_targets_list, 0)
        trans_targets = torch.cat(trans_targets_list, 0)
        shape_targets = torch.cat(shape_targets_list, 0)
        joints_pred = torch.cat(joints_pred_list, 0)
        param_weights = torch.cat(param_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt_rpn(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)
        
        #smpl loss
        pose_weight = param_weights.repeat(1, 24)
        num_valid_pose = torch.clamp(
            reduce_mean(pose_weight.sum()), min=1).item()
        pose_preds = pose_preds.reshape(-1, pose_preds.shape[-1])
        loss_pose = self.loss_pose_rpn(pose_preds, pose_targets, pose_weight)

        trans_weights = param_weights.repeat(1, 3)
        num_valid_trans = torch.clamp(
            reduce_mean(trans_weights.sum()), min=1).item()
        trans_preds = trans_preds.reshape(-1, trans_preds.shape[-1])
        loss_trans = self.loss_trans(trans_preds, trans_targets, param_weights, avg_factor=num_valid_trans)
        
        shape_weights = param_weights.repeat(1, 10)
        num_valid_shape = torch.clamp(
            reduce_mean(shape_weights.sum()), min=1).item()
        shape_preds = shape_preds.reshape(-1, shape_preds.shape[-1])
        loss_shape = self.loss_shape_rpn(shape_preds, shape_targets, shape_weights, avg_factor=num_valid_shape)

        loss_joints = self.loss_joint_rpn(
            joints_pred, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        # lanbo shan
        # loss_prior = self.loss_prior_rpn(pose_preds, shape_preds, pose_weight)
        # return loss_cls, loss_kpt, loss_pose, loss_trans, loss_shape, loss_joints, loss_prior
        # print(f'loss_cls: {loss_cls.shape} loss_pose: {loss_pose} loss_joints: {loss_cls.shape}')
        # loss_prior = torch.zeros()
        return loss_cls, loss_kpt, loss_pose, loss_trans, loss_shape, loss_joints, None

#    outputs_classes, outputs_kpts, outputs_poses, outputs_trans, outputs_shapes, \
#    enc_outputs_class, enc_outputs_kpt, enc_outputs_pose, enc_outputs_trans, \
#     enc_outputs_shape, memory, hs[-1]
    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_kpt_preds,
                   all_pose_preds,
                   all_trans_preds,
                   all_shape_preds,
                   enc_cls_scores,
                   enc_kpt_preds,
                   enc_pose_preds,
                   enc_transpreds,
                   enc_shape_preds,
                   memory,
                   human_queries,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is an (n, K, 3) tensor
                with [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        cls_scores = all_cls_scores[-1]
        kpt_preds = all_kpt_preds[-1]
        pose_preds = all_pose_preds[-1]
        trans_preds = all_trans_preds[-1]
        shape_preds = all_shape_preds[-1]
        # cls_scores = enc_cls_scores
        # kpt_preds = enc_kpt_preds
        beta_preds = pose_preds
        gamma_preds = shape_preds
        num_imgs, num_query, _ = shape_preds.shape
        shape_preds = shape_preds.reshape(num_imgs*num_query, -1)
        pose_preds = pose_preds.reshape(num_imgs*num_query, -1)

        global_orient_pred = pose_preds[..., :6]
        body_pose_pred = pose_preds[..., 6:]
        global_orient_pred = rot6D_to_angular(global_orient_pred)
        body_pose_pred = rot6D_to_angular(body_pose_pred)
        pose_preds = torch.cat((global_orient_pred, body_pose_pred, torch.zeros(num_imgs * num_query, 6).to(pose_preds.device)), 1)
        verts_pred, joints_preds, _ = self.smpl(shape_preds, pose_preds)
        joints_preds = joints_preds[:, :54].reshape(num_imgs, num_query, -1)
        verts_pred = verts_pred.reshape(num_imgs, num_query, -1)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            kpt_pred = kpt_preds[img_id]
            beta_pred = beta_preds[img_id]
            trans_preds = trans_preds[img_id]
            gamma_pred = gamma_preds[img_id]
            joints_pred = joints_preds[img_id]
            verts_pred = verts_pred[img_id]
            human_queries = human_queries[img_id]
            # TODO: only support single image test
            # memory_i = memory[:, img_id, :]
            # mlvl_mask = mlvl_masks[img_id]
            proposals = self._get_bboxes_single(cls_score, kpt_pred, beta_pred, trans_preds, gamma_pred, joints_pred, verts_pred, memory, human_queries)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           kpt_pred,
                           beta_pred,
                           trans_preds,
                           gamma_pred,
                           joints_pred,
                           verts_pred,
                           memory,
                           human_queries):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (x_{i}, y_{i}) and
                shape [num_query, K*2].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],
                    where the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with
                    shape [num_query].
                - det_kpts: Predicted keypoints with shape [num_query, K, 3].
        """
        assert len(cls_score) == len(kpt_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            kpt_pred = kpt_pred[bbox_index]
            beta_pred = beta_pred[bbox_index]
            trans_preds = trans_preds[bbox_index]
            joints_pred = joints_pred[bbox_index]
            gamma_pred = gamma_pred[bbox_index]
            verts_pred = verts_pred[bbox_index]
            human_queries = human_queries[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            kpt_pred = kpt_pred[bbox_index]
            beta_pred = beta_pred[bbox_index]
            trans_preds = trans_preds[bbox_index]
            gamma_pred = gamma_pred[bbox_index]
            det_labels = det_labels[bbox_index]
            joints_pred = joints_pred[bbox_index]
            verts_pred = verts_pred[bbox_index]
            human_queries = human_queries[bbox_index]

        # ----- results after pose decoder -----
        det_kpts = kpt_pred.reshape(beta_pred.size(0), -1, 3)

        global_orient_pred = beta_pred[..., :6]
        body_pose_pred = beta_pred[..., 6:]
        global_orient_pred = rot6D_to_angular(global_orient_pred)
        body_pose_pred = rot6D_to_angular(body_pose_pred)
        det_poses = torch.cat((global_orient_pred, body_pose_pred, torch.zeros(beta_pred.shape[0], 6).to(beta_pred.device)), 1)

        det_shapes = gamma_pred
        det_joints = joints_pred.reshape(joints_pred.size(0), -1, 3)
        if self.with_kpt_refine:
            # ----- results after joint decoder (default) -----
            # import time
            # start = time.time()
            # beta_preds, pose_targets, shape_targets, valid_target, kpt_targets, kpt_weights = refine_targets
            refine_targets = (beta_pred, None, gamma_pred, None, trans_preds, None, torch.ones(beta_pred.shape[0]), None, torch.zeros_like(kpt_pred))
            # def forward_refine(self, memory, refine_targets, losses,
            #                img_metas):
            refine_outputs = self.forward_refine(memory, human_queries, refine_targets, None, None)
            # end = time.time()
            # print(f'refine time: {end - start:.6f}')
            det_poses, det_shapes, det_trans = refine_outputs
            det_poses = det_poses[-1]
            det_shapes = det_shapes[-1]
            det_trans = det_trans[-1]
            det_verts, det_joints, _ = self.smpl(det_shapes, det_poses)
            det_joints = det_joints[:, :54]

        # det_kpts[..., 0] = det_kpts[..., 0] * img_shape[1]
        # det_kpts[..., 1] = det_kpts[..., 1] * img_shape[0]
        # det_kpts[..., 0].clamp_(min=0, max=img_shape[1])
        # det_kpts[..., 1].clamp_(min=0, max=img_shape[0])
        # if rescale:
        #     det_kpts /= det_kpts.new_tensor(
        #         scale_factor[:2]).unsqueeze(0).unsqueeze(0)

        # use circumscribed rectangle box of keypoints as det bboxes
        x1 = det_kpts[..., 0].min(dim=1, keepdim=True)[0]
        y1 = det_kpts[..., 1].min(dim=1, keepdim=True)[0]
        x2 = det_kpts[..., 0].max(dim=1, keepdim=True)[0]
        y2 = det_kpts[..., 1].max(dim=1, keepdim=True)[0]
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        # det_kpts = torch.cat(
        #     (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        return det_bboxes, det_labels, det_poses, det_shapes, det_trans
    
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

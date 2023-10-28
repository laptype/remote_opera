_base_ = [
    '../_base_/datasets/wifi_mesh_wo_wdt.py', '../_base_/default_runtime.py'
]
model = dict(
    type='opera.PETR',
    amp_wdt=False,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict()),
    bbox_head=dict(
        type='opera.PETRHead',
        num_query=100,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        num_keypoints=54,
        # smpl_path=smpl_path,
        smpl_path="/home/qianbo/git/wifi/smpl_packed_info.pth",
        transformer=dict(
            type='opera.PETRTransformer',
            num_keypoints=54,
            generated_query=False,
            pe_mode='learnable',
            mask_ratio=0,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.PetrTransformerDecoder',
                num_layers=3,
                num_keypoints=54,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            refine_decoder=dict(
                type='opera.PetrRefineTransformerDecoder',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=10.0),
        loss_kpt=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_kpt_rpn=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_kpt_refine=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_joint=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_joint_rpn=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_joint_refine=dict(type='mmdet.MSELoss', loss_weight=100.0),
        loss_pose=dict(type='opera.PoseLoss', loss_weight=40.0),
        loss_pose_rpn=dict(type='opera.PoseLoss', loss_weight=40.0),
        loss_pose_refine=dict(type='opera.PoseLoss', loss_weight=40.0),
        loss_shape=dict(type='mmdet.MSELoss', loss_weight=3.0),
        loss_shape_rpn=dict(type='mmdet.MSELoss', loss_weight=3.0),
        loss_shape_refine=dict(type='mmdet.MSELoss', loss_weight=3.0)),
        
    train_cfg=dict(
        assigner=dict(
            type='opera.MeshHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=10.0),
            kpt_cost=dict(type='opera.KptMSECost', weight=100.0),
            pose_cost=dict(type='opera.PoseCost', weight=40.0),
            shape_cost=dict(type='opera.MSECost', weight=3.0),
            joints_cost=dict(type='opera.KptMSECost', weight=100.0))),
    test_cfg=dict(max_per_img=100))  # set 'max_per_img=20' for time counting
# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
find_unused_parameters = True
work_dir = '/home/qianbo/git/opera-version2/results/four_cards_loss_cls_10_wo_wdt_kaiming_init'
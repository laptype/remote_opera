# dataset settings
dataset_type = 'opera.WifiMeshDataset'
data_root = '/data2/qianbo/wifidataset/'
smpl_path='/home/qianbo/git/opera-version1/smpl_packed_info.pth'

train_pipeline = [
    dict(type='opera.DefaultFormatBundle',
         extra_keys=['gt_poses', 'gt_shapes', 'gt_keypoints', 'gt_labels']),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_poses', 'gt_shapes', 'gt_keypoints', 'gt_areas'],
         meta_keys=[]),
]

test_pipeline = [
    dict(
        type='mmdet.MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(type='mmdet.Collect',
                keys=['img'],
                meta_keys=[]),
        ])
]



data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=train_pipeline,
        mode='train',
        smpl_path=smpl_path),
    val=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='val',
        smpl_path=smpl_path),
    test=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='test_two_person',
        smpl_path=smpl_path))
evaluation = dict(interval=1, metric='mpjpe')

# dataset settings
dataset_type = 'opera.WifiMeshDataset'
# data_root = '/data2/qianbo/wifidataset/'

# data_root = '/home/wangpengcheng/WiMU/opera_test/'   # mat格式  长度20  wimu数据
# data_root = '/home/wangpengcheng/WiMU/opera_dataset/'   # mat格式 长度20 学长数据
data_root = '/home/wangpengcheng/WiMU/wifi_processed_data_20/'   # h5格式 长度20 wimu数据

smpl_path='/home/wangpengcheng/tmp/remote_opera/smpl_packed_info.pth'

train_pipeline = [
    dict(type='opera.DefaultFormatBundle',
         extra_keys=['gt_poses', 'gt_shapes', 'gt_keypoints', 'gt_labels', 'cam_trans']),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_poses', 'gt_shapes', 'gt_keypoints', 'gt_areas', 'cam_trans','imu'],
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

"""
    When model is :obj:`DistributedDataParallel`,
    batch size = samples_per_gpu
"""

data = dict(
    samples_per_gpu=16,
    # samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=train_pipeline,
        mode='train',
        smpl_path=smpl_path,
        amp_wdt=False),
    val=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='val',
        smpl_path=smpl_path,
        amp_wdt=False),
    test=dict(
        type=dataset_type,
        dataset_root=data_root,
        pipeline=test_pipeline,
        mode='test',
        smpl_path=smpl_path,
        amp_wdt=False))
evaluation = dict(interval=1, metric='mpjpe')

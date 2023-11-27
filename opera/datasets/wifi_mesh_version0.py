import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
from .builder import DATASETS
from mmdet.datasets.pipelines import Compose
from opera.models.utils.smpl_utils import SMPL, batch_compute_similarity_transform_torch
from . import constants
import h5py
import scipy.fft as fft
import pandas as pd
import warnings

@DATASETS.register_module()
class WifiMeshDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, pipeline, mode, smpl_path, phase_denoising=True, amp_wdt=True, **kwargs):
        '''
        振幅做小波变换
        相位做线性去噪
        '''
        self.data_root = dataset_root
        self.phase_denoising = phase_denoising
        self.amp_wdt = amp_wdt
        self.pipeline = Compose(pipeline)
        # self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_list.txt'))
        self.filename_list = self.load_csv_name_list(os.path.join(self.data_root, mode + '_list_1.csv'))
        self.All54_to_LSP14_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.LSP_14)
        # self.filename_list = self.filename_list[:10]
        self.smpl = SMPL(smpl_path)
        self._set_group_flag()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index): 
        # data_name = self.filename_list[index]       # txt

        csi_path = self.filename_list[index][0]       # csv
        annotation_path = self.filename_list[index][1]

        # csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        # csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.h5')) 
        # annotation_path = os.path.join(self.data_root,'annotations',(str(data_name)+'.npz'))
    
        warnings.filterwarnings("ignore", category=UserWarning)
        """
          用来忽略烦人的  UserWarning: Casting complex values to real discards the imaginary part
        """
        
        # try:
        #     csi = io.loadmat(csi_path)['csi_out']
        #     csi = np.array(csi)
        # except:
        #     csi = h5py.File(csi_path)['csi_out'][()]
        #     csi = csi['real'] + csi['imag']*1j
        #     csi = np.array(csi).transpose(3,2,1,0)
        #     csi = csi.astype(np.complex128)
        
        # '''csi_amp = abs(csi)
        # csi_amp = torch.FloatTensor(csi_amp).permute(0,1,3,2) #csi tensor: (3*3*30*20 -> 3*3*20*30)
        
        # csi_ph = np.unwrap(np.angle(csi))
        # csi_ph = fft.ifft(csi_ph)
        # csi_phd = csi_ph[:,:,:,1:20] - csi_ph[:,:,:,0:19]
        # csi_phd = torch.FloatTensor(csi_phd).permute(0,1,3,2)
        # '''
        # if self.amp_wdt:
        #     csi_amp = self.wdt(csi)
        # else:
        #     csi_amp = np.abs(csi)
        #     # csi_amp = torch.FloatTensor(csi_amp)
        
        # if self.phase_denoising:
        #     csi_ph = self.phase_deno(csi)
        # else:
        #     csi_ph = np.angle(csi)
        #     # csi_ph = np.unwrap(csi_ph)
        #     # csi_ph = fft.ifft(csi_ph)
        # csi = np.concatenate((csi_amp, csi_ph), axis=2)
        # csi = torch.FloatTensor(csi).permute(0,1,3,2)

        # print(csi.dtype)
        # TODO: load h5 文件
        def load_mat(path: os.path):
            result = {}
            with h5py.File(path, mode='r') as file:
                for key in file.keys():
                    result[key] = np.array(file[key])
            return result
        csi = load_mat(csi_path)
        """
            imu data:
                (12, 50, 6)
                (6, 50, 6)
        """     
        imu = csi['imu']    # load IMU data
  
        csi = csi['wifi_amp'] + csi['wifi_pha']*1j
        csi = csi.transpose(1,2,3,0)
        csi = csi.astype(np.complex128)
        # print(f"原始数据:{csi.shape}")

        if self.amp_wdt:
            csi_amp = self.wdt(csi)
        else:
            csi_amp = abs(csi)
            csi_amp = torch.FloatTensor(csi_amp)
        
        if self.phase_denoising:
            csi_ph = self.phase_deno(csi)
        else:
            csi_ph = np.angle(csi)
            # csi_ph = np.unwrap(csi_ph)
            # csi_ph = fft.ifft(csi_ph)
        # print(f"ph数据:{csi_ph.shape}")
        # print(f"am数据:{csi_amp.shape}")
        csi = np.concatenate((csi_amp, csi_ph), axis=2)
        csi = torch.FloatTensor(csi).permute(0,1,3,2)
        # print(f"处理数据数据:{csi.shape}")



        annotation = np.load(annotation_path, allow_pickle=True)['results'].item()
        # 'cam', 'global_orient', 'body_pose', 'smpl_betas', 'smpl_thetas', 
        # 'center_preds', 'center_confs', 'cam_trans', 'verts', 'joints', 'pj2d_org'
        """
            lanbo:
                [双人]
                    pose:       [2, 72]
                    shape:      (2, 10)
                    keypoint:   (2, 54, 3)
                    cam_trans:  (2, 3)
                [单人]
                    pose:       [1, 72]
                    shape:      (1, 10)
                    keypoint:   (1, 54, 3)
                    cam_trans:  (1, 3)
        """
        pose = annotation["smpl_thetas"]
        shape = annotation["smpl_betas"]
        keypoint = annotation["joints"][:,:54]
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*71=(54+17)*3)
        verts = torch.FloatTensor(annotation['verts'])
        cam_trans = torch.FloatTensor(annotation['cam_trans'])

        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        gt_bboxes = torch.tensor([])
        gt_areas = torch.tensor([])
        result = dict(img=csi, gt_poses=pose, gt_shapes=shape, gt_keypoints=keypoint, \
                      gt_verts=verts, gt_labels = gt_labels, gt_bboxes = gt_bboxes, \
                      gt_areas = gt_areas, cam_trans = cam_trans)
        return result
    
    def __getitem__(self, index):
        result = self.get_item_single_frame(index)
        return self.pipeline(result)

    def __len__(self):
        return len(self.filename_list)

    def load_csv_name_list(self, file_path):
        # return pd.read_csv(file_path, header=None).iloc
        return pd.read_csv(file_path, header=None).values.tolist()


    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                file_name_list.append(lines.split()[0])
        return file_name_list

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    def CSI_sanitization(self, csi_rx):
        one_csi = csi_rx[0,:,:]
        two_csi = csi_rx[1,:,:]
        three_csi = csi_rx[2,:,:]
        pi = np.pi
        M = 3  # 天线数量3
        N = 30  # 子载波数目30
        T = one_csi.shape[1]  # 总包数
        fi = 312.5 * 2  # 子载波间隔312.5 * 2
        csi_phase = np.zeros((M, N, T))
        for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
            csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
            csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
            csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
            ai = np.tile(2 * pi * fi * np.array(range(N)), M)
            bi = np.ones(M * N)
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            rho_opt = (B * E - C * D) / (A * C - B ** 2)
            beta_opt = (B * D - A * E) / (A * C - B ** 2)
            temp = np.tile(np.array(range(N)), M).reshape(M, N)
            csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0), 
                                      np.expand_dims(antennaPair_Two,axis=0), 
                                      np.expand_dims(antennaPair_Three,axis=0),))
        return antennaPair


    def phase_deno(self, csi):
        #input csi shape (3*3*30*20)
        ph_rx1 = self.CSI_sanitization(csi[0,:,:,:])
        ph_rx2 = self.CSI_sanitization(csi[1,:,:,:])
        ph_rx3 = self.CSI_sanitization(csi[2,:,:,:])
        csi_phde = np.concatenate((np.expand_dims(ph_rx1,axis=0), 
                                   np.expand_dims(ph_rx2,axis=0), 
                                   np.expand_dims(ph_rx3,axis=0),))
        return csi_phde
    
    def wdt(self, csi):
        cA, cD = pywt.dwt(abs(csi), 'db11')
        csi_amp = np.concatenate((cA, cD), axis=2)
        return csi_amp
        

    def evaluate(self,
                 results,
                 metric='keypoints',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        mpjpe_3d_list = []
        mpjpe_smpl_list = []
        mpjpe_smpl_dp_list = []
        mpjpe_smpl_lsp_list = []
        pampjpe_lsp_list = []
        pve_list = []
        trans_error_list = []
        root = self.data_root
        for i in range(len(results)):
            # info = self.get_item_single_frame(i)
            # gt_keypoints = info['gt_keypoints']
            gt_keypoints = self.get_item_single_frame(i)['gt_keypoints']
            gt_verts = self.get_item_single_frame(i)['gt_verts']
            cam_trans = self.get_item_single_frame(i)['cam_trans']
            _, det_poses, det_shapes, det_trans = results[i]
            del _
            for label in range(len(det_poses)):
                poses_pred = det_poses[label]
                shapes_pred = det_shapes[label]
                trans_pred = det_trans[label]
                verts_pred, joints_pred, _ = self.smpl(shapes_pred, poses_pred)

                # joints_pred = det_joints[label]
                # joints_pred = torch.tensor(joints_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
                # mpjpe_joints, corres = self.calc_mpjpe(gt_keypoints, joints_pred)
                # mpjpe_smpl_list.append(mpjpe_joints.numpy())

                joints_pred_lsp = torch.tensor(joints_pred[:,self.All54_to_LSP14_mapper], dtype = joints_pred.dtype)
                gt_keypoints_lsp = torch.tensor(gt_keypoints[:,self.All54_to_LSP14_mapper], dtype = gt_keypoints.dtype)
                mpjpe_joints_lsp, corres = self.calc_mpjpe(gt_keypoints_lsp, joints_pred_lsp)
                mpjpe_smpl_lsp_list.append(mpjpe_joints_lsp.numpy())

                pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(joints_pred_lsp[corres], gt_keypoints_lsp)
                pampjpe_lsp, _ = self.calc_mpjpe(gt_keypoints_lsp, pred_tranformed)
                pampjpe_lsp_list.append(pampjpe_lsp)

                verts_pred = verts_pred[corres]
                PVE = torch.sqrt(((gt_verts - verts_pred)**2).sum(-1)).mean()*1000
                pve_list.append(PVE.numpy())

                trans_pred = trans_pred[corres]
                trans_error = torch.sqrt(((cam_trans - trans_pred)**2).sum(-1)).mean()*1000
                trans_error_list.append(trans_error)

                # frame_name = self.filename_list[i]
                # frame_verts_path = os.path.join(root, 'verts')
                # if not os.path.exists(frame_verts_path):
                #     os.mkdir(frame_verts_path)
                # np.save(os.path.join(frame_verts_path, frame_name + ".npy"), verts_pred)

                # frame_trans_path = os.path.join(root, 'trans')
                # if not os.path.exists(frame_trans_path):
                #     os.mkdir(frame_trans_path)
                # np.save(os.path.join(frame_trans_path, frame_name + ".npy"), cam_trans)
                # print("test")

        mpjpe_3d = np.array(mpjpe_3d_list).mean()   
        mpjpe_smpl = np.array(mpjpe_smpl_list).mean() 
        mpjpe_smpl_pd = np.array(mpjpe_smpl_dp_list).mean()  
        mpjpe_lsp = np.array(mpjpe_smpl_lsp_list).mean()   
        pampjpe_lsp = np.array(pampjpe_lsp_list).mean()
        pve = np.array(pve_list).mean()
        trans_error = np.array(trans_error_list).mean()
        result = {"mpjpe":mpjpe_lsp, "pampjpe":pampjpe_lsp,  "pve":pve, "trans_error":trans_error}

        return OrderedDict(result)
    
    def calc_mpjpe(self, real, pred):
        n = real.shape[0]
        m = pred.shape[0]
        j, c = pred.shape[1:]
        assert j == real.shape[1] and c == real.shape[2]
        #n*m*j  n*j
        distance_array = torch.ones((n,m), dtype=torch.float) * 2 ** 24  # TODO: magic number!
        for i in range(n):
            for j in range(m):
                distance_array[i][j] = torch.norm(real[i]-pred[j], p=2, dim=-1).mean()

        corres = torch.ones(n, dtype=torch.long)*-1
        occupied = torch.zeros(m, dtype=torch.long)

        while torch.min(distance_array) < 50:   # threshold 30.
            min_idx = torch.where(distance_array == torch.min(distance_array))
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1

        new_pred = pred[corres]
        mpjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1))

        return mpjpe.mean()*1000, corres


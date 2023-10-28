import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
from .builder import DATASETS
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module()
class WifiMeshDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, pipeline, mode, **kwargs):
        '''
        振幅做小波变换
        相位做线性去噪
        '''
        self.data_root = dataset_root
        self.pipeline = Compose(pipeline)
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, 'train_list.txt'))
        if mode == "val":
            self.filename_list = self.filename_list[::50]
        self._set_group_flag()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index): 
        data_name = self.filename_list[index]
        csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        annotation_path = os.path.join(self.data_root,'annotations',(str(data_name)+'.npz'))
        
        csi =  io.loadmat(csi_path)['csi_out']
        csi = np.array(csi)
        csi = csi.astype(np.complex128)

        group_name = data_name.split('_')[0]
        csi_avg_path = os.path.join(self.data_root,'avg_csi',(str(group_name)+'.mat'))
        csi_avg = io.loadmat(csi_avg_path)['avg_csi']
        csi_avg = np.array(csi_avg)
        csi_avg = csi_avg.astype(np.complex128)

        csi = csi - csi_avg
        '''csi_amp = abs(csi)
        csi_amp = torch.FloatTensor(csi_amp).permute(0,1,3,2) #csi tensor: (3*3*30*20 -> 3*3*20*30)
        
        csi_ph = np.unwrap(np.angle(csi))
        csi_ph = fft.ifft(csi_ph)
        csi_phd = csi_ph[:,:,:,1:20] - csi_ph[:,:,:,0:19]
        csi_phd = torch.FloatTensor(csi_phd).permute(0,1,3,2)'''
        
        
        csi_amp = self.wdt(csi)
        csi_ph = self.phase_deno(csi)
        csi = np.concatenate((csi_amp, csi_ph), axis=2)
        csi = torch.FloatTensor(csi).permute(0,1,3,2)
        
        # torch.save(csi, '/home/qianbo/csi_avg.pkl')
        annotation = np.load(annotation_path, allow_pickle=True)['results'].item()
        # 'cam', 'global_orient', 'body_pose', 'smpl_betas', 'smpl_thetas', 
        # 'center_preds', 'center_confs', 'cam_trans', 'verts', 'joints', 'pj2d_org'
        pose = annotation["smpl_thetas"]
        shape = annotation["smpl_betas"]
        keypoint = annotation["joints"][:,:54]
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*71=(54+17)*3)
        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)
        gt_bboxes = torch.tensor([])
        gt_areas = torch.tensor([])
        result = dict(img=csi, gt_poses=pose, gt_shapes=shape ,gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes, gt_areas = gt_areas)
        return result
    
    def __getitem__(self, index):
        result = self.get_item_single_frame(index)
        return self.pipeline(result)

    def __len__(self):
        return len(self.filename_list)

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
        root = "/data1/qianbo/wifi_mesh_data/save_result/"
        for i in range(len(results)):
            info = self.get_item_single_frame(i)
            gt_keypoints = info['gt_keypoints']
            det_bboxes, det_keypoints, det_joints, det_verts = results[i]
            for label in range(len(det_keypoints)):
                kpt_pred = det_keypoints[label]
                kpt_pred = torch.tensor(kpt_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
                mpjpe_3d, _ = self.calc_mpjpe(gt_keypoints, kpt_pred)
                mpjpe_3d_list.append(mpjpe_3d.numpy())

                joints_pred = det_joints[label]
                joints_pred = torch.tensor(joints_pred, dtype=gt_keypoints.dtype, device=gt_keypoints.device)
                mpjpe_joints, corres = self.calc_mpjpe(gt_keypoints, joints_pred)
                mpjpe_smpl_list.append(mpjpe_joints.numpy())

                # verts_pred = det_verts[label][corres]

                # frame_name = self.filename_list[i]
                # frameresult_path = os.path.join(root, frame_name)
                # if not os.path.exists(frameresult_path):
                #     os.mkdir(frameresult_path)
                # np.save(os.path.join(frameresult_path, "verts.npy"), verts_pred)
                # print("test")

        mpjpe_3d = np.array(mpjpe_3d_list).mean()   
        mpjpe_smpl = np.array(mpjpe_smpl_list).mean()      
        result = {'mpjpe_3d':mpjpe_3d, "mpjpe_smpl":mpjpe_smpl}
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


import os
import h5py
import numpy as np

def load_mat(path: os.path):
    result = {}
    with h5py.File(path, mode='r') as file:
        for key in file.keys():
            result[key] = np.array(file[key])
    return result

if __name__ == '__main__':
    data = load_mat("/home/wangpengcheng/WiMU/opera_test/csi/00000140.h5")
    print(data['wifi'][0][0][0][0])
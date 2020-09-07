# Spike Sorter file import methods    
import numpy as np
from matplotlib import pyplot as plt
		   
def LoadRawData_mat(self, path):
    
    self.dir=dir
    
    import scipy.io as sio
    mat_data = sio.loadmat(path)
    
    dat_dummy = mat_data['data'][0]
    self.dat = np.array(dat_dummy,ndmin=2)
    self.spike_times = mat_data['spike_times'][0][0][0]
    self.spike_class = np.array(list(map(lambda i: mat_data['spike_class'][0][i][0],range(mat_data['spike_class'].shape[1]))))


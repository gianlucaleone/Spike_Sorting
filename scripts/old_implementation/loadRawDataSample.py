path = 'sources/singlerecording/'
dir = 'rawDataSample.bin'
fin = open(path+dir,'r')

import numpy as np
dat_inLine = np.fromfile(fin, dtype=np.int16) 
fin.close()

dir = 'channel_map.npy'
fin = open(path+dir,'r')
chanMap_dirty = np.fromfile(fin, dtype=np.int32) 
chanMap = chanMap_dirty[20:]
fin.close()

sup1=dat_inLine.reshape(1800000,385)
sup2=np.transpose(sup1)

dat=sup2[chanMap]
# Spike Sorter Feature Extraction
import numpy as np
import matplotlib.pyplot as plt


def FSDE(self,w):
    
    d = np.array(list(map(lambda x,y : x-y, w[1:],w[0:-1])))           
    dd = np.array(list(map(lambda x,y: x-y, d[1:],d[0:-1] )))
    
    # since the differences order is inverted, are also inverted min and max
    return min(d),max(d),min(dd),max(dd)#, int(np.where(d==max(d))[0][0])
    
def FSDE_NO_Dmin    (self,w):
    
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[1,2,3]]
    
def FSDE_NO_Dmax    (self,w):
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,2,3]]

def FSDE_NO_DDmin   (self,w):
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,1,3]]
    
def FSDE_NO_DDmax   (self,w):
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,1,2]]
    
def FSDE_NO_Dminmax     (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[2,3]]
    
def FSDE_NO_DDminmax    (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,1]]
    
def FSDE_NO_DDDmin      (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[1,3]]

def FSDE_NO_DDDmax      (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,2]]

def FSDE_NO_DminDDmax   (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[1,2]]

def FSDE_NO_DmaxDDmin   (self,w):
        
    FSDE_features = np.array(FSDE(self,w))
    return FSDE_features[[0,3]]

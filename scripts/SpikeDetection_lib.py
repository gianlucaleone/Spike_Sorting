# Spike Sorter: detection methods
import numpy as np

def Offline_Median(self, offset = 0,filter_type='FIR'):
    
    ###### PEAK DETECTION ######
    import statistics
    
    ###### find all the peaks above thr  ######
    if filter_type == 'none':
        self.thr = np.array(list(map(lambda x: self.alpha*statistics.median(abs(x))/0.6745,self.dat)))
        condition = np.array(list(map(lambda x,y: abs(x) > y, self.dat, self.thr))) #all the sample above thr
        #index of the samples above the threshold
        self.index = np.array(list(map(lambda x: np.arange(len(self.dat[0,:]),dtype='i')[x], condition))) 
        ###### remove spikes that violate the Inter Spike Time  ######                 
        self.rindex = np.array(list(map(lambda x: istViolation(x,self.delta),self.index)))
        ###### SPIKE ISOLATION ######  
        # take bigger windows if alignment is performed (offset)
        w = int(self.delta/2)
        # on channels
        self.windows = np.array(list(map(lambda d,ri: self.getWindows(d,ri,w,self.delta), self.dat,self.rindex)))
    else: 
        self.thr = np.array(list(map(lambda x: self.alpha*statistics.median(abs(x))/0.6745,self.datafir)))
        condition = np.array(list(map(lambda x,y: abs(x) > y, self.datafir, self.thr))) #all the sample above thr
        #index of the samples above the threshold
        self.index = np.array(list(map(lambda x: np.arange(len(self.datafir[0,:]),dtype='i')[x], condition))) 
        ###### remove spikes that violate the Inter Spike Time  ######                 
        self.rindex = np.array(list(map(lambda x: istViolation(x,self.delta),self.index)))
        ###### SPIKE ISOLATION ######  
        # take bigger windows if alignment is performed (offset)
        w = int(self.delta/2)
        # on channels
        self.windows = np.array(list(map(lambda d,ri: self.getWindows(d,ri,w,self.delta), self.datafir,self.rindex)))
    
 
def istViolation(ind,d):
#index che non violano inter spike time
    rind = np.array([],dtype='i') 
    rind = np.append(rind,ind[0])  
    j=0
    for i in range(1,len(ind)):
        if (ind[i]-rind[j]) > d:
            rind=np.append(rind,ind[i])  
            j+=1
    return rind 

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

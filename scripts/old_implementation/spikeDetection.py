###### PEAK DETECTION ######
import statistics

###### find all the peaks above thr  ######
alpha=5 # [3,6]
thr = np.array(list(map(lambda x: alpha*statistics.median(abs(x))/0.6745,datafir)))
condition = np.array(list(map(lambda x,y: x > y, datafir, thr))) #all the sample above thr

#index of the samples above the threshold
index = np.array(list(map(lambda x: np.arange(len(datafir[0,:]),dtype='i')[x], condition))) 

###### remove spikes that violate the Inter Spike Time  ######
ist = 3 #inter spike time definition [ms]
delta = ist*f_s/1000 # ist in numero di sample 

#index che non violano inter spike time
def istViolation(ind,d):
    rind = np.array([],dtype='i') 
    rind = np.append(rind,ind[0])  
    j=0
    for i in range(1,len(ind)):
        if (ind[i]-rind[j]) > d:
            rind=np.append(rind,ind[i])  
            j+=1
    return rind    
 

 
rindex = np.array(list(map(lambda x: istViolation(x,delta),index)))
          
###### SPIKE ISOLATION ######
half_delta = int(delta/2)

def getWindows(data,rind,hd):
    if delta%2==0 :
        windows = np.array(list(map(lambda x: data[x-hd:x+hd],rind)))
    else:
        windows = np.array(list(map(lambda x: data[x-hd:x+hd+1],rind)))
    return windows



windows = np.array(list(map(lambda d,ri: getWindows(d,ri,half_delta), datafir,rindex)))
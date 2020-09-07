import numpy as np
from matplotlib import pyplot as plt

from scripts.LoadRawData_lib import *
from scripts.filtering import *
from scripts.FeatureExtraction_lib import *
from scripts.run_collection import *
from scripts.SpikeDetection_lib import *

 
class SpikeSorter:
    
    def __init__(self, nwaves=8):
       self.nwaves=nwaves # how many spikes were superimposed in the reports
       self.offset = 0
       self.barWidth = 0.30
       self.dont_print=True
       self.half_window=False
       
    def LoadRawData (self,
                  path,
                  CH,
                  type
                  ):
        print('loadrowdata new')                
        dispatch = {
                    'mat':LoadRawData_mat
                   }
        
        foo = dispatch[type]
        foo(self,path)      
        
    def Filter(self, f_s=30000,f1=300,f2=3000,order=10, CH=4, filter_type='fir'):
        print('filter new')
        self.filter_type = filter_type
        
        dispatch = {'fir'         : fir_filter,
                    'butterworth' : butterworth_filter,
                   }
        
        self.filter_type = filter_type
        filter_function = dispatch[filter_type]
        filter_function(self,f_s,f1,f2,order, CH)


    def Detection(self,alpha = 2,ist = 2.5, offset = 0,window=0,detection_type='offline',filter_type = 'FIR'):
        print('detection new')
        self.offset = offset # alignment: take bigger windows to align them later
        self.alpha=alpha # [3,6]
        self.ist = ist  #inter spike time definition [ms]
        delta = self.ist*self.f_s/1000 # ist in numero di sample 
        self.delta = int(delta)
        
        self.detection_type = detection_type        
        Offline_Median(self,offset,filter_type)
        

    def getWindows(self,data,rind,w,d): 
            # on spikes in the channel..
            print('getwindows new')
            def p(x,w):
                if (x-w) > 0:
                    return x-w
                else:
                    return 0
            
            if d%2==0 :
                windows = np.array(list(map(lambda x: data[p(x,w):x+w],rind)))
            else:
                windows = np.array(list(map(lambda x: data[p(x,w):x+w+1],rind)))
                
            return windows
        
    def FeatureExtraction(self, feature_type):
        print('fe new')
        dispatch = {'FSDE'  : FSDE,
                    'FSDE1' : FSDE_NO_Dmin,
                    'FSDE2' : FSDE_NO_Dmax,
                    'FSDE3' : FSDE_NO_DDmin,
                    'FSDE4' : FSDE_NO_DDmax,
                    'FSDE5' : FSDE_NO_Dminmax,
                    'FSDE6' : FSDE_NO_DDminmax,
                    'FSDE7' : FSDE_NO_DDDmin,
                    'FSDE8' : FSDE_NO_DDDmax,
                    'FSDE9' : FSDE_NO_DminDDmax,
                    'FSDE10': FSDE_NO_DmaxDDmin
                   }
        
        self.feature_type = feature_type
        feature_function = dispatch[feature_type]
        
        # on windows of the channel
        def features_onChannel(ch,feature_function):
        
            features = np.array(list(map(lambda x: feature_function(self,x), ch))) # for every window computes the feature
            return features

        # on channels
        features_dummy = np.array(list(map(lambda x: features_onChannel(x,feature_function), self.windows)))
        self.features_dummy = features_dummy
        self.featuresT = features_dummy#[:][0:-1]
        self.features = np.transpose(self.featuresT)
                     
    def getPC(self,ncomponents = 2):
        from sklearn.decomposition import PCA 
        pca = PCA(n_components = ncomponents)
        self.principalComponents = pca.fit_transform(self.windows[0])
    
    def run(self,which='normal',ist=2,f_s=24000,f1=300,f2=3500,order=63,path='C_Easy1_noise005',
            Feature_type='FSDE3',filter_type='fir',training_spikes=0):
        print('run new')
        dispatch = {'normal':run_normal,
                    'fsde':run_fsde,
                    'spike_times':run_spike_times,
                    'st_all_ds': run_spike_times_all_dataset
                   }
        
        foo = dispatch[which]
        print(foo)
        
        if which == 'spike_times':
            foo(self,path=path,Feature_type=Feature_type, 
                ist=ist, f_s=f_s, f1=f1, f2=f2, 
                filter_type=filter_type, order=order,training_spikes=training_spikes) 
        elif which == 'st_all_ds':
            return foo(self, path=path, Feature_type=Feature_type,
                ist=ist, f_s=f_s, f1=f1, f2=f2, 
                filter_type=filter_type, order=order,training_spikes=training_spikes)        
        else: 
            foo(self)   
    
            
    def Kmeans(self,k=3,pc=False):
        print('kmeans new')
        if pc:
            feature = self.principalComponents
        else:
            feature = self.featuresT
        
        from sklearn.cluster import KMeans
        
        # to work with single channel run methods 
        if type(k) != list:
            k=list([k])
        
        self.kmeans_mc = list([])
        for ch,i in zip(feature,range(feature.shape[0]+1)):
            kmeans = KMeans(n_clusters=k[i], init='k-means++', 
                            max_iter=300,n_init=10, random_state=0)
            pred_y = kmeans.fit_predict(ch)
            
            if self.dont_print == False:
                for j in range(max(kmeans.labels_)+1):        
                    here = np.where(kmeans.labels_ == j)
                    plt.scatter(ch[here][:,0],ch[here][:,1])
                    
                plt.scatter(kmeans.cluster_centers_[:,0], 
                            kmeans.cluster_centers_[:,1], 
                            s=300, c='red') 
                plt.savefig('output/kmeans_'+self.feature_type+'_'+str(j))
                plt.close()
            self.kmeans_mc.append(kmeans)
        self.kmeans=self.kmeans_mc[0]      
          	    

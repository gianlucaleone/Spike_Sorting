# Spike Sorter collection of run methods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def run_normal(self):

    self.LoadRawData(dir='C_Easy1_noise005',type = 'mat')
    self.Filter(f_s=24000,f1=300,f2=3400,order=63, CH=1)
    self.Detection(ist=2)
             
    self.FeatureExtraction('FSDE3')
    self.Kmeans(3) 

      
def Wave_superimposition(self):
    nrow = self.nwaves>>1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # print the first 100 waves superimposed for each and every cluster IT
    for j in range(0, max(self.kmeans.labels_)+1):
        for i,k in zip(np.where(self.kmeans.labels_ == j)[0][0:self.nwaves],range(self.nwaves)):
            w = self.windows[0,i]
            nfeat=int(len(self.featuresT[0]))
            #o = int(self.features[nfeat-1][i])
            plt.subplot(nrow,4,k+1)
            plt.plot(w,color=colors[k])
            plt.subplot(2,1,2)
            plt.plot(w)
        plt.savefig('output/'+self.dir+'_clustered_superimposition_'+self.feature_type+'_'+str(j)+'.png')
        plt.close()

def GT_plot(self):
    # print nwaves waves per each cluster from spike_times and spike_class
    if self.half_window:
        delta_int = self.delta<<1
    else:
        delta_int = self.delta
    nrow = self.nwaves>>1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for j in range(1, self.spike_class[0].max() + 1):
        for i,k in zip(np.where(self.spike_class[0]==j)[0][0:self.nwaves],range(self.nwaves)):
            I = int(self.spike_times[i])
            plt.subplot(nrow,4,k+1)
            plt.plot(self.dat[0, I : I+int(delta_int)],color=colors[k])
            plt.subplot(2,1,2)
            plt.plot(self.dat[0, I : I+int(delta_int)])
        plt.savefig('output/'+self.dir+'_GT_superimposition_'+str(j)+'.png')
        plt.close()   
        
        
def run_fsde(self):
    
    self.LoadRawData(type='mat')
    self.Filter(f_s=24000,f1=300,f2=3500,order=63, CH=1)
    self.Detection(ist=2.5,alpha=6)
    self.FeatureExtraction('FSDE')
 
    self.Kmeans(3) 
    Wave_superimposition(self)
    GT_plot(self)

def run_spike_times(self,ist=2,f_s=24000,f1=300,f2=3500,order=63,
                    path='C_Easy1_noise005', Feature_type='FSDE3', 
                    HalfWindow=True,filter_type='fir', training_spikes=0):
    
    alpha = 2.3
    
    self.LoadRawData(type='mat',path=path,CH=1)
    if filter_type != 'none':
        self.Filter(filter_type=filter_type,f_s=f_s,f1=f1,f2=f2,order=order, CH=1)

    self.ist = ist
    self.f_s = f_s
    self.delta= int(ist*f_s/1000)
    w=int(self.delta/2)
    self.half_window = HalfWindow
    import statistics
    
    # l'ultimo delta>>1 serve perché le finestre vengono prese con l'indice come
    # punto centrale ma spike_times da come indice il punto iniziale
    if filter_type == 'fir':
        spike_times_shift = np.array([self.spike_times - (order>>1) - 1 + (int(self.delta)>>1)]) 
        self.thr = np.array(list(map(lambda x: alpha*statistics.median(abs(x))/0.6745,self.datafir)))   
    else:
        spike_times_shift = np.array([self.spike_times + (int(self.delta)>>1)]) 
    
    if filter_type == 'none':
        spike_times_shift = np.array([self.spike_times + (int(self.delta)>>1)]) 
        self.windows = np.array(list(map(lambda d,ri: self.getWindows(d,ri,w,self.delta), self.dat,spike_times_shift)))
        self.thr = np.array(list(map(lambda x: alpha*statistics.median(abs(x))/0.6745,self.dat)))   
    else:
        self.windows = np.array(list(map(lambda d,ri: self.getWindows(d,ri,w,self.delta), self.datafir,spike_times_shift)))
   
    if training_spikes != 0:
        self.windows_real = np.copy(self.windows)
        self.windows = np.copy(self.windows[:,0:training_spikes])
    
    self.FeatureExtraction(Feature_type)

    k=self.spike_class.shape[0]
    self.Kmeans(k) 
    
    if self.dont_print == False:
        Wave_superimposition(self)
        GT_plot(self)
    
    c  = np.array(list(map(lambda i: np.where(self.kmeans.labels_==i), range(k))))
    gt = np.array(list(map(lambda i: np.where(self.spike_class[0]==(i+1)), range(k))))
    
    import texttable as tt
    tab = tt.Texttable()
    headings = list(map(lambda i: 'c'+str(i),range(k)))
    headings.insert(0,' ')
    headings.append('best')
    tab.header(headings)
    rows = list(map(lambda i: 'gt'+str(i),range(k)))
    
    self.c = c
    self.gt = gt
    
    filename = 'output/accuracy_recordings.txt'
    fp = open(filename,'a')
    ratio=np.arange(k*k,dtype='float').reshape(k,k)
    fp.write(path+'\n')
    fp.write(str(k)+'\n')
    
    for i in range(k):
        for j in range(k):
            ratio[i][j] = len(list(set(list(c[j][0])).intersection(list(gt[i][0]))))/len(gt[i][0])
    
    ratio2=np.copy(ratio)
    m=[]
    for i in range(k):    
        ma=ratio2.max()
        mm=ratio2.argmax()
        mx=int(mm/k)
        my=mm%k
        
        for j in range(k):
            ratio2[j][my] = 0  
        
        r=list()
        for j in range(k): 
            r.append(ratio[mx][j]) 
                           
        r.append(ma)
        m.append(ma)
        
        r.insert(0,rows[i]) #insert gt0-1-2
        tab.add_row(r)     
        fp.write(str(ma)+'\n')
        
    fp.close()
    print('\n'+path)
    last = ['']*(k+1)
    last.append(np.array(m).mean())
    tab.add_row(last)  
    s = tab.draw()
    print (s)

def run_spike_times_all_dataset(self, path, Feature_type, ist,
                                f_s, f1, f2, filter_type,
                                order, training_spikes):
    
    dirs= [
        'C_Easy1_noise005',
        'C_Easy1_noise01',
        'C_Easy1_noise015',
        'C_Easy1_noise02',
        'C_Difficult1_noise005',
        'C_Difficult1_noise01',
        'C_Difficult1_noise015',
        'C_Difficult1_noise02',
        'C_Easy2_noise005',
        'C_Easy2_noise01',
        'C_Easy2_noise015',
        'C_Easy2_noise02',
        'C_Difficult2_noise005',
        'C_Difficult2_noise01',
        'C_Difficult2_noise015',
        'C_Difficult2_noise02'        
        ]
    self.dirs=dirs
    
    filename = 'output/accuracy_recordings.txt'    
    fp = open(filename,'w')
    fp.close()
    
    for d in dirs:
        run_spike_times(self=self,path=path+d, Feature_type=Feature_type,
                        ist=ist, f_s=f_s, f1=f1, f2=f2, filter_type=filter_type,
                        training_spikes=training_spikes)
    
    fp = open(filename,'r')
    
    avg = list()
    while True:
        ds = fp.readline()
        if not ds:
            break
        k = int(fp.readline())
        sum=0
        for i in range(k):
            sum += float(fp.readline())
        avg.append(sum/k)
        
    fp.close()
    
    print(np.array(avg).mean())
    
    if self.dont_print == False:
        barlist=plt.barh(range(len(avg)),avg,color='#557f2d',height=self.barWidth*2, edgecolor='white')
        plt.yticks(range(len(avg)), dirs, fontsize=8, rotation=45)
        plt.xlabel('Average Accuracy')
        plt.ylabel('Quiroga Datasets')
        plt.grid(axis='x',zorder=0)
        plt.savefig('output/ds_'+'HW'+str(int(HalfWindow))+'_swipe.pdf', dpi=800, bbox_inches = "tight")
        plt.close()

    return avg  

import numpy as np

def fir_filter(self, f_s=30000,f1=300,f2=3000,order=10, CH=4):

        #analisi in frequenza
        #fft data0
        from scipy import fftpack
        
        self.f_s=f_s #sampling frequency
        self.CH=CH
        
        ###### FIR FILTER  ######
        from scipy.signal import firwin
        from numpy import convolve as np_convolve
        
        # b coefficient (order, [band], ..,..,sampling freq)
        b = firwin(order, [f1,f2], width=0.05, pass_zero='bandpass',fs=self.f_s) 
        ###### CONVOLUTION  ######
        self.datafir = np.array(list(map(lambda x: np_convolve(x,b,mode='valid'),self.dat[0:self.CH,:])))
        self.datafir0 = self.datafir[0,:]
        
def butterworth_filter(self, f_s=30000,f1=300,f2=3000,order=10, CH=4):
       
       self.f_s=f_s #sampling frequency
       self.CH=CH
       
       from scipy import signal
       
       w1 = f1/(f_s/2) # normalize cutoff frequency
       w2 = f2/(f_s/2) # normalize cutoff frequency
               
       b, a = signal.butter(order, [w1,w2], btype='band') 
       
       self.datafir = np.array(list(map(lambda x: signal.filtfilt(b,a,x),self.dat[0:self.CH,:])))

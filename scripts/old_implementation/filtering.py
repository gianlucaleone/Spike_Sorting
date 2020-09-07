#analisi in frequenza
#fft data0
from scipy import fftpack

f_s=30000 #sampling frequency

###### FIR FILTER  ######
from scipy.signal import firwin
from numpy import convolve as np_convolve

# b coefficient (order, [band], ..,..,sampling freq)
b = firwin(10, [300,3000], width=0.05, pass_zero='bandpass',fs=f_s) 
###### CONVOLUTION  ######
CH = 4 # SELECT THE NUMBER OF CHANNEL
datafir = np.array(list(map(lambda x: np_convolve(x,b,mode='valid'),dat[0:CH,:])))
datafir0 = datafir[0,:]


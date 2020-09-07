######### INTEGRAL TRANSOFRM (IT) #########
def IT_Window(w):
    Ip=In=0
    for item in w:
        if item > 0:
            Ip += item
        else:
            In += item
    return Ip,In
    
        
def IT_Channel(ch):
    IT = np.array(list(map(lambda x: IT_Window(x), ch)))
    return IT
    

IT = np.array(list(map(lambda x: IT_Channel(x), windows)))

######### ZERO CROSS FEATURE (ZCF) #########
def ZCF_Window(w):
    i = np.where(w <= 0)[0][0]
    if not(i):
        return sum(w), 0
    else:
        Ib = sum(w[0:i])
        Ia = sum(w[i:])
        return Ib,Ia
        
        
   
   
def ZCF_Channel(ch):
    ZCF = np.array(list(map(lambda x: ZCF_Window(x), ch)))
    return ZCF
    



ZCF = np.array(list(map(lambda x: ZCF_Channel(x), windows)))

import matplotlib.pyplot as plt
plt.plot(ZCF[0],'bo')
plt.xlabel('ZCF 1')
plt.ylabel('ZCF 2')

plt.savefig('img/ZCFchannel0') 

plt.close()



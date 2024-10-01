import numpy as np
from matplotlib import pyplot as plt

no_of_sources=8
pmax=2
gain_threshold=0.1

def source(no_of_spuces):
     return np.array([i for i in range(1,no_of_sources+1)])

msg=source(no_of_sources)
print("message:",msg)

np.random.seed(None)
channel_coeff=np.random.randn(len(msg))
print("channel_coeff:",channel_coeff)

index=channel_coeff>gain_threshold
print("index:",index)
 
 
   
transmitted_1=msg[index]
print("transmitted1-->",transmitted_1)

channel_coeff1=channel_coeff[index]
print("channe_coeff-->",channel_coeff1)

pre_process=1
# np.minimum((np.sqrt(pmax)) ,  gain_threshold/channel_coeff1)
#print("pre_process",pre_process)

post_process=1
# (np.sum(np.mean(pre_process*channel_coeff1))) / ( np.sum(np.var(pre_process*channel_coeff1)) + 2*(np.var(channel_coeff1)) )
#print("post process",post_process)
noise=np.random.normal(0,1,len(transmitted_1))
print("gaus noise:    ",noise)
    
receieved = post_process * ( (transmitted_1 * channel_coeff1 * pre_process) )+ noise 
print("recieved1   -->",receieved)

print("\n\n")




def demod(transmitted,received_signal,channel_coeff):
    possible_symbols=transmitted
    print("msg:",possible_symbols,"\nmsg_sum:",possible_symbols.sum())
    print("channel_coeff\n",channel_coeff)
    print("received:\n",received_signal,"\n\n")
    recovered_sum=[]
    
    for r in received_signal:
        min_distance = float('inf')
        best_symbol = None  # To keep track of the symbol with the minimum distance
        for x in possible_symbols:
            # print("x=",x,"r=",r)
            distance = np.abs(r - x)
            # print("\t\t\t  d=",distance)
            if distance < min_distance:
               min_distance = distance
               best_symbol = x
        # print("min_distance:",min_distance,"best_symb:",best_symbol) 
        print("---------------------------------------------------------------------")
        recovered_sum.append(best_symbol)       
    return np.array(recovered_sum)

recoverd=demod(transmitted_1,receieved,channel_coeff1)
print("recoverd:",recoverd,"\nrecovered_sum:",recoverd.sum())













































# while source.size>1:
#     source=source[channel_coeff<=gain_threshold]
#     print("source:",source)

#     channel_coeff=np.random.rayleigh(1,len(source))
#     #print("channel_coeff:",channel_coeff)

#     index=channel_coeff>gain_threshold
#     print("index:",index)
    
    
#     trasnmitted_2=source[index]
#     print("transmitted2-->",trasnmitted_2)

#     channel_coeff1=channel_coeff[index]
#     print("channel_coeff2-->",channel_coeff1)

#     pre_process=np.minimum((np.sqrt(pmax)) ,  gain_threshold/channel_coeff1)
#     #print("pre_process",pre_process)

#     post_process=(np.sum(np.mean(pre_process*channel_coeff1))) / ( np.sum(np.var(pre_process*channel_coeff1)) + 2*(np.var(channel_coeff1)) )
#     #print("post process",post_process)
        

        
#     receieved =   post_process * ( (trasnmitted_2 * channel_coeff1 * pre_process) + np.random.normal(0,1,len(trasnmitted_2)))
#     print("recieved-->",receieved)

#     print("\n")
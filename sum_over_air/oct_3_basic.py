import numpy as np
from matplotlib import pyplot as plt

no_of_sources=5
gain_threshold=0.1
snr_db=np.arange(1,2,1)


def source(no_of_sources):
    # np.random.uniform(-2,2,no_of_sources)
    return np.random.uniform(-1,1,no_of_sources)


def multislot(msg):
    np.random.seed(None)
    channel_coeff=np.random.randn(len(msg))
    print("channel_coeff:",channel_coeff)
    return msg*channel_coeff

def channel(transmitted,snr_db):
    faded_msg=multislot(transmitted)
    print("faded msg:",faded_msg)
    
    noise=np.random.randn()
    print("noise:",noise)
    # Add the noise to the combined symbols and return the result
    return np.sum(faded_msg) + noise

def demod(transmitted,received_signal):
    return received_signal

def calculate_error(msg,recoverd):
    print("msg}",msg.sum())
    print("rvd}",recoverd)
    # print("------------------------------------------------------------")
    return  np.abs(msg.sum()-recoverd.sum())

def simulate(snr_db,no_of_sources):
    error=[]
    for snr in snr_db:
        print("====================================================================================")
        transmitted=source(no_of_sources)
        print("transmitted:",transmitted)
        
        # pre_processed=pre_process(transmitted)
        # print("pre processed:",pre_processed)
        
        received=channel(transmitted,snr)
        print("recieved:",received)
        
        recoverd=demod(transmitted,received)
        print("recoverd:",recoverd)
        error.append(calculate_error(transmitted,recoverd))
        print("====================================================================================")
    return error

result=simulate(snr_db,no_of_sources)
print("error:",result)


# plt.plot(snr_db,result)
# plt.show()
import numpy as np
import pandas as pd

xls_file = pd.ExcelFile('D:\IRIS_data_try1.xlsx')
xls_file
xls_file.sheet_names 
df = xls_file.parse('all_data')

tvs = xls_file.parse('train_validation_set')
tds = xls_file.parse('test_set')
train_data = np.zeros((120,5))
train_data[:,0] = np.array(tvs.a.tolist())
train_data[:,1] = np.array(tvs.b.tolist())
train_data[:,2] = np.array(tvs.c.tolist())
train_data[:,3] = np.array(tvs.d.tolist())
train_data[:,4] = np.array(tvs.e.tolist())

pattern = np.zeros((5,120))
desired_op = np.zeros((1,120))

hidden_op=np.zeros((6,1))
final_sum=np.zeros((1,6))

#hidden_sum=np.zeros((1,6))
pattern[0:4,:]=train_data[:,0:4].T
desired_op[0,:]=train_data[:,4].T
learning_rate=0.5

for i in range(120):
    pattern[4,i] = 1
    #desired_op[1,i] =1
    #pattern[1,i] = 1
#print(pattern)

def activation(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

np.random.seed(1)

def structure(n,m):
    pattern_matrix=np.random.random((n,(m+1)))
    return pattern_matrix
    
input_to_hidden= structure(5,4)  #neural_network
#print(input_to_hidden)
hidden_to_output= structure(1,5) 
#print(hidden_to_output)

for i in range(1):
    #hidden_sum[i,5]=1
    hidden_sum=(np.dot(input_to_hidden, pattern[:,i]))
    
    hidden_op[0:5,i]=activation(hidden_sum,deriv=False)
    hidden_op[5,i]=1
    
    final_sum=(np.dot(hidden_to_output, hidden_op[:,i]))
    final_op=activation(final_sum,deriv=False)
    del_op= final_op[i]*(1-final_op[i]*(final_op[i]-desired_op[i]))
    print("final_op")
    print(final_op)
    print("del_op")
    print(del_op)

#for j in range(100):
#    layer0=pattern
#    layer1=activation(np.dot(layer0,synapse0))
#    layer2=activation(np.dot(layer1,synapse1))
   
#    layer2_error=desired_op-layer2
#    if(j%10)==0:
#       print("error: "+str(np.mean(np.abs(layer2_error))))   
 #   layer2_delta=layer2_error*activation(layer2,deriv=True)
 #   layer1_error=layer2_error.dot(synapse1.T)
 #   layer1_delta=layer1_error*activation(layer1,deriv=True)
    
 #   synapse1 +=learning_rate*(layer1.T.dot(layer2_delta))
 #   synapse0 +=learning_rate*(layer0.T.dot(layer1_delta))
 #   print("output after training")
 #   print(layer2)
  #  print(synapse0)
  #  print(synapse1)
    
    


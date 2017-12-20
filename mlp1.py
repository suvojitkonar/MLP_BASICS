import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import tensorflow as tf
#loading the dataset as an excel file
xls_file = pd.ExcelFile('D:\IRIS_data_try1.xlsx')
xls_file
xls_file.sheet_names 
df = xls_file.parse('all_data')
#dividing the total dataset into training and testing dataset
tvs = xls_file.parse('train_validation_set')
tds = xls_file.parse('test_set')
#training starts
train_data = np.zeros((120,5))
train_data[:,0] = np.array(tvs.a.tolist())
train_data[:,1] = np.array(tvs.b.tolist())
train_data[:,2] = np.array(tvs.c.tolist())
train_data[:,3] = np.array(tvs.d.tolist())
train_data[:,4] = np.array(tvs.e.tolist())
#declarations of the matrices or arrays used in training
pattern = np.zeros((5,120))
desired_op = np.zeros((120))
del_op=np.zeros((5000))
final_op=np.zeros((1,120))
hidden_op=np.zeros((6,5000))
final_sum=np.zeros((1,6))
del_hidden=np.zeros((5,5000))
del_w_hidden=np.zeros((1,6))
del_w_input=np.zeros((5,5))
train_final=np.zeros((120))

#s=np.zeros((500))
#e=np.zeros((120))
#



#breaking of the train data set into pattern and desired
#the number of 1's in the last row of the pattern determines the biases

pattern[0:4,:]=train_data[:,0:4].T
desired_op[:]=train_data[:,4].T
#learning rate
learning_rate=0.01

for i in range(120):
    pattern[4,i] = 1
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#sigmoid function
def activation(x):
    return 1/(1+np.exp(-x))

np.random.seed(1)

def structure(n,m):
    pattern_matrix=np.random.random((n,(m+1)))
    return pattern_matrix
 #input to hidden and hidden to output weights of the neurons   
input_to_hidden= structure(5,4)  #neural_network
hidden_to_output= structure(1,5) 


t1 = np.zeros((5,1))
t2 = np.zeros((1,4))
epoch=2000

epoch_w8=np.zeros((1,6,epoch))
epoch_wt=np.zeros((5,5,epoch))
error=np.zeros((epoch))
m=0
c=0
tr=120
#training loop starts here
for j in range(epoch):    
    s=0
    for i in range(120):
        #forward feeding
       hidden_sum=(np.dot(input_to_hidden, pattern[:,i]))
    
       hidden_op[0:5,i]=activation(hidden_sum)
       hidden_op[5,i]=1
    
       final_sum=(np.dot(hidden_to_output, hidden_op[:,i]))
       final_op[0,i]=activation(final_sum)
       train_final[i]=final_op[0,i]
       e=(final_op[:,i]-desired_op[i])**2    
       s=s+e
       #backpropagation
       del_op[i]= final_op[0,i]*(1-final_op[0,i])*(final_op[0,i]-desired_op[i])
       for x in range(5):
           del_hidden[x,i]= hidden_op[x,i] * (1-hidden_op[x,i]) * (del_op[i] * hidden_to_output[0,x])

       del_w_hidden[0,0:5]= -learning_rate * del_op[i] * hidden_op[0:5,i].T
    
       del_w_hidden[0,5] = learning_rate * del_op[i]
    
       t1[:,0] = del_hidden[:,i]
       t2[0,:] = pattern[0:4,i].T
    
       del_w_input[:,0:4] = -learning_rate * (t1 * t2)
       del_w_input[:,4]= learning_rate * del_hidden[:,i]
       #updation of all the weights
       hidden_to_output = hidden_to_output + del_w_hidden
       input_to_hidden = input_to_hidden + del_w_input    
    
    epoch_w8[:,:,j] = hidden_to_output
    epoch_wt[:,:,j] = input_to_hidden
    
    avg_squared_error=np.sqrt(e)
    error[j]=avg_squared_error/120
    
    print("epoch",(j+1)," complete\n")
for i in range(120):  
 if(train_final[i] <=0.33 and desired_op[i] <= 0.33):
        c = c + 1
 elif((train_final[i] > 0.33 and train_final[i] <= 0.66) and (desired_op[i] > 0.33 and desired_op[i] <= 0.66)):
        c = c + 1
 elif((train_final[i] > 0.66) and (desired_op[i] > 0.33)):
          c = c + 1
 else:
          m = m + 1

print("MLP training complete\n")    
print("MLP testing starts\n")   
    #testing starts
test_data = np.zeros((30,5))
test_data[:,0] = np.array(tds.a.tolist())
test_data[:,1] = np.array(tds.b.tolist())
test_data[:,2] = np.array(tds.c.tolist())
test_data[:,3] = np.array(tds.d.tolist())
test_data[:,4] = np.array(tds.e.tolist())  
#declarations of test matrices and arrays
test_pattern=np.zeros((5,30)) 
test_hidden_op=np.zeros((6,30))
test_final_op=np.zeros((1,30))
test_desired_op=np.zeros((30))
test_final=np.zeros((30))   
test_pattern[0:4,:]=test_data[:,0:4].T
test_desired_op[:]=test_data[:,4].T
test_final=np.zeros((30))

correct=0
miss=0
t=30

for i in range(30):
    test_pattern[4,i] = 1
    #testing loop starts here
for i in range(30):
      test_hidden_sum=(np.dot(input_to_hidden, test_pattern[:,i]))
      test_hidden_op[0:5,i]=activation(test_hidden_sum)
      test_hidden_op[5,i]=1
      
      test_final_sum=(np.dot(hidden_to_output, test_hidden_op[:,i]))
      test_final_op[0,i]=activation(test_final_sum)
      test_final[i]=test_final_op[0,i]
     
      #classification according to the final values
      if(test_final[i] >=0.33):
              print("iris setosa")
      elif((test_final[i] > 0.33 and test_final[i] <= 0.66)):
              print("iris versicolor")
      else:
              print("iris verginica")
              #calculation of correct classification and miss classification
      if(test_final[i] <=0.33 and test_desired_op[i] <= 0.33):
        correct = correct + 1
      elif((test_final[i] > 0.33 and test_final[i] <= 0.66) and (test_desired_op[i] > 0.33 and test_desired_op[i] <= 0.66)):
        correct = correct + 1
      elif((test_final[i] > 0.66) and (test_desired_op[i] > 0.33)):
        correct = correct + 1
      else:
        miss = miss + 1

y=error
x=np.zeros((epoch))
for i in range(epoch):
    x[i]=i
plt.title('error vs epoch')
plt.xlabel('epochs')
plt.ylabel('errors')
plt.plot(x,y)
plt.savefig('D:\data.jpg')
plt.axis([0, epoch, 0, 0.004])
plt.show()
print('Percentage of of correct classification for training = ',c/tr * 100)
print('total no of train samples= ',tr)
print('No. of correct classification = ',c)          
print('No. of miss classification = ',m)
print('Percentage of of correct classification for testing = ',correct/t*100) 
print('total no of samples= ',t)
print('No. of correct classification = ',correct)          
print('No. of miss classification = ',miss)  
print("END OF TESTING")

       

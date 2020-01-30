
# coding: utf-8

# In[20]:


import sys
import numpy as np
from scipy.spatial import distance
import time


if len(sys.argv) != 5:
	print('usage: trainingDataFileName testDataFileName predictionsFile k')
	sys.exit(99)
trainingFileName = sys.argv[1]
testFileName     = sys.argv[2]
outFileName      = sys.argv[3]
k                = int(sys.argv[4])
start_time = time.clock()
trainingData=np.loadtxt(fname=trainingFileName, delimiter=',')
testData=np.loadtxt(fname=testFileName, delimiter=',')
K=k
userID=trainingData[:,0].astype(np.int16)
movieID=trainingData[:,1].astype(np.int16)
rating=trainingData[:,2]
userIDMax=np.max(userID)
movieIDMax=np.max(movieID)
R = np.zeros(shape=(userIDMax+1,movieIDMax+1))
for r in trainingData:
    R[r[0].astype(np.int16),r[1].astype(np.int16)]=r[2]


# In[25]:


def Knn_unw(User, Movie,K):
    knn_dist=[]
    knn_ind=[]
    rates=[]
    for j,a in enumerate(R):
        if(j != User and R[j,Movie]!=0):
            knn_dist.append (distance.euclidean(R[User],a))
            rates.append(R[j,Movie])
            
    knn_dist=np.array(knn_dist)
    rates=np.array(rates)
    
    knn_index_sort= np.argsort(knn_dist)
    knn_sort=knn_dist[knn_index_sort]

    K=min(len(knn_dist),K)
    
    if (K==0):
        final_rate=0.0
    else:
        if(K!=(len(knn_dist))):
            while knn_sort[K-1] == knn_sort[K]:
                if(K==len(knn_dist)):
                    break
                else:
                    K=K+1
        final_rate=np.sum(rates[knn_index_sort][:K])/K
        
    
    return final_rate;


# In[26]:


userID_test=testData[:,0].astype(np.int16)
movieID_test=testData[:,1].astype(np.int16)
rating_actual=testData[:,2]
RMSE1=0
N=0
reportFile = open(outFileName,'w')
for i, j in enumerate (userID_test):
    rating_predict=Knn_unw(userID_test[i],movieID_test[i],K)
    bufstr = '%d,%d,%.1f,%.1f'%(userID_test[i],movieID_test[i],rating_actual[i],rating_predict)
    reportFile.write(bufstr+'\n')
    print (i, j,rating_predict)
    RMSE1=(rating_actual[i]-rating_predict)*(rating_actual[i]-rating_predict)+RMSE1
    N=N+1


RMSE='%.2f'%(np.sqrt(RMSE1/N))
reportFile.write("RMSE="+str(RMSE)+'\n')
reportFile.write("Total time:"+str(time.clock() - start_time)+ "seconds"+'\n')
reportFile.close()
print  ("Total time:",time.clock() - start_time, "seconds")
print(("RMSE=",RMSE))

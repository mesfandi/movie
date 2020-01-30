
# coding: utf-8

# In[7]:


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


# In[8]:


def Knn_w(User, Movie,K):
    knn_dist=[]
    knn_ind=[]
    rates=[]
    weight=[]
    knn_dist_sort=[]
    
    for j,a in enumerate(R):
        if(j != User and R[j,Movie]!=0):
            knn_dist.append (distance.euclidean(R[User],a))
            #knn_ind.append(j)
            rates.append(R[j,Movie])

    knn_dist=np.array(knn_dist)
    #knn_ind=np.array(knn_ind)
    rates=np.array(rates)
    

    knn_index_sort= np.argsort(knn_dist)
    knn_rates_sort=rates[knn_index_sort][:K]
    knn_dist_sort=knn_dist[knn_index_sort][:K]
    
    K=min(len(knn_dist),K)
    if (K==0):
        final_rate=0.0
    else:
        for z in range(K):
            if (knn_dist_sort[K-1]-knn_dist_sort[0]!=0):    
                weight.append((knn_dist_sort[K-1]-knn_dist_sort[z])/(knn_dist_sort[K-1]-knn_dist_sort[0]))
            else:
                weight.append(0)
                
        weight=np.array(weight)

        weight*knn_rates_sort
        if np.sum(weight)!=0:
            final_rate=np.sum(weight*knn_rates_sort)/np.sum(weight)
        else:
            final_rate=0
    return final_rate;


# In[9]:


userID_test=testData[:,0].astype(np.int16)
movieID_test=testData[:,1].astype(np.int16)
rating_actual=testData[:,2]
RMSE1=0.0
N=0
reportFile = open(outFileName,'w')
for i, j in enumerate (userID_test):
    rating_predict=Knn_w(userID_test[i],movieID_test[i],K)
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

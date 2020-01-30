
# coding: utf-8

# In[4]:


import sys
import numpy as np
from scipy.spatial import distance
import time

if len(sys.argv) != 4:
	print('usage: trainingDataFileName testDataFileName predictionsFile k')
	sys.exit(99)
trainingFileName = sys.argv[1]
testFileName     = sys.argv[2]
outFileName      = sys.argv[3]



start_time = time.clock()
trainingData=np.loadtxt(fname=trainingFileName, delimiter=',')
testData=np.loadtxt(fname=testFileName, delimiter=',')


userID=trainingData[:,0].astype(np.int16)
movieID=trainingData[:,1].astype(np.int16)
rating=trainingData[:,2]
userIDMax=np.max(userID)
movieIDMax=np.max(movieID)
R = np.zeros(shape=(userIDMax+1,movieIDMax+1))
for r in trainingData:
    R[r[0].astype(np.int16),r[1].astype(np.int16)]=r[2]
RT=R.T
R_centered_UIIman=np.copy(RT)
total_mean_UIIman = np.divide(R.sum(0), (R!=0).sum(0), out=np.zeros_like(R.sum(0)), where=((R!=0).sum(0))!=0)
for index, arr in enumerate(RT):
    R_centered_UIIman[index,R_centered_UIIman[index].nonzero()] += -total_mean_UIIman[index] 
Similarity=np.corrcoef(R_centered_UIIman)


# In[8]:







def UIIman(User, Movie):
    final_rate_UIIman=0.0
    weight=0.0
    temp1=0
    for j,a in enumerate(R[User]):
        if(j != Movie and not np.isnan(Similarity[j,Movie]) and Similarity[j,Movie]>0 and a!=0):
            temp1=Similarity[j,Movie]*a+temp1
            weight=Similarity[j,Movie]+weight
        

    if weight==0:
        final_rate_UIIman=0
    else:
        final_rate_UIIman=temp1/weight
    return final_rate_UIIman


# In[9]:



userID_test=testData[:,0].astype(np.int16)
movieID_test=testData[:,1].astype(np.int16)
rating_actual=testData[:,2]
RMSE1=0
N=0
reportFile = open(outFileName,'w')
for i, j in enumerate (userID_test):
    rating_predict=UIIman(userID_test[i],movieID_test[i])
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

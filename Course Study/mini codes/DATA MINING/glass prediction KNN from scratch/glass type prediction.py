
# coding: utf-8

# Attribute Information:
#    1. Id number: 1 to 214
#    2. RI: refractive index
#    3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
#    4. Mg: Magnesium
#    5. Al: Aluminum
#    6. Si: Silicon
#    7. K: Potassium
#    8. Ca: Calcium
#    9. Ba: Barium
#    10. Fe: Iron
#    11. Type of glass: (class attribute)
#    
#       - 1 building_windows_float_processed
#       - 2 building_windows_non_float_processed
#       - 3 vehicle_windows_float_processed
#       - 4 vehicle_windows_non_float_processed (none in this database)
#       - 5 containers
#       - 6 tableware
#       - 7 headlamps

# In[99]:


def calc(arr,test):
    arr1=[]
    for i in arr:
        calc=0
        for k in range (1,10):
            calc+=pow((i[k]-test[k]),2)
        arr1+=[[calc,i[10]]]
    return arr1    
            


# In[100]:


def KNNcalc(arr,n):
    arr1=[9999 for i in range(0,n)]
    arr2=[9999 for i in range(0,n)]
    for i in arr:
        if (i[0]<max(arr1)):
            for j in range(0,n):
                if(max(arr1)==arr1[j]):
                    arr1[j]=i[0]
                    arr2[j]=i[1]
                    break
    return arr1,arr2                
                


# In[101]:


def accuraccy(arr,n):
    count=0
    for i  in  arr:
        if i==n:
            count+=1
    return count*100/len(arr)    


# In[102]:


def split(data):
    train=[]
    test=[]
    for i in range(0,len(data)):
        if(i%15==0):
            test+=[data[i]]
        else:
            train+=[data[i]]
    return train,test        


# In[103]:


def pred(a):
    arr=[0 for i in range(0,8)]
    for i in a:
        arr[int(i)]+=1
    flag=max(arr)
    for i in range(0,8):
        if(arr[i]==flag):
            return(i)


# In[104]:


def acc(test,train,n):
    arr=[]
    error=0
    for i in test:
        print(i)
        z=calc(train,i)
        arr1,arr2=KNNcalc(z,n)
        acc=accuraccy(arr2,i[10])
        arr+=[acc]
        print("actual value: "+str(i[10])+"  predicted values: "+str(arr2))
        print("accuraccy :"+str(acc)+"\n\n")
        if(pred(arr2)!=i[10]):
            error+=1
    print("total accuraccy"+str(sum(arr)/15))
    
    print("total error"+str(error*100/15))
    return([sum(arr)/15,error*100/15])
        


# In[105]:


arr=[[1,2],[3,5],[8,4],[7,3],[2,5]]
arr1,arr2=KNNcalc(arr,4)
arr2


# In[106]:


import pandas as pd
import numpy as np
dataset = pd.read_csv("C:\\Users\\Sid\\Desktop\\python files\\glass prediction KNN from scratch\\glass.csv")
data=dataset.values.tolist()
train,test=split(data)


# In[107]:


dataset


# In[108]:


print(len(train))
print(len(test))


# In[109]:


z=calc(data,[1.0, 1.52101, 13.64, 4.49, 1.1, 71.78, 0.06, 8.75, 0.0, 0.0, 1.0])
r1,r2=KNNcalc(z,7)
accuraccy(r2,1)


# In[110]:


bestKNN=[]
for i in range(2,15):
    bestKNN+=[[i]+acc(test,train,i)]


# In[111]:


acc(test,train,4)


# In[114]:


#the best result is found using 8 neighbours having 64% acc match and 20% error
bestKNN


# In[119]:


z=calc(data,[176.0, 1.52119, 12.97, 0.33, 1.51, 73.39, 0.13, 11.27, 0.0, 0.28, 5.0])
arr1,arr2=KNNcalc(z,8)


# In[120]:


arr2


# In[121]:


pred(arr2)


# Intialisation. 
import pandas as pd
import random

na = 6 # Number of attribut
no = 1000 # Number of object
ns = 10 # Number of source
attributs = ['Property'+str(x) for x in range(1,na+1)]
objects = ['Object'+str(x) for x in range(1,no+1)]
sources = ['Source'+str(x) for x in range(1,ns+1)]

# Set of paramete (the value wil be between 0 and 1) : description un the papar
m1 = 1.0 
m2 = 0.0
m3 = 0.8 

# Floder to save the data after generated.
floder = "" 


# Definistion des valeurs vraix (GT)
def get_truth(objects,attributs):
    data = {
        'Object':[],
        'Property':[],
        'Value':[],
        'False':[]
    }
    for o in objects:
        for a in attributs:
            data['Object'].append(o)
            data['Property'].append(a)
            data['Value'].append(random.choice(range(100000,499999)))
            data['False'].append(random.sample(range(500000,999999),k=20))
    truth = pd.DataFrame(data)
    return truth
data_truth = get_truth(objects,attributs)


# Random selection of partition of P. 
def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]
n = random.choice(range(2,len(attributs)))
partition = partition(attributs,n)




# For each subset in P, we randomly choose a source from S which is deemed to be highly accurate on this subset
sources_ = {
    
}


for i in range(len(partition)):
    sources_[i]=[sources[i]]


l=0
for u in range(len(partition),len(sources)):
    sources_[l].append(sources[u])
    l+=1
    if l == len(partition):
        l=0



#  For every subset X1 in P together with the corresponding chosen source s in S 0 , 
#  we uniformly set using our distribution functions U1 and U2
indice_partition = 0
dataframe = pd.DataFrame({
        'ID':[],
        'Object':[],
        'Property':[],
        'Value':[],
        'Source':[]
    })

ID = 0
for x1 in partition:
    data = {
        'ID':[],
        'Object':[],
        'Property':[],
        'Value':[],
        'Source':[]
    }
    indice_sources = 0 
    for sources in sources_.values():
        accuracy = 0
        accuracy_ = 0
        if indice_sources==indice_partition:
            accuracy = random.uniform(m1,1)
            accuracy_ = accuracy
        else:
            accuracy = random.uniform(0,m2)
            accuracy_ = accuracy
        for source_index in range(len(sources)):
            if source_index!=0:
                accuracy = random.uniform(accuracy_-0.5,accuracy_)
                nunber_cov = random.uniform(m3,1)*len(x1)*len(objects)*accuracy
            else:
                nunber_cov = len(x1)*len(objects)*accuracy_
            
            nunber_ = 0
            for o in objects:
                for a in x1:
                    value = 0
                    value_truth = data_truth[(data_truth['Object'] == o) & (data_truth['Property'] == a)].values[0][2]
                    if nunber_ < nunber_cov:
                        value = value_truth
                    else:
                        while True:
                            value = random.choice(data_truth[(data_truth['Object'] == o) & (data_truth['Property'] == a)].values[0][3])
                            if value!=value_truth:
                                break

                    data['ID'].append(ID)
                    data['Object'].append(o)
                    data['Property'].append(a)
                    data['Value'].append(int(value))
                    data['Source'].append(sources[source_index])
                    nunber_+=1
            
        indice_sources+=1
        

    #print(tell_truths) 
    indice_partition+=1
    dataframe = pd.concat([dataframe,pd.DataFrame(data)],axis=0)
    dataframe['Value']=dataframe.Value.astype('int')





dataframe.to_csv(floder+'data.csv',index=False)
data_truth.to_csv(floder+'data_truth.csv',index=False)

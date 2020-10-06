#!/usr/bin/env python
# coding: utf-8

# <center><h1>Code Python</h1></center>

# ## Importation des librairies

# In[1]:


# Pandas
import pandas as pd
# Contenant l'algorithme du Truth Finder
import TruthFinder as tf 
# Contenant le partitionnement par force bruite appliqué avec le Truth Finder(Pour le moment)
import AccuGenPartition as agp 
import DepenAccuAccuSim as daas 


# In[2]:


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# In[3]:


import AccuAttCluster as aac


# In[4]:


#import AccuPartitionClustering as apc


# In[ ]:





# ## Importation des données

# In[5]:


dataframe = pd.read_csv("data_semi/62_attributs/domaine25/data.csv")
dataframe = pd.read_csv("stock/stock_clean_2011-07-01.csv")

dataframe = pd.read_csv("examens/62_attributs/data.csv")

dataframe = pd.read_csv("flight/flight_clean_2011-12-01.csv")

#dataframe = pd.read_csv("data_conf22/data.csv")
#dataframe['ID'] = [i for i in range(dataframe.shape[0])]
#dataframe = dataframe[['ID','Object','Property','Value','Source']]
dataframe.head(1)


# In[6]:


#dataframe.to_csv('dataconf1/data.csv',index=False)
#data_truth.to_csv('dataconf1/data_truth.csv',index=False)


# In[7]:


#dataframe[['Object','Property','Value']].drop_duplicates().groupby(by=['Object','Property'])
#for key, value_of_di in dataframe[['Object','Property','Value']].drop_duplicates().groupby(by=['Object','Property']):
    #print(key[0]+key[1])


# ## Importation des données vrai pour le calcule de précision avec la fonction __oracle__

# In[8]:


data_truth = pd.read_csv("data_semi/124_attributs/domaine25/data_truth.csv")


data_truth = pd.read_csv("examens/124_attributs/data_truth.csv")

#data_truth = pd.read_csv("stock/stock_truth_2011-07-01.csv")
#data_truth = pd.read_csv("data_conf22/data_truth.csv")
data_truth = pd.read_csv("flight/flight_truth_2011-12-01.csv")

#data_truth = data_truth[['Object','Property','Value']]
data_truth.head(1)


# In[9]:


dataframe = dataframe.drop_duplicates(subset=['Object', 'Property','Value','Source'], keep="first")


# In[ ]:





# ## Execution de MajorityVote sur les données

# In[17]:


def majorityVote(dataframe,data_truth):
    dataframe["Object"]=[str(x) for x in dataframe["Object"]]
    dataframe["Property"]=[str(x) for x in dataframe["Property"]]
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
    resultat = pd.DataFrame(columns=['Object','Property','Value'])
    u=0
    for o_p in dataframe["ObjectProperty"].unique():
        indices = dataframe["ObjectProperty"] == o_p
        df = dataframe.loc[indices]
        dic = dict(df.Value.value_counts())
        max_ = 0
        key = 0
        #print(dic)
        for i in range(len(dic)):
            if list(dic.values())[i] > max_:
                max_ = list(dic.values())[i]
                key = list(dic.keys())[i]
        resultat.loc[u] = [list(df.Object)[0],list(df.Property)[0],key]
        u+=1
        
    data_truth1 = tf.get_truth_to_dict(data_truth)
    resultat = tf.get_truth_to_dict(resultat)
    evaluation_r = tf.evaluation(data_truth1,resultat,dataframe)
    
    return evaluation_r


# In[18]:


get_ipython().run_cell_magic('time', '', 'majorityVote(dataframe,data_truth)')


# ## Execution de Truth Finder sur les données

# L'algorthme du Truth Finder que nous avons implémenté tient compte des données dont les valeurs sont continue d'une part et catégoriel d'autre. 
# 
# Il retourne une liste contenant respectivement : __les données de départ ajouter de deux nouvelles colonne (fiabilité des source et confidence des valeurs)__ , __Nombre itération__ , __Max accuracy__ , __Avg accuracy__ et __Oracle accuracy__ .

# In[50]:


get_ipython().run_cell_magic('time', '', 'out_tf = tf.train(dataframe,\n                  initial_trustworthiness=0.8,\n                  max_iterations=20,\n                  threshold=1e-6,\n                  data_truth=data_truth,\n                  rho=0.5,\n                  afficher=True)')


# ## Execution de DEPEN sur les données

# In[20]:


get_ipython().run_cell_magic('time', '', 'out_depen = daas.train(dataframe, max_iterations=20,\n          threshold=1e-6,initial_trustworthiness=0.8,c=0.2,n=50,alpha=0.2,rho=0.5,afficher=True,algo="DEPEN",data_truth=data_truth)')


# In[ ]:





# ## Execution de ACCU sur les données

# In[52]:


get_ipython().run_cell_magic('time', '', 'out_depen = daas.train(dataframe, max_iterations=20,\n          threshold=1e-6, initial_trustworthiness=0.8,c=0.2,n=50,alpha=0.2,rho=0.5,afficher=True,algo="ACCU",data_truth=data_truth)')


# ## Execution de ACCUSIM sur les données

# In[22]:


get_ipython().run_cell_magic('time', '', 'out_depen = daas.train(dataframe, max_iterations=20, \n          threshold=1e-6, initial_trustworthiness=0.8,c=0.2,n=50,alpha=0.2,rho=0.5,afficher=True,algo="ACCUSIM",data_truth=data_truth)')


# In[ ]:





# ## Execution de l'algorithme de partitionnement d'attribut par force bruite avec Truth Finder / DEPEN / ACCU ou ACCUSIM sur les données
# Attention cette exécution prend énormement de temps lorsque le nombre d'attribut est grande 

# In[11]:


get_ipython().run_cell_magic('time', '', '# algo= TF ou DEPEN ou ACCU ou ACCUSIM\nout_agp = agp.train(dataframe,\n                  initial_trustworthiness=0.8,\n                  max_iterations=10,\n                  threshold=1e-06,algo="ACCU",\n                  data_truth=data_truth,\n                  afficher=False)')


# In[13]:


out_agp1 = out_agp 


# In[18]:


out_agp = pd.read_csv('AccuGenPartion_Start_Dimanche_13_09_DS2.txt',sep=';',header=None)

out_agp.columns = ['Partition','AccuracyMax','AccuracyAvg','AccuracyOracle','precision','recall','accuracy','f1_score']


# In[ ]:





# In[30]:


out_agp[out_agp['AccuracyMax'] == max(out_agp['AccuracyMax'])].values[0]


# In[ ]:


out_agp[ (out_agp['AccuracyMax'] == max(out_agp['AccuracyMax'])) & (out_agp['AccuracyAvg'] == max(out_agp['AccuracyAvg']))]


# In[24]:


out_agp[out_agp['AccuracyAvg'] == max(out_agp['AccuracyAvg'])].values


# In[32]:


out_agp[out_agp['AccuracyOracle'] == max(out_agp['AccuracyOracle'])].values


# In[ ]:





# ## Execution de l'algorithme de partitionnement d'attribut par clustering (K-mean) avec Truth Finder / DEPEN / ACCU ou ACCUSIM sur les données
# Notre proposition

# In[11]:


get_ipython().run_cell_magic('time', '', 'aac.train(dataframe,initial_trustworthiness=0.8,\n                  max_iterations=10,max_iter_part_check=1,\n                  threshold=1e-06,algo="TF",\n                  data_truth=data_truth,rho=0.5,\n                  afficher=False)')


# In[ ]:





# In[ ]:





# In[16]:


out_aac[out_aac['AccuracyMax'] == max(out_aac['AccuracyMax'])].values


# In[17]:


out_aac[out_aac['AccuracyAvg'] == max(out_aac['AccuracyAvg'])].values


# In[18]:


out_aac[out_aac['AccuracyOracle'] == max(out_aac['AccuracyOracle'])].values


# In[ ]:





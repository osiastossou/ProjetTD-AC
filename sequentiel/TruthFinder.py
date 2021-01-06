from utils import utils 
import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
np.seterr(divide='ignore', invalid='ignore') 

import warnings
warnings.filterwarnings("error")

class TruthFinder:
    
    def __init__(self,rho=0.5,theta=0.1,max_iterations=5,initial_trustworthiness=0.8,threshold=1e-6):
        self.dataframe = pd.DataFrame()
        self.resultat = None
        self.rho = rho
        self.theta = theta
        self.max_iterations = max_iterations
        self.initial_trustworthiness = initial_trustworthiness
        self.threshold = threshold
        self.dict_sim = None

        self.max = None
        self.mean = None

    def calculate_confidence(self,df):
        trustworthiness_score = lambda x: -math.log(1-x)  # Eq. 3

        """Calculate confidence for each Value"""
        z = 0
        
        for key, value_of_di in df[['Object','Property','Value']].drop_duplicates().groupby(by=['Object','Property']):
            
            start_time = time.time()
            
            row = key[0]+key[1] # ObjectProperty
            value = list(value_of_di["Value"])
            conf_deja = {
                'value':[],
                'conf':[],
                'adjust':[]
            }
            
            for u in range(len(value)):
                # Eq. 5
                # trustworthiness of corresponding websites `W(f)`
                ts = df.loc[((df["Value"] == value[u]) & (df["ObjectProperty"]==row)), "trustworthiness"]
                
                v = sum(trustworthiness_score(t) for t in ts)
                #indeces = df[(df["ObjectProperty"]==row) & (df["Value"]==value[u])].index
                conf_deja['value'].append(value[u])
                conf_deja['conf'].append(v)
                conf_deja['adjust'].append(v)
            
                
                
                if u>0:
                    for k in range(u):
                        # Avant
                        # dernier = u
                        sim = utils.implication(conf_deja['value'][k],conf_deja['value'][u],row,self.dict_sim)
                        conf_deja['adjust'][k] = conf_deja['adjust'][k] + self.rho*conf_deja['conf'][u]*sim
                        
                        # dernier
                        # sim = implication(conf_deja['value'][u],conf_deja['value'][k],row,dict_sim)
                        conf_deja['adjust'][u] = conf_deja['adjust'][u] + self.rho*conf_deja['conf'][k]*sim
                
            for p in range(len(conf_deja['value'])):
                
                indeces_ = (df["ObjectProperty"]==row) & (df["Value"]==conf_deja['value'][p])
                
                df.loc[indeces_, 'Value_confidence'] = utils.sigmoid(self.theta * conf_deja['adjust'][p])
                
            z+=1

        return df
        
    def update_source_trustworthiness(self,df):
        for source in df["Source"].unique():
            indices = df["Source"] == source
            cs = df.loc[indices, "Value_confidence"]
            #print(source,cs)
            #print(source,cs)
            df.loc[indices, "oldtrustworthiness"] = df.loc[indices, "trustworthiness"]
            t = sum(cs) / len(cs)
            
            if t >=1:
                t = 1-0.0001
            elif t <= 0:
                t = 0 + 0.0001
            df.loc[indices, "trustworthiness"] = t
        return df


    def iteration(self,df):
        df = self.calculate_confidence(df)
        df = self.update_source_trustworthiness(df)
        return df


    def train(self,dataframe):

        self.dataframe = dataframe
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * self.initial_trustworthiness
        dataframe["oldtrustworthiness"] = np.ones(len(dataframe.index)) * self.initial_trustworthiness
        dataframe["Value_confidence"] = np.zeros(len(dataframe.index))
        dataframe["Value_confidence_adjust"] = np.zeros(len(dataframe.index))
        dataframe["Object"]=[str(x) for x in dataframe["Object"]]
        dataframe["Property"]=[str(x) for x in dataframe["Property"]]
        dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]

        
        self.dict_sim = utils.similarity(dataframe)

        k = 0
        for i in range(self.max_iterations):

            t1 = dataframe.drop_duplicates("Source")["trustworthiness"]
            t1old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

            dataframe = self.iteration(dataframe)

            t2 = dataframe.drop_duplicates("Source")["trustworthiness"]
            t2old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]
            
            if utils.stop_condition(t1, t2, t1old , t2old, self.threshold):
                break
            k = i+1
        print("Nombre d'itération : ",k)

        out_data = dataframe.drop("ObjectProperty",axis=1)

        self.resultat = utils.get_result(out_data)

        self.max = max(out_data.drop_duplicates("Source")["trustworthiness"])
        self.mean = out_data.drop_duplicates("Source")["trustworthiness"].mean()
        return self.resultat

    def evaluation(self,data_truth):
        if self.dataframe.shape[0] == 0:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None

        data_truth1 = utils.get_truth_to_dict(data_truth)
        #resultat = utils.get_truth_to_dict(self.resultat)
        evaluation_r = utils.evaluation(data_truth1,self.resultat,self.dataframe)
        return evaluation_r
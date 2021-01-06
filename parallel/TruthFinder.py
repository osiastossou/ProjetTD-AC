from utils import utils 
import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def calculate_confidence(self,df_data):
        trustworthiness_score = lambda x: -math.log(1-x)  # Eq. 3

        """Calculate confidence for each Value"""
        #('Object1', 'Property5', '234041', 'Source1', 0.8, 0.0, 0.8)
        def reduceByKey1(x,y):
            return utils.Union(x, y) #Union(x[0], y[0]),Union(x[1], y[1])

        df_trandform = df_data.map( lambda x:( (x[0],x[1],x[2],x[5]),  [(x[3],x[4])] ) ).reduceByKey(reduceByKey1)
        df_trandform = df_trandform.map( lambda x:( (x[0][0],x[0][1]),  [ ( (x[0][2],x[0][3]),x[1] ) ] ) ).reduceByKey(lambda x,y: x+y)
        #[(('Object1', 'Property5'),
            #[(('234041', 0.0), [('Source9', 0.8), ('Source1', 0.8), ('Source5', 0.8)]),
            #(('613099', 0.0), [('Source2', 0.8)]),
            #(('807969', 0.0), [('Source6', 0.8)]),
            #(('778235', 0.0), [('Source4', 0.8), ('Source10', 0.8)]),
            #(('594299', 0.0), [('Source3', 0.8)]),
            #(('801575', 0.0), [('Source7', 0.8)]),
            #(('839346', 0.0), [('Source8', 0.8)])])]

        dict_sim = self.dict_sim
        rho = self.rho
        theta = self.theta
        def mapComputeValueConf(data):
    
            listValueConf = []
            for value in data[1]:
                conf_v = 0
                for ts in value[1]:
                    conf_v += trustworthiness_score(ts[1])
                conf_v_ = 0
                for value_ in data[1]:
                    if value!=value_:
                        conf_v_ +=value_[0][1]*dict_sim[data[0]][(value[0][0],value_[0][0])]
                conf_v_ = rho*conf_v_ + conf_v

                valueConf = utils.sigmoid(theta*conf_v_)
                listValueConf.append((value[0][0],valueConf))
            return data[0],dict(listValueConf)
        

        resultats = dict(df_trandform.map(mapComputeValueConf).collect())
        
        # [('Object1', 'Property5', '234041', 'Source1', 0.8, 0.0)]
        df_out = df_data.map( lambda x: ( x[0],x[1],x[2],x[3], x[4] , resultats[(x[0],x[1])][x[2]] , x[6] ) ) 
        
        return df_out
        
    def update_source_trustworthiness(self,dataframe):
        df_per_source = dataframe.map(lambda x: (x[3],[(x[0],x[1],x[2],x[4],x[5])])).reduceByKey(lambda x,y: x+y)
        
        def mapComputeTrustworthiness(data):
            tws = 0
            for row in data[1]:
                #math.exp( round(r[1]))
                tws += row[4]
            t = tws / len(data[1])
            if t >=1:
                t = 1-0.0001
            elif t <= 0:
                t = 0 + 0.0001
            return data[0],t
        
        resultats = dict(df_per_source.map(mapComputeTrustworthiness).collect())
        df_out = dataframe.map( lambda x: ( x[0],x[1],x[2],x[3], resultats[x[3]], x[5] , x[4] ) ) 
        return df_out


    def iteration(self,df):
        df = self.calculate_confidence(df)
        df = self.update_source_trustworthiness(df)
        return df


    def train(self,dataframe):

        self.dataframe = dataframe
        
        self.dict_sim = dict(utils.similarity_p(dataframe).collect())
        
        k = 0
        for i in range(self.max_iterations):
            
            t1 = dataframe.map(lambda x: (x[3],x[4])).distinct().map(lambda x: x[1]).collect()  # A voir
            t1old = dataframe.map(lambda x: (x[3],x[6])).distinct().map(lambda x: x[1]).collect()  # A voir

            dataframe = self.iteration(dataframe)

            t2 = dataframe.map(lambda x: (x[3],x[4])).distinct().map(lambda x: x[1]).collect()  # A voir
            t2old = dataframe.map(lambda x: (x[3],x[6])).distinct().map(lambda x: x[1]).collect()  # A voir

            k=i+1
            if utils.stop_condition(t1, t2, t1old , t2old, self.threshold):
                break
        
        print("Nombre d'itération : ",k)

        def reduceVote(x,y):
            if x[1] > y[1]:
                return x
            return y

        df_tmp = dataframe.map(lambda x: ((x[0],x[1],x[2]),x[5])).distinct()

        df_tmp_rdd = utils.getResult(df_tmp)

        self.resultat = df_tmp_rdd

        #print(self.resultat.take(2))

        return self.resultat

    def evaluation(self,data_truth):
        if self.dataframe == None:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None
    
        evaluation_r = utils.evaluation_p(data_truth,self.resultat,self.dataframe)
        return evaluation_r
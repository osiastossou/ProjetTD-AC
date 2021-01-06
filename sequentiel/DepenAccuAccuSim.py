from utils import utils 
import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sequentiel.MajorityVoting import MajorityVoting
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
np.seterr(divide='ignore', invalid='ignore') 

import warnings
warnings.filterwarnings("error")

class DepenAccuAccuSim:
    
    def __init__(self,rho=0.5,theta=0.1,max_iterations=5,initial_trustworthiness=0.8,threshold=1e-6,c=0.3,n=100,alpha = 0.5,algo="DEPEN"):
        self.dataframe = pd.DataFrame()
        self.resultat = None
        self.rho = rho
        self.theta = theta
        self.max_iterations = max_iterations
        self.initial_trustworthiness = initial_trustworthiness
        self.threshold = threshold
        self.c=c
        self.n=n
        self.alpha = alpha
        self.algo=alpha
        self.df_sim = None


        self.max = None
        self.mean = None


    def get_kt_kf_kd(self,dataframe,vote_truth,source1,source2):

        df = dataframe[['ObjectProperty','Source','Value']].groupby(by=['Source'])
        claim_source1 = df.get_group(source1).set_index('ObjectProperty')
        claim_source2 = df.get_group(source2).set_index('ObjectProperty')
        
        kt,kf,kd = 0,0,0
        for row,value in vote_truth.items():
            claim_source11 = claim_source1['Value']
            claim_source22 = claim_source2['Value']
            #print(value1,value2)
            #value = values[2]
            #print(row)
            if( (row in  claim_source11) and (row in  claim_source22)):
                value1 = claim_source11[row]
                value2 = claim_source22[row]
                if(str(value1) == str(value) and str(value2) == str(value)):
                    kt = kt +1
                elif(str(value1) != str(value) and str(value2) != str(value)):
                    kf = kf +1
                else:
                    kd = kd +1
        return kt,kf,kd



    def compDepen(self,dataframe,vote_truth,source1,source2,error):
        kt,kf,kd = self.get_kt_kf_kd(dataframe,vote_truth,source1,source2)
        return 1/( 1 + ((1-self.alpha)/self.alpha)*
                ( ((1-error)/(1-error+self.c*error))**kt)*
                ( (error/(self.c*self.n+error-self.c*error))**kf)*
                ( int(1/(1-self.c))**kd) )

    def compAllDepen(self,dataframe,vote_truth):
        #dict_depen = pd.DataFrame(columns=['Source1','Source2','Depen'])
        dict_depen = { # Dictionnaire de dictionnaire

        }
        sources = dataframe[["Source","trustworthiness"]].drop_duplicates().values
        for line in sources:
            depen = 0
            error = 1-line[1]
            #error = 1-dataframe.loc[dataframe["Source"] == source, "trustworthiness"].values[0]
            dict_depen_one_sources = {}
            for line1 in sources:
                #if source!=source1:
                depen=self.compDepen(dataframe,vote_truth,line[0],line1[0],error)
                dict_depen_one_sources[line1[0]]=depen

            dict_depen[line[0]] = dict_depen_one_sources
        return dict_depen 



    def orderSourceByDepen(self,sources,dict_depen):
        '''
        sources : liste des sources dont on veux faire l'ordre
        retourne une liste de source dans l'ordre decroisante
        '''

        df_sourceOrderByDepen = {}
        #df_sourceOrderByDepen = pd.DataFrame(columns=['Source','Depen'])
        for s in sources:
            df_sourceOrderByDepen[s]=[s,max(dict_depen[s].values())]
            
        return sorted(df_sourceOrderByDepen.items(), key=lambda x: x[1]) # Renvoie une liste de couple (source,depen)





    #En cours
    def calculate_confidence(self,dataframe,dict_depen):
        """Calculate confidence for each Value"""
        #print(dict_depen)

        z = 0
        #for o_p in dataframe["ObjectProperty"].unique():
        for key, df_value in dataframe.groupby(by=['ObjectProperty']):
            

            #df_value = dataframe[dataframe["ObjectProperty"] == o_p]
            conf_deja = {
                'value':[],
                'conf':[],
                'adjust':[]
            }
            values = df_value["Value"].unique()
            for u  in range(len(values)):
                value = values[u]
                listSources = list(df_value[df_value["Value"] == value]['Source']) # Liste des sources qui ont emisent cette valeurs
                ordreListSources = self.orderSourceByDepen(listSources,dict_depen) # Ordornement
                pre = []
                valueConfidence = 0
                tScore = 1
                voteCount = 0
                for source_depen in ordreListSources:
                    source = source_depen[0]
                    if self.algo!="DEPEN":
                        ts = df_value[df_value['Source']==source][['trustworthiness']].values[0]
                        if ts<=0:
                            ts = 0.001
                        tScore = math.log(self.n*ts/(1-ts))
                    if len(pre) == 0:
                        voteCount = 1
                    else:
                        voteCount = 1
                        depen = dict_depen[source]
                        for s in pre:
                            voteCount *= (1 - self.c*depen[s])
                    pre.append(source)
                    valueConfidence += tScore*voteCount

                conf_deja['value'].append(value)
                conf_deja['conf'].append(valueConfidence)
                conf_deja['adjust'].append(valueConfidence)

                if self.algo=="ACCUSIM":
                    
                    if u>0:
                        for k in range(u):
                            # Avant
                            # dernier = u
                            # 
                            sim = utils.implication(conf_deja['value'][k],conf_deja['value'][u],key,df_sim)
                            conf_deja['adjust'][k] = conf_deja['adjust'][k] + rho*conf_deja['conf'][u]*sim
                            
                            # dernier
                            # sim = implication(conf_deja['value'][u],conf_deja['value'][k],row,dict_sim)
                            conf_deja['adjust'][u] = conf_deja['adjust'][u] + rho*conf_deja['conf'][k]*sim
                # else:
                #     indices = dataframe["Value"] == value
                #     dataframe.loc[indices, "Value_confidence"] = valueConfidence
            
            #if algo=="ACCUSIM":
            for p in range(len(conf_deja['value'])):
                
                indeces_ = (dataframe["ObjectProperty"]==key) & (dataframe["Value"]==conf_deja['value'][p])
                
                dataframe.loc[indeces_, 'Value_confidence'] =  conf_deja['adjust'][p]
            

        return dataframe



    def update_source_trustworthiness(self,dataframe):
        #for source in dataframe["Source"].unique():
        df_per_data_item = dataframe.groupby(by=['ObjectProperty'])
        for key, df_value in dataframe.groupby(by=['Source']):
            
            tws = 0
            for i, row in df_value.iterrows():
                data_item = df_per_data_item.get_group(row['ObjectProperty'])[['Value','Value_confidence']].drop_duplicates()
                #print(data_item['Value_confidence'])
                divise = sum([math.exp(r)  for r in data_item['Value_confidence']]) 
                #divise = sum([math.exp(tf.sigmoid(r))  for r in data_item['Value_confidence']]) 
                # for u, row1 in data_item.iterrows():
                #     valueOfDataItem = row1['Value_confidence']
                #     divise+=math.exp(valueOfDataItem)
                tws += math.exp(row['Value_confidence'])/divise # A revoir
                #tws += math.exp(tf.sigmoid(row['Value_confidence']))/divise # A revoir
            indices = dataframe["Source"] == key
            dataframe.loc[indices, "oldtrustworthiness"] = dataframe.loc[indices, "trustworthiness"]

            t = tws / len(df_value)
            if t >=1:
                t = 1-0.0001
            elif t <= 0:
                t = 0 + 0.0001
            dataframe.loc[indices, "trustworthiness"] = t
        return dataframe



    def iteration(self,df,dict_depen):
        df = self.calculate_confidence(df,dict_depen)
        df = self.update_source_trustworthiness(df)
        return df


    def train(self,dataframe):

        self.dataframe = dataframe
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * self.initial_trustworthiness
        dataframe["oldtrustworthiness"] = np.ones(len(dataframe.index)) * self.initial_trustworthiness
        dataframe["Value_confidence"] = np.zeros(len(dataframe.index))
        dataframe["Object"]=[str(x) for x in dataframe["Object"]]
        dataframe["Property"]=[str(x) for x in dataframe["Property"]]
        dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
        
        if self.algo=='ACCUSIM':
            self.df_sim = utils.similarity(dataframe)
        
        mv = MajorityVoting()
        vote_truth = utils.get_truth_to_dict(mv.train(dataframe)) # Dict
        
        dict_depen = self.compAllDepen(dataframe,vote_truth)
        k = 0
        for i in range(self.max_iterations):
            
            t1 = dataframe.drop_duplicates("Source")["trustworthiness"]
            t1old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

            dataframe = self.iteration(dataframe,dict_depen)

            t2 = dataframe.drop_duplicates("Source")["trustworthiness"]
            t2old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

            if utils.stop_condition(t1, t2, t1old , t2old, self.threshold):
                k = i+1
                break
            
            truth_compute = utils.get_result(dataframe)
            if vote_truth!=truth_compute:
                dict_depen = self.compAllDepen(dataframe,truth_compute)

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
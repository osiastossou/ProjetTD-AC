from utils import utils 
import numpy as np
import pandas as pd
import time
from numpy.linalg import norm
import math
from sklearn.feature_extraction.text import TfidfVectorizer

from sequentiel.DepenAccuAccuSim import DepenAccuAccuSim

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
np.seterr(divide='ignore', invalid='ignore') 

import warnings
warnings.filterwarnings("error")

class AccuGenPartition:
    
    def __init__(self,nbr_partition_explorer=None,algo=DepenAccuAccuSim(algo="DEPEN"),file_name='logs/AccuGenPartion_Out_'+ str(int(time.time())) +'.csv'):
        self.dataframe = pd.DataFrame()
        self.resultat = None
        self.nbr_partition_explorer = nbr_partition_explorer
        self.file_name = file_name
        self.algo = algo

    def get_partition_of_data(self,attribute,dataframe):
        indices = []
        for i,row in dataframe.iterrows():
            if row['Property'] in attribute:
                indices.append(i)
        return dataframe.loc[indices]

    def train(self,dataframe,data_truth=pd.DataFrame()):
        
        dataframe["Source"]=[str(x) for x in dataframe["Source"]]
        attribute = list(dataframe["Property"].unique())
        data = pd.DataFrame(columns=['Partition','AccuracyMax','AccuracyAvg','AccuracyOracle','precision','recall','accuracy','f1_score'])
        
        data_truth = utils.get_truth_to_dict(data_truth)

        nbr_partition_deja = 0 
        
        f = open(self.file_name, "w+")
        nbr_fait = len(f.read().split('\n'))-1
        for p in list(utils.partition(attribute)):
            nbr_partition_deja +=1
            
            if nbr_partition_deja <= nbr_fait:
                continue
            else:
                max_ = 0
                avg_ = 0
    
                evaluation_r = None
                out_fusion = {}
                for attribut in p:
                    data_attribute = self.get_partition_of_data(attribut,dataframe)
                    
                    out = self.algo.train(data_attribute)
                    
                    out_fusion.update(out)
                    if(max_ < self.algo.mean):
                        max_ = self.algo.mean
                    avg_ += self.algo.mean
                if len(data_truth)!=0:
                    
                    resultat = out_fusion
                    evaluation_r =  utils.evaluation(data_truth,out_fusion,data_attribute)
                
                self.file_result(self.file_name,[sorted(p),max_,avg_/len(p),evaluation_r['precision'],evaluation_r['precision'],evaluation_r['recall'],evaluation_r['accuracy'],evaluation_r['f1_score']])
                data.loc[len(data)] = [sorted(p),max_,avg_/len(p),evaluation_r['precision'],evaluation_r['precision'],evaluation_r['recall'],evaluation_r['accuracy'],evaluation_r['f1_score']]
                if self.nbr_partition_explorer!=None and self.nbr_partition_explorer==nbr_partition_deja:
                    break
        return data


    def file_result(self,file_name,valeur):
        f = open(file_name, "a")
        f.write(str(valeur[0])+';'+str(valeur[1])+';'+str(valeur[2])+';'+str(valeur[3])+';'+str(valeur[4])+';'+str(valeur[5])+';'+str(valeur[6])+';'+str(valeur[7])+'\n')
        f.close()
    
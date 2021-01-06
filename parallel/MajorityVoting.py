import pandas as pd
from utils import utils 


class MajorityVoting:
    def __init__(self):
        self.dataframe = None
        self.resultat = None

    # Majority vote 
    def train(self,dataframe):
        self.dataframe = dataframe

        def reduceCount(x,y):
            return x+y
        
        df_tmp = self.dataframe.map(lambda x: ((x[0],x[1],x[2]),1)).reduceByKey(reduceCount)
        df_tmp_rdd = utils.getResult(df_tmp)

        self.resultat = df_tmp_rdd

        return self.resultat

    def evaluation(self,data_truth):
        if self.dataframe == None:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None
    
        evaluation_r = utils.evaluation_p(data_truth,self.resultat,self.dataframe)
        return evaluation_r
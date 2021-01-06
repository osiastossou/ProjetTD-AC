import pandas as pd
from utils import utils 


class MajorityVoting:
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.resultat = None

    # Majority vote 
    def train(self,dataframe):
        self.dataframe = dataframe
        
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
            self.resultat = resultat
        
        return resultat

    def evaluation(self,data_truth):
        if self.dataframe.shape[0] == 0:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None
        
        data_truth1 = utils.get_truth_to_dict(data_truth)
        resultat = utils.get_truth_to_dict(self.resultat)
        evaluation_r = utils.evaluation(data_truth1,resultat,self.dataframe)
        return evaluation_r
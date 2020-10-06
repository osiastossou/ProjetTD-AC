
import pandas as pd
# Contenant l'algorithme du Truth Finder
import TruthFinder as tf 


# Majority vote 
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



# Compute the covrage of data
def get_cavrage(dataframe):
    total = 0
    manque = 0
    for o, df_object in dataframe.groupby(by=['Object']):
        nbr_source = len(df_object["Source"].unique())
        nbr_property = len(df_object["Property"].unique())
        nbr_data = df_object.shape[0]
        nbr_totale = nbr_source*nbr_property
        total +=nbr_totale
        manque += 1*(nbr_totale-nbr_data)
    return (1-manque/total)
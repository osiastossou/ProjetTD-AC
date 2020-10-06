import numpy as np
from numpy.linalg import norm
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import TruthFinder as tf
import DepenAccuAccuSim as daas 


def get_partition_of_data(attribute,dataframe):
    indices = []
    for i,row in dataframe.iterrows():
        if row['Property'] in attribute:
            indices.append(i)
    indices
    return dataframe.loc[indices]

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def train(dataframe,max_iterations=5,
          threshold=1e-6,nbr_partition_explorer=None,initial_trustworthiness=0.8,rho=0.5,theta = 0.1,afficher=False,algo="TF",data_truth=pd.DataFrame()):
    dataframe["Source"]=[str(x) for x in dataframe["Source"]]
    attribute = list(dataframe["Property"].unique())
    data = pd.DataFrame(columns=['Partition','AccuracyMax','AccuracyAvg','AccuracyOracle','precision','recall','accuracy','f1_score'])
    
    data_truth = tf.get_truth_to_dict(data_truth)

    nbr_partition_deja = 0 
    f = open('AccuGenPartion_Start_Dimanche_13_09_DS2.txt', "r")
    nbr_fait = len(f.read().split('\n'))-1
    for p in list(partition(attribute)):
        nbr_partition_deja +=1
        
        if nbr_partition_deja <= nbr_fait:
            print(nbr_partition_deja,end=" - ")
            continue
        else:
            max_ = 0
            avg_ = 0
            out_fusion = pd.DataFrame()
            
            evaluation_r = None
            for attribut in p:
                data_attribute = get_partition_of_data(attribut,dataframe)
                out = None
                if (algo=="TF"):
                    out = tf.train(data_attribute,max_iterations=max_iterations,threshold=threshold, initial_trustworthiness=initial_trustworthiness,rho=rho,theta = theta,afficher=afficher,data_truth=pd.DataFrame())
                elif(algo=="ACCU" or algo=="DEPEN" or algo=="ACCUSIM"):
                    out = daas.train(data_attribute, max_iterations=max_iterations,threshold=threshold, initial_trustworthiness=initial_trustworthiness,c=0.2,n=50,alpha=0.2,rho=rho,afficher=afficher,algo=algo,data_truth=pd.DataFrame())

                out_fusion = pd.concat([out_fusion,out[0]]) 
                if(max_ < out[3]):
                    max_ = out[3]
                avg_ += out[3]
            if len(data_truth)!=0:
                
                resultat = tf.get_result(out_fusion)
                evaluation_r = tf.evaluation(data_truth,resultat,dataframe)
            #print(sorted(p),max_,avg_/len(p),oracle)
            file_result("AccuGenPartion_Start_Dimanche_13_09_DS2.txt",[sorted(p),max_,avg_/len(p),evaluation_r['precision'],evaluation_r['precision'],evaluation_r['recall'],evaluation_r['accuracy'],evaluation_r['f1_score']])
            data.loc[len(data)] = [sorted(p),max_,avg_/len(p),evaluation_r['precision'],evaluation_r['precision'],evaluation_r['recall'],evaluation_r['accuracy'],evaluation_r['f1_score']]
            if nbr_partition_explorer!=None and nbr_partition_explorer==nbr_partition_deja:
                break
    return data


def file_result(file_name,valeur):
    f = open(file_name, "a")
    f.write(str(valeur[0])+';'+str(valeur[1])+';'+str(valeur[2])+';'+str(valeur[3])+';'+str(valeur[4])+';'+str(valeur[5])+';'+str(valeur[6])+';'+str(valeur[7])+'\n')
    f.close()
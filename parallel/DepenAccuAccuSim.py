from utils import utils 
import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import time

import itertools
import random

from parallel.MajorityVoting import MajorityVoting
np.seterr(divide='ignore', invalid='ignore') 

import warnings
warnings.filterwarnings("error")

class DepenAccuAccuSim:
    
    def __init__(self,rho=0.5,theta=0.1,max_iterations=5,initial_trustworthiness=0.8,threshold=1e-6,c=0.3,n=100,alpha = 0.5,algo="DEPEN",sc = None):
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
        self.sc = sc


        self.max = None
        self.mean = None


    # Fonction de transformation des données
    def transform(self,df_data,truth):
        '''
        df_data : les données brute
        truth : les données vrai par data item (Elle peux venir du vote)
        
        Retourne une RDD dont chaque ligne est sous la format : 
                    (
                        ('Object', 'Property', 'ValeurVrai'),
                        [
                            ('ValeurSource8', 'Source8', 0.8),
                            ('ValeurSource9', 'Source9', 0.8)
                        ]
                    )
        
        '''
        def reduceD(x,y):
            return x+y
        
        # Cette jointure est faire pour ajouter les données vrai à coté
        t = df_data.map(lambda r:((r[0],r[1]),(r[2],r[3],r[4]) ))
        v = truth.map(lambda r:((r[0],r[1]),r[2]))
        data_join = t.join(v)

        # Application de mapReduce pour faire la transformation et obtenur la sortir
        data_transform = data_join.map(lambda r:( (r[0][0],r[0][1],r[1][1]), [(r[1][0][0],r[1][0][1],r[1][0][2])] )).reduceByKey(reduceD)
        return data_transform

# Application du transform
#transform(df_data,df_data_vote).take(1)
    # Application du transform
    #transform(df_data,df_data_vote).take(1)

    # Fonction Map pour les différentes count. Calcule de kt, kf, kd


    def compDepen(self,dataframe,vote_truth):


        # ---------------
        def get_kt_kf_kd(data_line):
            '''
            data_line : parametre d'entrer de la fonction, cest un tuple de deux élément.
            le premier élément contient le data item et sa valeur
            le deuxième contient les valeurs données par des sources pour ce data item.
                Exemple : (
                            ('Object', 'Property', 'ValeurVrai'),
                            [
                                ('ValeurSource8', 'Source8'),
                                ('ValeurSource9', 'Source9')
                            ]
                        )
                        Retourn : 
                        <(Source8,Source9),(1,0,0)>
            
            kt : nombre de même valeur trouver ensemble par les sources
            kf : nombre de même valeur fausse ensemble par les sources
            kd : nombre de différente valeur donnée par les sources
            '''
            # Get the truth value of the line data item 
            truth_value = data_line[0][2]
            
            from functools import reduce
            # Extraire toutes les sources qui apparait dans l'entrée
            all_sources = map(lambda x: [(x[1],x[2])],data_line[1])
            all_sources = reduce(lambda x,y: x+y,all_sources)
            
            #print(all_sources)
        
            # Construitre les pairs de sources
            all_pair_sources = itertools.product(all_sources,all_sources)
            
            retour = []
            for sources in list(all_pair_sources):
                if sources[0]!=sources[1]:
                    value_source1,value_source2 = None,None
                    kt,kf,kd = 0,0,0
                    for line in data_line[1]:
                        if sources[0]==line[1]:
                            value_source1 = line[0]
                        elif sources[1]==line[1]:
                            value_source2 = line[0] 

                    if value_source1 == value_source2:
                        #print(value_source1)
                        if value_source1 == truth_value:
                            kt = 1
                        else:
                            kf = 1
                    else:
                        kd = 1
                    retour.append((sources,(kt,kf,kd)))
            
            return retour
        # ---------------
        
        alpha=self.alpha
        n=self.n
        c=self.c
        sc = self.sc
        def compDepenOfTwoSource(line,alpha=alpha,n=n,c=c):
            error = 1-line[0][0][1]
            kt,kf,kd = line[1]
            return 1/( 1 + ((1-alpha)/alpha)*
                    (((1-error)/(1-error+c*error))**kt)*
                    ((error/(c*n+error-c*error))**kf)*
                    ((1/(1-c))**kd))
        
        
        # Etape 1 : Fransformation des données
        data_transforme = self.transform(dataframe,vote_truth) 
        
        #print(data_transforme.take(2))
        # Etape 2 : Calcule des kt, kf, kd
        g = data_transforme.map(get_kt_kf_kd).reduce(lambda x,y: x+y)
        #((('Source4', 0.8), ('Source3', 0.8)), (0, 1, 0))
        kt_kf_kds = sc.parallelize(g).reduceByKey(lambda x,y:tuple(map(sum, zip(x, y))))
        
        # Etape 3 : Calcul de la dépendance de chaque avec la fonction "compDepenOfTwoSource"
        #((('Source2', 0.8), ('Source10', 0.8)), (0, 6000, 0))
        #retour = kt_kf_kds.map(lambda x: ((x[0][0][0],x[0][1][0]), compDepenOfTwoSource(x,alpha,n,c)))
        
        def reduceDict(x,y):
            x.update(y)
            return x

        #print(kt_kf_kds.take(1))
        #( (('Source2', 0.8), ('Source10', 0.8)), (0, 6000, 0) )
        retour = kt_kf_kds.map(lambda x: (x[0][0][0],{x[0][1][0]:compDepenOfTwoSource(x,alpha,n,c)})).reduceByKey(reduceDict)
        
        return retour
    # Fonction de calcule de toutes les dépendences


    


    def calculate_confidence(self,dataframe,dict_depen):
        """ Calculate confidence for each Value """

        def orderSourceByDepen(sources,dict_depen):
            '''
            sources : liste des sources dont on veux faire l'ordre
            retourne une liste de source dans l'ordre decroisante
            '''

            df_sourceOrderByDepen = {}
            #df_sourceOrderByDepen = pd.DataFrame(columns=['Source','Depen'])
            for s in sources:
                df_sourceOrderByDepen[s]=[s,max(dict_depen[s[0]].values())]
                
            return sorted(df_sourceOrderByDepen.items(), key=lambda x: x[1]) # Renvoie une liste de couple (source,depen)
        
        #  [('Object1', 'Property5', '234041', 'Source1', 0.8, 0.0, 0.8)]
        df_data = dataframe.map(lambda x: ((x[0],x[1]),[(x[2],x[3],x[4],x[5],x[6])])).reduceByKey(lambda x,y: x+y)
        
        dict_depen1 = dict(dict_depen.collect())
        # Définition de la fonction map
        data_sim = dict(utils.similarity_p(dataframe).collect())
        c = self.c
        n = self.n
        rho= self.rho
        algo= self.algo
        theta = self.theta
        def mapComputeValueConf(data):
            
            rddValueDataItemTmp = map(lambda x: ((x[0],x[3]),[(x[1],x[2],x[4])]),data[1])
            
            rddValueDataItem = []
            dIDeja = []
            for dI in rddValueDataItemTmp:
                if dI[0] in dIDeja:
                    index = dIDeja.index(dI[0])
                    rddValueDataItem[index][1].append(dI[1][0])
                else:
                    dIDeja.append(dI[0])
                    rddValueDataItem.append( (dI[0],dI[1]) )
                
            
            listValueConf = []
            for line in rddValueDataItem:
                sources = [(i[0],i[1]) for i in line[1]]
                sourcesOrdering = orderSourceByDepen(sources,dict_depen1)
                pre = []
                valueConf = 0
                tScore = 1
                voteCount = 0
                for source in sourcesOrdering:
                    if algo!="DEPEN":
                        ts = source[0][1]
                        if ts<=0:
                            ts = 0.001
                        tScore = math.log(n*ts/(1-ts))
                    if len(pre) == 0:
                        voteCount = 1
                    else:
                        voteCount = 1
                        depen = dict_depen1[source[0][0]]
                        for s_j in pre:
                            # [(('Source2', 'Source10'), 1.0)]
                            depenSourceS_j = depen[s_j] #dict_depen.filter(lambda r: r[0][0] == source and r[0][1] == s_j).collect()[0][1]
                            voteCount = voteCount * (1 - c*depenSourceS_j)
                    pre.append(source[0][0])
                    valueConf = valueConf + tScore * voteCount
                    
                    if algo=="ACCUSIM":
                        tmp_conf = 0
                        for value_d in rddValueDataItem:
                            if line[0][0]!=value_d[0][0]:
                                tmp_conf = tmp_conf + value_d[0][1]*data_sim[data[0]][(line[0][0],value_d[0][0])]
                        valueConf = valueConf+ rho*tmp_conf # rho*sigmoid(tmp_conf)
                listValueConf.append((line[0][0],valueConf))
            
            return data[0],dict(listValueConf)
            
            
        resultats = dict(df_data.map(mapComputeValueConf).collect())
        
        
        # [('Object1', 'Property5', '234041', 'Source1', 0.8, 0.0)]
        df_out = dataframe.map( lambda x: ( x[0],x[1],x[2],x[3], x[4] , resultats[(x[0],x[1])][x[2]] , x[6] ) ) 
        
        return df_out


    def update_source_trustworthiness(self,dataframe):
        
        #print(dataframe.take(1))
        
        def reduceDf_per_data_item(x,y):
            if x in y:
                return y
            elif y in x:
                return x
            else:
                return x+y
        
        df_per_data_item = dict(dataframe.map(lambda x: ((x[0],x[1]),[(x[2],x[5])])).reduceByKey(reduceDf_per_data_item).collect()) # Revoire unicité
        df_per_source = dataframe.map(lambda x: (x[3],[(x[0],x[1],x[2],x[4],x[5])])).reduceByKey(lambda x,y: x+y)
        
        def mapComputeTrustworthiness(data):
            tws = 0
            for row in data[1]:
                data_item = df_per_data_item[(row[0],row[1])]
                divise = sum([np.exp(r[1]) for r in data_item]) 
                #math.exp( round(r[1]))
                tws += np.exp(row[4])/divise
            t = tws / len(data[1])
            if t >=1:
                t = 1-0.0001
            elif t <= 0:
                t = 0 + 0.0001
            return data[0],t
        
        resultats = dict(df_per_source.map(mapComputeTrustworthiness).collect())
        df_out = dataframe.map( lambda x: ( x[0],x[1],x[2],x[3], resultats[x[3]], x[5] , x[4] ) ) 
        return df_out


    def iteration(self,df,dict_depen):
        df = self.calculate_confidence(df,dict_depen)
        df = self.update_source_trustworthiness(df)
        return df


    def train(self,dataframe):

        self.dataframe = dataframe
        
        mv = MajorityVoting()
        vote_truth = mv.train(dataframe)
        #print(vote_truth.take(2))
        
        df_depen = self.compDepen(dataframe,vote_truth)
        #print(df_depen.take(2))
        
        def reduceVote(x,y):
            if x[1] > y[1]:
                return x
            return y
        
        k = 0
        for i in range(self.max_iterations):
            t1 = dataframe.map(lambda x: (x[3],x[4])).distinct().map(lambda x: x[1]).collect()  # A voir
            t1old = dataframe.map(lambda x: (x[3],x[6])).distinct().map(lambda x: x[1]).collect()  # A voir

            dataframe = self.iteration(dataframe,df_depen)
            
            # [('Object1', 'Property5', '234041', 'Source1', 0.10689952923986035, 2.19, 0.8)]
            df_tmp = dataframe.map(lambda x: ((x[0],x[1],x[2]),x[5])).distinct()
            
            vote_truth = utils.getResult(df_tmp)
            df_depen = self.compDepen(dataframe,vote_truth)

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

        return self.resultat


    def evaluation(self,data_truth):
        if self.dataframe == None:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None
    
        evaluation_r = utils.evaluation_p(data_truth,self.resultat,self.dataframe)
        return evaluation_r
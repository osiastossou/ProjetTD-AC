from utils import utils 
import numpy as np
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer

from sequentiel.DepenAccuAccuSim import DepenAccuAccuSim

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from sequentiel.MajorityVoting import MajorityVoting

import pandas as pd
from sklearn.cluster import KMeans


warnings.simplefilter("ignore")


class TDAC:
    
    def __init__(self,nbr_partition_explorer=None,algo=DepenAccuAccuSim(algo="DEPEN"),max_iter_part_check=5,file_name='AccuGenPartion_Out_'+ str(time.time()) +'.csv'):
        self.dataframe = pd.DataFrame()
        self.resultat = None
        self.max_iter_part_check=max_iter_part_check
        self.algo = algo

    def get_index(self,l,e):
        u=0
        for i in l:
            if i==e:
                return u
            u=u+1
    def make_matrix(self,data,resultat):
        pro = []
        obj = []
        src , val = [],[]
        #val = [],[]
        for line in data.values:
            if line[2] not in pro:
                pro.append(line[2])
            #i_p = get_index(pro,line[2])
            if len(obj)== 0 or line[1] not in obj:
                obj.append(line[1])
                #sum_obj[i_p]+=1
            if len(src)== 0 or line[4] not in src:
                src.append(line[4])
                #sum_src[i_p]+=1
            if len(val)== 0 or line[3] not in val:
                #line[3]
                val.append(line[3])  
                #sum_val[i_p]+=1
        col = []
        for o in obj:
            for s in src:
                col.append(o+'_os_'+s)

        df = pd.DataFrame(columns = col,index=pro) 
        for line_pro in pro:
            for line in data.values:
                v = 0
                if(line[2]==line_pro):
                    if resultat[line[1]+line_pro]==line[3]:
                        v=1
                    else:
                        v=0
                    df.loc[line_pro][line[1]+'_os_'+line[4]] = v
        #df = df.dropna(axis=1, thresh=2)



        df = df.fillna(0)
        return df


    def get_partition_with_label(self,label_,make_matrix):
        p = {}
        u = 0
        label = list(make_matrix.index)
        for i in label_:
            if i in p:
                p[i].append(label[u])
            else:
                p[i] = [label[u]]
            u+=1
        retour = []
        for part in p.values():
            retour.append(sorted(part))
        return sorted(retour)

    def get_all_partion_with_kmean(self,make_matrix):
        partition = []
        #len(make_matrix.index)+1
        for k in range(2, len(make_matrix.index)+1):
            #print(k)
            kmeans = KMeans(n_clusters=k, max_iter=10).fit(make_matrix)
            make_matrix["clusters"] = kmeans.labels_
            partition.append(get_partition_with_label(list(kmeans.labels_),make_matrix))
        return partition


    def get_best_cluster(self,make_matrix):

        X = make_matrix.values
        range_n_clusters = [i for i in range(2,X.shape[0])]
        k = 0
        silhouette = None
        for n_clusters in range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            silhouette_avg = silhouette_score(X, cluster_labels)
            if n_clusters == 2:
                silhouette = silhouette_avg
                k = n_clusters
            elif silhouette < silhouette_avg:
                k = n_clusters
                silhouette = silhouette_avg
        
        kmeans = KMeans(n_clusters=k, max_iter=10).fit(make_matrix)
        make_matrix["clusters"] = kmeans.labels_
        best_cluster = get_partition_with_label(list(kmeans.labels_),make_matrix)
        return best_cluster



    def train(self,dataframe):
        '''
        /len(get_all_partion_with_kmean)
        '''
        dataframe["Source"]=[str(x) for x in dataframe["Source"]]


        mv = MajorityVoting()
        truth = utils.get_truth_to_dict(mv.train(dataframe)) # Dict

        

        make_matrix_start = None
        partition_start = []

        for i in range(self.max_iter_part_check):

            # Etape 2 : Chercher matrix
            make_matrix_ = self.make_matrix(dataframe,truth)
            

            # Etape 3 : Chercher partition
            get_all_partion_with_kmean = self.get_best_cluster(make_matrix_)

            if i == 0:
                make_matrix_start = make_matrix_
                partition_start = get_all_partion_with_kmean
            else:
                if get_all_partion_with_kmean == partition_start:
                    break
                else:
                    partition_start = get_all_partion_with_kmean
                    X = make_matrix_.values
                    Y = make_matrix_start.values
                    X_ = X.reshape(1, X.shape[0]*X.shape[1])[0]
                    Y_ = Y.reshape(1, Y.shape[0]*Y.shape[1])[0]
                    conver = 1 - np.dot(X_, Y_) / (norm(X_) * norm(Y_))
                    if conver < threshold:
                        break

            
            out_fusion = {}
            for attribut in get_all_partion_with_kmean:

                data_attribute = self.get_partition_of_data(attribut,dataframe)
                out = self.algo.train(data_attribute)
                
                out_fusion.update(out)
                
            
            #out_data = out_fusion
            resultat = out_fusion
            evaluation_r =  utils.evaluation(data_truth,out_fusion,data_attribute)
    
        self.resultat = out_data

        return self.resultat
    
    def evaluation(self,data_truth):
        if self.dataframe.shape[0] == 0:
            print("Vous devez faire l'entrainement avant de faire une évaluation")
            return None

        data_truth1 = utils.get_truth_to_dict(data_truth)
        #resultat = utils.get_truth_to_dict(self.resultat)
        evaluation_r = utils.evaluation(data_truth1,self.resultat,self.dataframe)
        return evaluation_r
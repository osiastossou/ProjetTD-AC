import numpy as np
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import TruthFinder as tf
import AccuGenPartition as agp
from sklearn.cluster import KMeans
import DepenAccuAccuSim as daas
import warnings


warnings.simplefilter("ignore")




def vote(dataframe):
    #dataframe["Object"]=[str(x) for x in dataframe["Object"]]
    #dataframe["Property"]=[str(x) for x in dataframe["Property"]]
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
    resultat = {}
    
    for key, df in dataframe.groupby(by=['ObjectProperty']):
        l = list(df['Value'])
        resultat[key]=max(set(l), key = l.count)
        #print([df.values[0][1],df.values[0][2],max(set(l), key = l.count)])
    return resultat





def get_index(l,e):
    u=0
    for i in l:
        if i==e:
            return u
        u=u+1
def make_matrix(data,resultat):
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


def get_partition_with_label(label_,make_matrix):
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

def get_all_partion_with_kmean(make_matrix):
    partition = []
    #len(make_matrix.index)+1
    for k in range(2, len(make_matrix.index)+1):
        #print(k)
        kmeans = KMeans(n_clusters=k, max_iter=10).fit(make_matrix)
        make_matrix["clusters"] = kmeans.labels_
        partition.append(get_partition_with_label(list(kmeans.labels_),make_matrix))
    return partition


def get_best_cluster(make_matrix):

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



def train(dataframe,max_iterations=10,max_iter_part_check=5,
          threshold=1e-6, initial_trustworthiness=0.8,rho=0.5,theta = 0.1,n=500,afficher=False,algo="TF",data_truth=pd.DataFrame()):
    '''
    /len(get_all_partion_with_kmean)
    '''
    dataframe["Source"]=[str(x) for x in dataframe["Source"]]

    #out1 = daas.train(dataframe, max_iterations=max_iterations,threshold=threshold, initial_trustworthiness=initial_trustworthiness,c=0.2,n=50,alpha = 0.2,rho=rho,afficher=afficher,algo=algo)
    #truth = tf.get_result(out1[0])
    # Etape 1 : Faire le vote
    truth = vote(dataframe)

    if initial_trustworthiness==None:
        dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
        dataframe["trustworthiness"] = np.ones(len(dataframe.index))
        dataframe["oldtrustworthiness"] = np.ones(len(dataframe.index))

        for key, df_value in dataframe.groupby(by=['Source']):
            t = 0
            totale = 0
            for i, row in df_value.iterrows():
                if row['Value']==truth[row["ObjectProperty"]]:
                    t+=1
                totale+=1
            indices = dataframe["Source"] == key
            comput_t = t/totale
            #print(key,comput_t)
            dataframe.loc[indices, "oldtrustworthiness"] = comput_t
            dataframe.loc[indices, "trustworthiness"] = comput_t


    #print(truth)


    evaluation_r = None
    data_truth = tf.get_truth_to_dict(data_truth)
    #out_data = None

    make_matrix_start = None
    partition_start = []

    for i in range(max_iter_part_check):

        start_time = time.time()
        # Etape 2 : Chercher matrix
        make_matrix_ = make_matrix(dataframe,truth)
        

        # Etape 3 : Chercher partition
        get_all_partion_with_kmean = get_best_cluster(make_matrix_)

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
                print('Covergence : ',conver)
                if conver < threshold:
                    break

        
        out_fusion = pd.DataFrame()
        for attribut in get_all_partion_with_kmean:
            data_attribute = agp.get_partition_of_data(attribut,dataframe)
            out = None
            if (algo=="TF"):
                out = tf.train(data_attribute,max_iterations=max_iterations,threshold=threshold, initial_trustworthiness=initial_trustworthiness,rho=rho,theta = theta,afficher=afficher)
            elif(algo=="ACCU" or algo=="DEPEN" or algo=="ACCUSIM"):
                out = daas.train(data_attribute, max_iterations=max_iterations,threshold=threshold, initial_trustworthiness=initial_trustworthiness,c=0.2,n=n,alpha = 0.2,rho=rho,afficher=afficher,algo=algo)
            out_fusion = pd.concat([out_fusion,out[0]]) 
        
        end_time = time.time() 
        #out_data = out_fusion
        truth = tf.get_result(out_fusion)
        evaluation_r = tf.evaluation(data_truth,truth,dataframe)
        print('Iter : ',i+1,': ',get_all_partion_with_kmean,' | ',evaluation_r['precision'],' | Time : ',(end_time-start_time))


    return evaluation_r
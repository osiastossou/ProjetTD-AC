import numpy as np

from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
np.seterr(divide='ignore', invalid='ignore') 
import random

import warnings
warnings.filterwarnings("error")

warnings.simplefilter("ignore")

def get_result(dataframe,seed=10):
    '''
    Renvoie un dictionnaire 
    {
        ObjetProperty: value

    }
    '''
    dataframe["Object"]=[str(x) for x in dataframe["Object"]]
    dataframe["Property"]=[str(x) for x in dataframe["Property"]]
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
    resultat = {}
    #for o_a in dataframe["ObjectProperty"].unique():
    for key, df in dataframe.groupby(by=['ObjectProperty']):
        r = df.loc[(df["Value_confidence"] == max(df["Value_confidence"])),'Value'].values
        if len(r)==1:
            resultat[key]= r[0]
        else:
            random.seed(seed)
            resultat[key]= random.choice(sorted(r))
        
    return resultat

def get_truth_to_dict(data_truth):
    #data_truth["Object"]=[str(x) for x in data_truth["Object"]]
    #data_truth["Property"]=[str(x) for x in data_truth["Property"]]
    data_truth["ObjectProperty"] = data_truth["Object"]+data_truth["Property"]
    truth = {}
    #for o_a in dataframe["ObjectProperty"].unique():
    for row in data_truth[['ObjectProperty','Value']].values:
        truth[row[0]]= row[1]
    #resultat["Value"]=[int(x) for x in resultat["Value"]]
    return truth


def evaluation(truth,find,dataframe):

    #print(find)
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]


    find_,truth_ = [],[]
    for key, data_ in dataframe.groupby(by=['ObjectProperty']):
        value_find = find[key]
        value_truth = truth[key]
        for value in list(data_['Value']):
            if (value == value_find) and (value== value_truth):
                find_.append(1)
                truth_.append(1)
            elif (value != value_find) and (value != value_truth):
                find_.append(0)
                truth_.append(0)
            elif (value == value_find) and (value != value_truth):
                find_.append(1)
                truth_.append(0)
            elif (value != value_find) and (value == value_truth):
                find_.append(0)
                truth_.append(1)
            
    precision = precision_score(truth_,find_) 
    recall = recall_score(truth_,find_) 
    f1_score_ = f1_score(truth_,find_) 
    out = {'precision' : precision,
    'recall' : recall,
    'accuracy' : accuracy_score(truth_,find_),
    'f1_score' : f1_score_}
    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


def similarity(dataframe):
    '''
    Pré-calcule de tout les similarité par attribut d'objet.
    '''
    dict_sim = {
        
    }
    for key, df in dataframe.groupby(by=['Object','Property']):
        Values = df['Value'].unique()
        row = key[0]+key[1]
        for  i in range(len(Values)):
            w1 = Values[i]
            for  u in range(len(Values)):

                w2 = Values[u]
                sim = 1
                if w1!=w2:
                    #df.values[0][3]
                    if( isinstance(w1, int) or isinstance(w1, float) ):
                        #v1, v2 = [w1], [w2]
                        t = abs(w1-w2)
                        sim = 1/t
                    else:
                        # vectorizer = TfidfVectorizer(min_df=1)
                        # vectorizer.fit(df["Value"])
                        # w1, w2 = w1.lower(), w2.lower()
                        # V = vectorizer.transform([w1, w2])
                        # v1, v2 =  np.asarray(V.todense())

                        sim = cosdis(word2vec(str(w1)), word2vec(str(w2)))
                        
                        #sim = np.dot(v1, v2) / (norm(v1) * norm(v2))

                #t = abs(ord(w1)-ord(w2)) 
                #sim = 1
                #if t != 0:
                    #sim = 1/t
                

                #if w1!=w2:
                #sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
                #abs(w1-w2)
                #sim = cosdis(word2vec(str(w1)), word2vec(str(w2)))*100
                
                #np.dot(v1, v2) / (norm(v1) * norm(v2))
                dict_sim[row+str(w1)+str(w2)] = sim
                dict_sim[row+str(w2)+str(w1)] = sim
                #dict_sim['ObjectProperty'].append(df['ObjectProperty'].values[0])
                #dict_sim['w1'].append(w1)
                #dict_sim['w2'].append(w2)
                #dict_sim['sim'].append(sim)
    
    #return pd.DataFrame(dict_sim)
    return dict_sim


def implication(f1,f2,objectProperty,dict_sim):
    #print(f1,f2,objectProperty)
    sim = dict_sim[objectProperty+str(f1)+str(f2)]
    return sim


def stop_condition(t1, t2, t1old , t2old, threshold):
    r = np.dot(t1, t1old) / (norm(t1) * norm(t1old)) - np.dot(t2, t2old) / (norm(t2) * norm(t2old))
    #print('Valeur de convergence actuel',abs(r))
    return abs(r)  < threshold




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




### Parallele

from pyspark.mllib.evaluation import MulticlassMetrics,BinaryClassificationMetrics


def read(file,sc,truth=False):
    read_ = sc.textFile(file)
    if truth == False:
        return read_.map(lambda line: line.split(",")).map(lambda line: (line[0],line[1],line[2],line[3],line[4]))
    else:
        return read_.map(lambda line: line.split(",")).map(lambda line: (line[0],line[1],line[2]))
        #return read_.map(lambda line: line.split(",")).map(lambda line: (line[1],line[2],line[3]))


def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def getResult(df_data,seed=10):
    def reduceResult(x,y):
        if x[1] > y[1]:
            return x
        elif x[1] == y[1]:
            return ( sorted(Union(x[0], y[0])) ,x[1])
        return y
    df_data_tmp = df_data.map(lambda x: ( (x[0][0],x[0][1]),( [x[0][2]],x[1]) ) ).reduceByKey(reduceResult)
    
    def mapResult(r):
        if len(r[1][0])==1:
            return (r[0][0],r[0][1],r[1][0][0])
        random.seed(seed)
        return (r[0][0],r[0][1],random.choice(sorted(r[1][0])))
        
    return df_data_tmp.map(mapResult)


def similarity_p(dataframe):
    '''
    Pré-calcule de tout les similarité par attribut d'objet.
    '''
    dict_sim = {
        #'ObjectProperty':[],
        #'w1':[],
        #'w2':[],
        #'sim':[]
    }
    # Grouper les valeurs 
    df_tmp = dataframe.map(lambda x: ((x[0],x[1]),[x[2]])).reduceByKey(lambda x,y: x+y)
    
    def mapCompSim(data):
        values = data[1] 
        resultats_sim = []
        for  i in range(len(values)):
            w1 = values[i]
            for  u in range(len(values)):
                w2 = values[u]
                sim = 1
                if w1!=w2:
                    if( isinstance(w1, int) or isinstance(w1, float) ):
                        t = abs(w1-w2)
                        sim = 1/t
                    else:
                        sim = cosdis(word2vec(str(w1)), word2vec(str(w2)))
                        
                if(( (w1,w2) ,sim) not in resultats_sim):
                    resultats_sim.append(( (w1,w2) ,sim))
                if (( (w2,w1) ,sim) not in resultats_sim):
                    resultats_sim.append(( (w2,w1) ,sim))
        return data[0],dict(resultats_sim)
    
    dict_sim = df_tmp.map(mapCompSim)
    
    
    return dict_sim
#dict(similarity(df_data).collect())
#similarity(df_data).take(1)

# Fonction pour le calcule des précisions
def evaluation_p(truth,find,dataframe):
    
    t = truth.map(lambda r:((r[0],r[1]),r[2]))
    v = find.map(lambda r:((r[0],r[1]),r[2]))
    join = t.join(v)
    df = dataframe.map(lambda r:((r[0],r[1]),(r[2],r[3])))
    df = df.join(join)
    
    #('Object1', 'Property5', '234041', 'Source1')
    #(x[0][0],x[0][1],x[1][0][1])
    def mapDataValue(x):
        value = x[1][0][0]
        line  = x[1][1]
        value_truth = line[0]
        value_find = line[1] 
        if (value == value_find) and (value == value_truth):
            return (1.0,1.0)
        elif (value != value_find) and (value != value_truth):
            return (0.0,0.0)
        elif (value == value_find) and (value != value_truth):
            return (1.0,0.0)
        elif (value != value_find) and (value == value_truth):
            return (0.0,1.0)
    
    predictionAndLabels = df.map(mapDataValue)
    #.map(lambda x: (x[1],x[0]))
    #print(predictionAndLabels.take(2))
    metrics = MulticlassMetrics(predictionAndLabels)
    
    return {"precision" : metrics.precision(1.0), "recall" : metrics.recall(1.0), "accuracy" : metrics.accuracy, "fMeasure" : metrics.fMeasure(1.0)}

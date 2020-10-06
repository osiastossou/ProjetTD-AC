import numpy as np
import pandas as pd
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
np.seterr(divide='ignore', invalid='ignore') 

import warnings
warnings.filterwarnings("error")


# Pré-Calcule de la similarité entre les valeur

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
        #'ObjectProperty':[],
        #'w1':[],
        #'w2':[],
        #'sim':[]
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


# Calcule de confidence

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_confidence(df,rho,theta,dict_sim):
    trustworthiness_score = lambda x: -math.log(1-x)  # Eq. 3

    """Calculate confidence for each Value"""
    z = 0
    #for row in df['ObjectProperty'].unique():
    for key, value_of_di in df[['Object','Property','Value']].drop_duplicates().groupby(by=['Object','Property']):
        
        start_time = time.time()
        #value_of_di = df[df['ObjectProperty']==row]
        row = key[0]+key[1] # ObjectProperty
        value = list(value_of_di["Value"])
        conf_deja = {
            'value':[],
            'conf':[],
            'adjust':[]
        }
        
        for u in range(len(value)):
            # Eq. 5
            # trustworthiness of corresponding websites `W(f)`
            ts = df.loc[((df["Value"] == value[u]) & (df["ObjectProperty"]==row)), "trustworthiness"]
            
            v = sum(trustworthiness_score(t) for t in ts)
            #indeces = df[(df["ObjectProperty"]==row) & (df["Value"]==value[u])].index
            conf_deja['value'].append(value[u])
            conf_deja['conf'].append(v)
            conf_deja['adjust'].append(v)
        
            #df.loc[indeces, 'Value_confidence'] = v
            #df.loc[(df["ObjectProperty"]==row) & (df["Value"]==value[u]), 'Value_confidence'] = v

            #df.loc[indeces, 'Value_confidence_adjust'] = v
            #print(v)
            
            if u>0:
                for k in range(u):
                    # Avant
                    # dernier = u
                    sim = implication(conf_deja['value'][k],conf_deja['value'][u],row,dict_sim)
                    conf_deja['adjust'][k] = conf_deja['adjust'][k] + rho*conf_deja['conf'][u]*sim
                    
                    # dernier
                    # sim = implication(conf_deja['value'][u],conf_deja['value'][k],row,dict_sim)
                    conf_deja['adjust'][u] = conf_deja['adjust'][u] + rho*conf_deja['conf'][k]*sim
            
        for p in range(len(conf_deja['value'])):
            
            indeces_ = (df["ObjectProperty"]==row) & (df["Value"]==conf_deja['value'][p])
            #df.loc[(df["ObjectProperty"]==row) & (df["Value"]==value[u]), 'Value_confidence'] = sigmoid(theta * conf_deja['adjust'][p])
            #print(z,df.loc[indeces_, 'Value_confidence'])
            #time.sleep(30)
            df.loc[indeces_, 'Value_confidence'] = sigmoid(theta * conf_deja['adjust'][p])
            #print(z,df.loc[indeces_, 'Value_confidence'])
            #print(row,sigmoid(theta * conf_deja['adjust'][p]))
            
            
        #print(z,df[df['Value_confidence']>1])
        end_time = time.time() 

        #print(z,value,(end_time-start_time))
        z+=1

    return df
    
def update_source_trustworthiness(df):
    for source in df["Source"].unique():
        indices = df["Source"] == source
        cs = df.loc[indices, "Value_confidence"]
        #print(source,cs)
        #print(source,cs)
        df.loc[indices, "oldtrustworthiness"] = df.loc[indices, "trustworthiness"]
        t = sum(cs) / len(cs)
        
        if t >=1:
            t = 1-0.0001
        elif t <= 0:
            t = 0 + 0.0001
        df.loc[indices, "trustworthiness"] = t
    return df

def stop_condition(t1, t2, t1old , t2old, threshold):
    r = np.dot(t1, t1old) / (norm(t1) * norm(t1old)) - np.dot(t2, t2old) / (norm(t2) * norm(t2old))
    #print('Valeur de convergence actuel',abs(r))
    return abs(r)  < threshold



def iteration(df,rho,theta,dict_sim):
    df = calculate_confidence(df,rho,theta,dict_sim)
    #print('End conf')
    df = update_source_trustworthiness(df)
    #print('End accu source')
    return df


def get_result(dataframe):
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
        resultat[key]= df.loc[(df["Value_confidence"] == max(df["Value_confidence"])),'Value'].values[0]
    #resultat["Value"]=[int(x) for x in resultat["Value"]]
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

# def get_oracle(dataframe,data_truth):
#     data_truth = data_truth.astype('object')
#     resultat = get_result(dataframe)
    
#     data_truth["Object"]=[str(x) for x in data_truth["Object"]]
#     data_truth["Property"]=[str(x) for x in data_truth["Property"]]
#     data_truth["ObjectProperty"] = data_truth["Object"]+data_truth["Property"]
    
#     resultat["Object"]=[str(x) for x in resultat["Object"]]
#     resultat["Property"]=[str(x) for x in resultat["Property"]]
#     resultat["ObjectProperty"] = resultat["Object"]+resultat["Property"]
    
#     count = 0
#     for i,row in resultat.iterrows():
#         if data_truth[data_truth['ObjectProperty']==row['ObjectProperty']]['Value'].values[0] == row['Value']:
#             count = count +1
#     return count/resultat.shape[0]

def evaluation(truth,find,dataframe):

    #print(find)
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]

    
    #for i,row_find in find.iterrows():

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
            
    #print(truth_)
    #print(find_)
            
    precision = precision_score(find_, truth_) 
    recall = recall_score(find_, truth_) 
    f1_score_ = f1_score(find_, truth_) 
    out = {'precision' : precision,
    'recall' : recall,
    'accuracy' : accuracy_score(find_, truth_),
    'f1_score' : f1_score_}
    return out

def train(dataframe, max_iterations=5,
          threshold=1e-6, initial_trustworthiness=0.8,rho=0.5,theta = 0.1,afficher=False,data_truth=pd.DataFrame()):
    if initial_trustworthiness!=None:
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["oldtrustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
    dataframe["Value_confidence"] = np.zeros(len(dataframe.index))
    dataframe["Value_confidence_adjust"] = np.zeros(len(dataframe.index))
    dataframe["Object"]=[str(x) for x in dataframe["Object"]]
    dataframe["Property"]=[str(x) for x in dataframe["Property"]]
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]

    if(afficher==True):
        print('---- Info sur data ----- ')
        print('Nombre object : ',len(dataframe["Object"].unique()))
        print('Nombre attribue : ',len(dataframe["Property"].unique()))
        print('Nombre source : ',len(dataframe["Source"].unique()))
        print('Nombre observation : ',len(dataframe))
        print('---- ------------ ----- \n')
    
    #data_truth = get_truth_to_dict(data_truth)
    dict_sim = similarity(dataframe)
    #print(dict_sim)

    print('End sim')
    evaluation_r = None
    for i in range(max_iterations):
        print('---- Itération : ',i+1,'----- ')
        t1 = dataframe.drop_duplicates("Source")["trustworthiness"]
        t1old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

        dataframe = iteration(dataframe,rho,theta,dict_sim)

        t2 = dataframe.drop_duplicates("Source")["trustworthiness"]
        t2old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]
        
        
        if stop_condition(t1, t2, t1old , t2old, threshold):
            break

    out_data = dataframe.drop("ObjectProperty",axis=1)
    if len(data_truth)!=0:
        #oracle = get_oracle(out_data,data_truth)
        data_truth = get_truth_to_dict(data_truth)
        resultat = get_result(dataframe)
        evaluation_r = evaluation(data_truth,resultat,dataframe)
    else:
        return out_data,i+1,max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean()
    if(afficher==True):
        print('---- Resultat----- ')
        print('Nombre itération : ',i+1)
        #print('Max accuracy : ',max(dataframe.drop_duplicates("Source")["trustworthiness"]))
        #print('Avg accuracy : ',dataframe.drop_duplicates("Source")["trustworthiness"].mean())
        #print('Oracle accuracy : ',oracle)
        print(evaluation_r)

        #out_data,i,max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean(),oracle

    return out_data,i+1,max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean(),evaluation_r
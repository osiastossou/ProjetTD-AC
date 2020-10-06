import numpy as np
from numpy.linalg import norm
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import TruthFinder as tf




def vote(dataframe):
    #dataframe["Object"]=[str(x) for x in dataframe["Object"]]
    #dataframe["Property"]=[str(x) for x in dataframe["Property"]]
    dataframe["ObjectProperty"] = dataframe["Object"]+dataframe["Property"]
    resultat = {}
    
    for key, df in dataframe.groupby(by=['ObjectProperty']):
        l = list(df['Value'])
        resultat[key]=[max(set(l), key = l.count)]
        #print([df.values[0][1],df.values[0][2],max(set(l), key = l.count)])
    return resultat


def get_kt_kf_kd(dataframe,vote_truth,source1,source2):

    df = dataframe[['ObjectProperty','Source','Value']].groupby(by=['Source'])
    claim_source1 = df.get_group(source1).set_index('ObjectProperty')
    claim_source2 = df.get_group(source2).set_index('ObjectProperty')
    
    kt,kf,kd = 0,0,0
    for row,value in vote_truth.items():
        claim_source11 = claim_source1['Value']
        claim_source22 = claim_source2['Value']
        #print(value1,value2)
        #value = values[2]
        #print(row)
        if( (row in  claim_source11) and (row in  claim_source22)):
            value1 = claim_source11[row]
            value2 = claim_source22[row]
            if(str(value1) == str(value) and str(value2) == str(value)):
                kt = kt +1
            elif(str(value1) != str(value) and str(value2) != str(value)):
                kf = kf +1
            else:
                kd = kd +1
    return kt,kf,kd



def compDepen(dataframe,vote_truth,source1,source2,alpha,n,c,error):
    kt,kf,kd = get_kt_kf_kd(dataframe,vote_truth,source1,source2)
    return 1/( 1 + ((1-alpha)/alpha)*
              ( ((1-error)/(1-error+c*error))**kt)*
              ( (error/(c*n+error-c*error))**kf)*
              ( int(1/(1-c))**kd) )

def compAllDepen(dataframe,vote_truth,alpha,n,c):
    #dict_depen = pd.DataFrame(columns=['Source1','Source2','Depen'])
    dict_depen = { # Dictionnaire de dictionnaire

    }
    sources = dataframe[["Source","trustworthiness"]].drop_duplicates().values
    for line in sources:
        depen = 0
        error = 1-line[1]
        #error = 1-dataframe.loc[dataframe["Source"] == source, "trustworthiness"].values[0]
        dict_depen_one_sources = {}
        for line1 in sources:
            #if source!=source1:
            depen=compDepen(dataframe,vote_truth,line[0],line1[0],alpha,n,c,error)
            dict_depen_one_sources[line1[0]]=depen

        dict_depen[line[0]] = dict_depen_one_sources
    return dict_depen 



def orderSourceByDepen(sources,dict_depen):
    '''
    sources : liste des sources dont on veux faire l'ordre
    retourne une liste de source dans l'ordre decroisante
    '''

    df_sourceOrderByDepen = {}
    #df_sourceOrderByDepen = pd.DataFrame(columns=['Source','Depen'])
    for s in sources:
        df_sourceOrderByDepen[s]=[s,max(dict_depen[s].values())]
        
    return sorted(df_sourceOrderByDepen.items(), key=lambda x: x[1]) # Renvoie une liste de couple (source,depen)



# def adjust_confidence(df,rho,df_sim):
#     """Eq. 6"""
    
#     for i, row1 in df[['ObjectProperty']].iterrows():
#         value_of_di = df[df['ObjectProperty']==row1["ObjectProperty"]]
#         for j, row2 in value_of_di.drop_duplicates("Value").iterrows():
#             f1 = row2["Value"]
#             s = 0
#             conf_ajust = 0
#             for k, row3 in value_of_di.drop_duplicates("Value").iterrows():
#                 f2 = row3["Value"]
#                 if f1 == f2:
#                     continue
#                 else:
#                     # implication(f2, f1)
#                     sim = tf.implication(f2, f1,row1["ObjectProperty"],df_sim)
#                     if str(sim) == 'nan':
#                         sim = 0.0000000001
#                     s += row3["Value_confidence"]*sim
#                     #print(sim)
#             conf_ajust = rho * s + row2["Value_confidence"]
#             df.at[j, 'Value_confidence'] = conf_ajust

#     return df




#En cours
def calculate_confidence(dataframe,dict_depen,c,n,rho,df_sim,algo="DEPEN"):
    """Calculate confidence for each Value"""
    #print(dict_depen)

    z = 0
    #for o_p in dataframe["ObjectProperty"].unique():
    for key, df_value in dataframe.groupby(by=['ObjectProperty']):
        
        start_time = time.time()

        #df_value = dataframe[dataframe["ObjectProperty"] == o_p]
        conf_deja = {
            'value':[],
            'conf':[],
            'adjust':[]
        }
        values = df_value["Value"].unique()
        for u  in range(len(values)):
            value = values[u]
            listSources = list(df_value[df_value["Value"] == value]['Source']) # Liste des sources qui ont emisent cette valeurs
            ordreListSources = orderSourceByDepen(listSources,dict_depen) # Ordornement
            pre = []
            valueConfidence = 0
            tScore = 1
            voteCount = 0
            for source_depen in ordreListSources:
                source = source_depen[0]
                if algo!="DEPEN":
                    ts = df_value[df_value['Source']==source][['trustworthiness']].values[0]
                    if ts<=0:
                        ts = 0.001
                    tScore = math.log(n*ts/(1-ts))
                if len(pre) == 0:
                    voteCount = 1
                else:
                    voteCount = 1
                    depen = dict_depen[source]
                    for s in pre:
                        voteCount *= (1 - c*depen[s])
                pre.append(source)
                valueConfidence += tScore*voteCount

            conf_deja['value'].append(value)
            conf_deja['conf'].append(valueConfidence)
            conf_deja['adjust'].append(valueConfidence)

            if algo=="ACCUSIM":
                
                if u>0:
                    for k in range(u):
                        # Avant
                        # dernier = u
                        # 
                        sim = tf.implication(conf_deja['value'][k],conf_deja['value'][u],key,df_sim)
                        conf_deja['adjust'][k] = conf_deja['adjust'][k] + rho*conf_deja['conf'][u]*sim
                        
                        # dernier
                        # sim = implication(conf_deja['value'][u],conf_deja['value'][k],row,dict_sim)
                        conf_deja['adjust'][u] = conf_deja['adjust'][u] + rho*conf_deja['conf'][k]*sim
            # else:
            #     indices = dataframe["Value"] == value
            #     dataframe.loc[indices, "Value_confidence"] = valueConfidence
        
        #if algo=="ACCUSIM":
        for p in range(len(conf_deja['value'])):
            
            indeces_ = (dataframe["ObjectProperty"]==key) & (dataframe["Value"]==conf_deja['value'][p])
            
            dataframe.loc[indeces_, 'Value_confidence'] =  conf_deja['adjust'][p]
        

        end_time = time.time() 

        #print(z,value,(end_time-start_time))
        #z+=1

    #if algo=="ACCUSIM":
        #dataframe = adjust_confidence(dataframe,rho,df_sim)
    return dataframe





def update_source_trustworthiness(dataframe):
    #for source in dataframe["Source"].unique():
    df_per_data_item = dataframe.groupby(by=['ObjectProperty'])
    for key, df_value in dataframe.groupby(by=['Source']):
        #indices = dataframe["Source"] == source
        #df_value = dataframe[indices]
        tws = 0
        for i, row in df_value.iterrows():
            data_item = df_per_data_item.get_group(row['ObjectProperty'])[['Value','Value_confidence']].drop_duplicates()
            #print(data_item['Value_confidence'])
            divise = sum([math.exp(r)  for r in data_item['Value_confidence']]) 
            #divise = sum([math.exp(tf.sigmoid(r))  for r in data_item['Value_confidence']]) 
            # for u, row1 in data_item.iterrows():
            #     valueOfDataItem = row1['Value_confidence']
            #     divise+=math.exp(valueOfDataItem)
            tws += math.exp(row['Value_confidence'])/divise # A revoir
            #tws += math.exp(tf.sigmoid(row['Value_confidence']))/divise # A revoir
        indices = dataframe["Source"] == key
        dataframe.loc[indices, "oldtrustworthiness"] = dataframe.loc[indices, "trustworthiness"]

        t = tws / len(df_value)
        if t >=1:
            t = 1-0.0001
        elif t <= 0:
            t = 0 + 0.0001
        dataframe.loc[indices, "trustworthiness"] = t
    return dataframe



def stop_condition(t1, t2, t1old , t2old, threshold):
    r = np.dot(t1, t1old) / (norm(t1) * norm(t1old)) - np.dot(t2, t2old) / (norm(t2) * norm(t2old))
    #print(abs(r))
    return abs(r)  < threshold



def iteration(df,dict_depen,c,n,rho,algo,df_sim):
    df = calculate_confidence(df,dict_depen,c,n,rho,df_sim,algo)
    df = update_source_trustworthiness(df)
    return df





def train(dataframe, max_iterations=10,
          threshold=1e-6, initial_trustworthiness=0.8,c=0.3,n=100,alpha = 0.5,rho=0.3,afficher=False,algo="DEPEN",data_truth=pd.DataFrame()):
    if initial_trustworthiness!=None:
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["oldtrustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
    dataframe["Value_confidence"] = np.zeros(len(dataframe.index))
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


    
    df_sim = {}
    if algo=='ACCUSIM':
        df_sim = tf.similarity(dataframe)
    
    #print(df_sim)

    #oracle = -1
    evaluation_r = None
    vote_truth = vote(dataframe)
    dict_depen = compAllDepen(dataframe,vote_truth,alpha,n,c)
    for i in range(max_iterations):
        #print('---- Itération : ',i+1,'----- ')
        t1 = dataframe.drop_duplicates("Source")["trustworthiness"]
        t1old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

        dataframe = iteration(dataframe,dict_depen,c,n,rho,algo,df_sim)

        t2 = dataframe.drop_duplicates("Source")["trustworthiness"]
        t2old = dataframe.drop_duplicates("Source")["oldtrustworthiness"]

        if tf.stop_condition(t1, t2, t1old , t2old, threshold):
            break
        
        truth_compute = tf.get_result(dataframe)
        if vote_truth!=truth_compute:
            dict_depen = compAllDepen(dataframe,truth_compute,alpha,n,c)

    out_data = dataframe.drop("ObjectProperty",axis=1)
    if len(data_truth)!=0:
        #oracle = get_oracle(out_data,data_truth)
        data_truth = tf.get_truth_to_dict(data_truth)
        resultat = tf.get_result(dataframe)
        evaluation_r = tf.evaluation(data_truth,resultat,dataframe)
    else:
        return out_data,i+1,max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean()
    if(afficher==True):
        print('---- Resultat----- ')
        print('Nombre itération : ',i+1)
        # print('Max accuracy : ',max(dataframe.drop_duplicates("Source")["trustworthiness"]))
        # print('Avg accuracy : ',dataframe.drop_duplicates("Source")["trustworthiness"].mean())
        # print('Oracle accuracy : ',oracle)
        print(evaluation_r)

    return out_data,i+1,max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean(),evaluation_r
    #max(dataframe.drop_duplicates("Source")["trustworthiness"]),dataframe.drop_duplicates("Source")["trustworthiness"].mean(),oracle



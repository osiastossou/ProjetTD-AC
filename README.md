
# Projet TD-AC: Efficient Data Partitioning based Truth Discovery

#### Author 1 : Osias Noël Nicodème Finagnon TOSSOU (African Institute for Mathematical Sciences at Mbour,Senegal) osias.tossou@aims-senegal.org
#### Author 2 : Mouhamadou Lamine Ba (Université Alioune Diop de Bambey at Bambey,Senegal) mouhamadoulamine.ba@uadb.edu.sn

## Note :
*Note: For confidentiality reasons, some of the actual data we used are not online, others are publicly available actual data as we have obtained them in other articles. The synthetic data are generated by an algorithm described in the paper and implemented in python in the DataSynthetiqueGenerator.py file. Three synthetic data whose configuration is in the paper is present here in the data folder (DS1, DS2, DS3)*

# Absract :
This paper presents **TD-AC** which is an effective algorithm for the truth discovery problem *when the attributes over data are structurally correlated*. We build our procedure on an abstract representation of the truth in the data, the k-means clustering technique and the silhouette measure to automatically find  an optimal partitioning of the input data (or a near-optimal) maximizing the accuracy of any *base* truth discovery process. The intensive experiments conducted on synthetic and real datasets show that **TD-AC** outperforms existing  partitioning approaches with a more reasonable running time. It improves on synthetic datasets the accuracy of standard truth discovery algorithms by 6% at least and by 16%  at most and also significantly when the data coverage rate is high for the other types of datasets.

# Code Run description :



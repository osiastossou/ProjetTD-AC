import pandas as pd
from utils import utils 
import time

version = "p" # s = sequentiel et p = parallel
algo = 'acs' # mv, tf, dp, ac, acs, agp, tdac
max_iterations=20

if version == "s":

    from sequentiel.MajorityVoting import MajorityVoting

    from sequentiel.TruthFinder import TruthFinder

    from sequentiel.DepenAccuAccuSim import DepenAccuAccuSim

    from sequentiel.AccuGenPartition import AccuGenPartition

    from sequentiel.TDAC import TDAC

    dataframe = pd.read_csv("data/DS1/data.csv")
    data_truth = pd.read_csv("data/DS1/data_truth.csv")


    if algo == 'mv':
        # MajorityVoting
        print("Version sequentiel de MajorityVoting")
        mv = MajorityVoting()
        start = time.time()
        mv.train(dataframe)
        end = time.time()
        print("Time : ",end-start)
        print(mv.evaluation(data_truth))
    elif algo == 'tf':
        # TruthFinder
        print("Version sequentiel de TruthFinder")
        tf = TruthFinder(max_iterations=max_iterations)
        start = time.time()
        tf.train(dataframe)
        end = time.time()
        print("Time : ",end-start)
        print(tf.evaluation(data_truth))
    elif algo == 'dp':
        # Depen
        print("Version sequentiel de Depen")
        dp = DepenAccuAccuSim(algo="DEPEN",max_iterations=max_iterations) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        dp.train(dataframe)
        print(dp.evaluation(data_truth))
    elif algo == 'ac':
        # Accu
        print("Version sequentiel de Accu")
        ac = DepenAccuAccuSim(algo="ACCU",max_iterations=max_iterations) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        ac.train(dataframe)
        print(ac.evaluation(data_truth))
    elif algo == 'acs':
        # AccuSim
        print("Version sequentiel de AccuSim")
        acs = DepenAccuAccuSim(algo="ACCUSIM",max_iterations=max_iterations) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        acs.train(dataframe)
        print(acs.evaluation(data_truth))
    elif algo == 'tdac':
        # TDAC
        print("Version sequentiel de TDAC")
        tdac = TDAC()
        tdac.train(dataframe,max_iter_part_check=1,max_iterations=max_iterations)
        print(tdac.evaluation(data_truth))
    elif algo == 'agp':
        # AccuGenPartition
        print("Version sequentiel de AccuGenPartition")
        acs = DepenAccuAccuSim(algo="DEPEN",max_iterations=max_iterations) # 
        agp = AccuGenPartition(nbr_partition_explorer=2,algo=acs,max_iterations=max_iterations)
        agp.train(dataframe,data_truth)
        # print(acs.evaluation(data_truth))

elif version == "p":
    # {'precision': 0.6022918166546676, 'recall': 0.667442324313543, 'accuracy': 0.80615, 'f1_score': 0.6331956226938724}
    from pyspark import  SparkContext,SparkConf
    conf = SparkConf().setAll([('spark.driver.host','localhost'), ('spark.executor.cores', '4'), ('spark.cores.max', '4'), ('spark.driver.memory','8g')])
    sc = SparkContext(conf=conf)


    from parallel.MajorityVoting import MajorityVoting

    from parallel.TruthFinder import TruthFinder

    from parallel.DepenAccuAccuSim import DepenAccuAccuSim

    import unittest
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    rdd_data = utils.read('data/DS1/data.csv',sc)
    dataframe = rdd_data.map(lambda x: (x[1],x[2],x[3],x[4],0.8,0.0,0.8)).filter(lambda x: x[0] != 'Object')

    rdd_data_truth = utils.read('data/DS1/data_truth.csv',sc,truth=True)
    data_truth = rdd_data_truth.filter(lambda x: x[0] != 'Object')

    if algo == 'mv':
        # MajorityVoting
        print("Version parallel de MajorityVoting")
        mv = MajorityVoting()
        mv.train(dataframe)
        print(mv.evaluation(data_truth))

    elif algo == 'tf':
        # TruthFinder
        print("Version parallel de TruthFinder")
        tf = TruthFinder(max_iterations=max_iterations)
        tf.train(dataframe)
        print(tf.evaluation(data_truth)) 
    elif algo == 'dp':
        # Depen
        print("Version parallel de Depen")
        dp = DepenAccuAccuSim(algo="DEPEN",sc=sc,max_iterations=20,threshold=1e-6, initial_trustworthiness=0.8,c=0.3,n=60,alpha = 0.5,rho=0.3) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        start = time.time()
        dp.train(dataframe)
        end = time.time()
        print(dp.evaluation(data_truth))
        print("Time : ",end-start)
    elif algo == 'ac':
        # Accu
        print("Version parallel de Accu")
        ac = DepenAccuAccuSim(algo="ACCU",sc=sc,max_iterations=20,threshold=1e-6, initial_trustworthiness=0.8,c=0.3,n=60,alpha = 0.5,rho=0.3) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        start = time.time()
        ac.train(dataframe)
        end = time.time()
        print(ac.evaluation(data_truth))
        print("Time : ",end-start)

    elif algo == 'acs':
        # AccuSim
        print("Version parallel de AccuSim")
        acs = DepenAccuAccuSim(algo="ACCUSIM",sc=sc,max_iterations=20,threshold=1e-6, initial_trustworthiness=0.8,c=0.3,n=60,alpha = 0.5,rho=0.3) # Le parm algo pour switcher entre DEPEN, ACCU et ACCUSIM
        start = time.time()
        acs.train(dataframe)
        end = time.time()
        print(acs.evaluation(data_truth))
        print("Time : ",end-start)

else:
    print("Veuillez choisir entre s et p pour la version de l'algorithme")






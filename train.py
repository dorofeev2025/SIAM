import pandas as pd
import numpy as np
from scipy.stats import f
import dtutils as dt

##############################

if __name__ == "__main__":


   # Select all sensors covering area A
   sensors = pd.read_csv('data/sensors.csv') 
   sensors = sensors[sensors['Area']=='A']['Name'].tolist()

   # Main data file
   datafile = pd.read_csv('data/2018_datah.csv', parse_dates=[0], index_col=0)

   # Setup leakage data
   leaks = dt.LeakSet.load_csv('data/leaks.csv')
   leaks18 = leaks.filter(start_before='2018-12-31')
   known_leaks = leaks.filter(area='A', start_before='2018-12-31')

   # Setup training period: all of 2018 without known leaks except p257 and p427
   trainset = dt.TrainSet.all_year(2018)
   known_leaks.remove('p257')
   known_leaks.remove('p427')
   trainset.exclude_leaks(known_leaks)
   trainmask = trainset.to_mask(datafile.index)

   # Filter training data
   traindata = datafile.loc[trainmask, sensors]

   # Setup time clusters
   train_map = dt.group_by_hour(traindata.index)
   
   # Estimate group means and covariance matrices
   mean, cov, nobs = dt.est_cov_by_group(traindata, train_map)
   dt.save_cov('results/areaA.pkl', mean, cov, nobs)

   # Generate cluster summary (for analyzing similarity)
   rows = []
   for cluster in nobs:
       nс = nobs[cluster]
       mс = mean[cluster]
       stdс = np.sqrt(np.diag(cov[cluster]))  # standard deviation

       row = {'cluster': cluster, "nobs": nс}
       # Add means and standard deviations for all sensors
       for i, m in enumerate(mс):
           row[f"mean_{i}"] = m
       for i, s in enumerate(stdс):
           row[f"std_{i}"] = s
       rows.append(row)
       сf = pd.DataFrame(rows)
       сf.to_csv('results/clustersA_summary.csv', sep=';', index=False)


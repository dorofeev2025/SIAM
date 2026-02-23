import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f   # Fisher distribution
import dtutils as dt

##############################

if __name__ == "__main__":

   # Select all sensors covering area A
   sensors = pd.read_csv('data/sensors.csv') 
   sensors = sensors[sensors['Area']=='A']['Name'].tolist()

   # Main data file
   datafile = pd.read_csv('data/2018_datah.csv', parse_dates=[0], index_col=0)
   evaldata = datafile[sensors]
   eval_map = dt.group_by_hour(evaldata.index)

   # Setup leakage data & training period. Same as in train.py. 
   # This block is only needed for plotting training periods and actual leeks 
   leaks = dt.LeakSet.load_csv('data/leaks.csv')
   leaks18 = leaks.filter(start_before='2018-12-31')
   known_leaks = leaks.filter(area='A', start_before='2018-12-31')
   trainset = dt.TrainSet.all_year(2018)
   known_leaks.remove('p257')
   known_leaks.remove('p427')
   trainset.exclude_leaks(known_leaks)

   # Loading saved moments
   mean, cov, nobs = dt.load_cov('results/areaA.pkl')

   # Critical values for 'Fisherized' Hotelling distribution
   nobs_min =  min(nobs.values())
   df1 = len(sensors)
   df2 = nobs_min - df1

   cv99 = f.ppf(0.99, df1, df2) 
   cv95 = f.ppf(0.95, df1, df2)
   cvmean = df2 / (df2 -2)
   
   # Whitening transformation 
   z_transf = dt.whiten_by_group(evaldata, eval_map, mean, cov)

   # Compute Z^2 as sum of squares of transformed values
   z_transf['z_tss'] = (z_transf ** 2).sum(axis=1)
   # Convert Hotelling distribution to Fisher distribution
   z_transf['f_stat'] = dt.f_stat(z_transf['z_tss'], eval_map, nobs, df1) 
 
   # Compute alarm filter
   alarm_flags = dt.alarm_filter(z_transf['f_stat'], crit_on=cv95, crit_off=cvmean)

   # Auxiliary statistics
   fs_wht = dt.wht(z_transf['f_stat'], df1)
   fs_ma = fs_wht.rolling("24H").mean()

   z_transf['fs_ma'] = fs_ma

   # Saving the results in the CSV format
   z_transf.to_csv('results/areaA_18.csv', float_format='%.2f')

   # Plot results
   ymax = max(fs_wht)
   min_time = datafile.index[0] 
   max_time = datafile.index[-1]
   
   fig, ax = plt.subplots(figsize=(12, 6))

   # Main graphs
   # "Fisherized' Hotelling statistic, plotted using Wilson-Hilferty transformation
   ax.plot(z_transf.index, fs_wht)
   # Its moving average
   ax.plot(z_transf.index, fs_ma, color='purple', linewidth=2)
   # Critical values transformed to Wilson-Hilferty scale
   ax.plot([min_time, max_time], [dt.wht(cv95,df1)] * 2, 
       color='orange', linestyle='--', label=f'Critical Value @ 95% = {cv95:.2f}')
   ax.plot([min_time, max_time], [dt.wht(cv99,df1)] * 2,
       color='red', linestyle='--', label=f'Critical Value @ 99% = {cv99:.2f}')

   # Alarm filter
   ax.plot(z_transf.index, alarm_flags, color='black',linewidth=3)
   # Actual leaks plotted as a Gantt chart
   leaks18.plot(ax=ax, method = 'gantt', ystep = 0.1)
   # Training periods
   trainset.plot_as_arrows(ax=ax, label='Training interval(s)')

   # Labels, title, formatting
   ax.set_xlabel('Date')
   ax.set_ylabel('Value')
   ax.set_title('$T^2_F$ statistic (Wilson-Hilferty transformation)')
   ax.grid(axis='y')
   ax.legend(ncol=2)
   ax.set_xlim(z_transf.index.min(), z_transf.index.max())
   ax.set_ylim(bottom=0)

   fig.tight_layout()
   plt.savefig('results/AreaA_18.png')
   plt.show()
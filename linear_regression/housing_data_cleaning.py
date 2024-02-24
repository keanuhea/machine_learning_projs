
#dataset comes from https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market?resource=download

import tqdm as tqdm

import pandas as pd 
import numpy as np 

from itertools import accumulate

import matplotlib.pylab as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from scipy.stats import norm
from scipy.stats import boxcox
from scipy.stats.mstats import normaltest
from scipy import stats 


df = pd.read_csv(r'/Users/anuheaparker/Desktop/coding projects/personal_projects/projects/machine_learning/NY-House-Dataset.csv')
#print(df)

#see first five of the df
#df.head(5)

#see info in the df 
df.info()

#see more information of attribute
print(df['PRICE'].describe())
#print(df["BATH"].value_counts())


#CHECK FOR MISSING DATA AND DUPLICATES
#print(df.isnull().sum())
#print(sum(df.duplicated(subset = "ADDRESS")) == 0) #this dataset doesn't have unique ids
#print(df.index.is_unique)

#GRAPH OF COUNT OF A FEATURE 
#fig, ax = plt.subplots(figsize = (15,5))
#plt1 = sns.countplot(x=df['BEDS'], order=pd.value_counts(df['BEDS']).index)
#plt1.set(xlabel = 'Beds', ylabel='Count of Beds')
#plt.show()
#plt.tight_layout()

#SCATTERPLOT OF RELATIONSHIP WITH ALL FEATURES
#sns.pairplot(df)
#plt.show()


#CHECK IF TARGET IS NORMALLY DISTRIBUTED 
def plotting_3_chart(data, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(data.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(data.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(data.loc[:,feature], orient='v', ax = ax3);
    
    plt.show()
    
#plotting_3_chart(df, 'PRICE')

previous_data = df.copy()

#LOG TRANSFORMATION & PLOTTING
df["PRICE"] = np.log(df["PRICE"])
#plotting_3_chart(df, "PRICE")

#hey im trying this out 


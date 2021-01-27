#Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from numpy.core.shape_base import vstack
from scipy.cluster.hierarchy import dendrogram, linkage

# Read tickets to get the data from Yahoo

'Read Tickers'
tickers = np.append(pd.read_excel('mapping.xlsx')['Ticker'].values, 'ACWI')
prices_list = []

'Get the data from yahoo and put it in DataFrame'
for ticker in tickers:
    try:
        prices = dr.DataReader(ticker,'yahoo','01/01/2018')['Adj Close']
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        prices_list.append(prices)
    except:
        pass
    prices_df = pd.concat(prices_list,axis=1)
prices_df.sort_index(inplace=True)
print(prices_df)

"""
Calculate average annual percentage return and volatilities
over a theoretical one year period
"""
returns = prices_df.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)

'Format the data as a numpy array to feed into the K-Means algorithm'
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distorsions = []

# Elbow Curve graphic
'Draws the elbow curve from 2 to 15 clusters'
for k in range(2, 15):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 15), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()

# Creating and showing the clusters
'Computing K-Means with K = 5 (5 clusters)'
centroids,_ = kmeans(data,5)

'Assign each sample to a cluster'
idx,_ = vq(data,centroids)
Cluster_points = pd.DataFrame(data=data)                           # 1st row as the column names
Cluster_points.columns = ['Ret','Vol']
Cluster_points['Cluster'] = idx
colors = ['#fcba03', '#02cf10', '#07dce3', '#e810d2', '#ff0800']

'Scatter plot of everything'
plt.figure(figsize=(18,8))
for i in range(len(idx)):
    'Plots the name of each ticker'
    plt.text( data[i,0], data[i,1], tickers[i], size=10)
for i in range(len(colors)):
    'Plot the points of each ticker'
    plt.scatter(x = Cluster_points.loc[Cluster_points['Cluster']==i, 'Ret'],
                y = Cluster_points.loc[Cluster_points['Cluster']==i,'Vol'],
                color = colors[i], label=i)    
    'Plots the centroids as stars'
    plt.plot(centroids[i,0],centroids[i,1],'*',color='black',markersize=10)
plt.legend()
plt.title('Clusters')
plt.show()

'Creates an excel file with the ticker and the cluster asigned'
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
Det = pd.DataFrame(details)
Det.columns = ['Ticker','Cluster']
Det.to_excel("Ticker_Cluster.xlsx")

'Plot Hierarchical Clustering Dendrogram'
plt.figure(figsize=(15, 15))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('ETFs Countries')
plt.ylabel('Distance')
dendrogram(linkage(vstack([Cluster_points['Vol'],Cluster_points['Ret']]).T, 
                   'ward'),
            orientation='left',
            leaf_rotation=0.,
            leaf_font_size=16.,
            labels=returns.index )
plt.tight_layout()
plt.show()

"""
References:
https://www.pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python
"""

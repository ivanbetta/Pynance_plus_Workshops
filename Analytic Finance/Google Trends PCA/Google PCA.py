
# Google Trends PCA

"""
The objective of this code is compare searches in Google Trends with stock prices.
"""

# Librerias

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import date,datetime
from sklearn import decomposition
from pytrends.request import TrendReq
from yahoo_historical import Fetcher

def one_df(df,data):
    'Makes a column with a date'
    df['Date'] = [ datetime.strptime( df['Date'].iloc[i], '%Y-%m-%d') for i in range(len(df['Date']))]
    'Puts everything in just 1 DF'
    new_df = pd.concat([df, data], ignore_index=True, sort=False)
    'Sort data by date'
    new_df = new_df.sort_values(by=['Date'])
    'Fills na with ffill'
    new_df[new_df.columns[0]] = new_df[new_df.columns[0]].fillna(method = 'ffill')
    new_df=new_df.dropna()
    'Puts Date in the idex using a fix if is necessary'    
    dif_len = len(data)-len(new_df)
    if dif_len>0:
        new_df.index = data['Date'].index[dif_len::]
    elif dif_len==0:
        new_df.index = data['Date']
    elif dif_len<0:
        new_df.index = range(len(new_df)-1)
        new_df = new_df.drop(range(dif_len))
        new_df.index = data['Date']
    return new_df

def yahoo_finance(stok,m =0, y=0):
    'Gets the data from Yahoo finance'
    today = date.today()
    df = pd.DataFrame(data={})
    dif = pd.DataFrame(data={})
    'Fix for months'
    if y == 0 and m >0:
        y = 0
        while m > today.month:
            y += 1
            m2 = today.month - m
            m = 12*y + m2
    else:
        m = today.month
    'Gets the data from yahoo finance'
    try:
        stock_prices = Fetcher(stok, [today.year-y,m,today.day], [today.year,today.month,today.day])
        dif = stock_prices.getHistorical()
        df[stok] = dif['Adj Close'].to_list()
        if len(dif) < 55:
            print('We need more data for: ' + stock_ticker)
    except:
        print('Oops! that was no valid ticker. Please delete or correct it from the Stock List file: ', stok)
    'Puts the date in a column'
    df['Date'] = dif['Date'].to_list()
    return df

def google_trends(keywords,m=0,y=0):
    'Fix for the time'
    if m > 0 and y == 0:
        period = m
        time = 'm'
    elif y > 0 and m == 0:
        period = y
        time = 'y'
    'Gets the info from google trends'
    pytrend.build_payload(
         # List of keywords to search
         kw_list = keywords, 
         # Category number'
         cat = '203', 
         #data for 1 year, can be 'today 12-m' for 12 months, 'now 12-d' for 12 days, 'now 12-H' for 12 hours
         timeframe = 'today '+str(period)+'-'+str(time), 
         #GL for global searching, MX is for mexico
         geo = 'US',
         #Can be images, news, youtube or froogle for sales,
         gprop = '') 
    data = pytrend.interest_over_time()
    data['Date'] = data.index
    data.index = range(len(data))
    return data

def PCA_gen(Data,n_pca = 1,graph=False, inv = False, size = (10,5)):
    """" Makes the PCA and The Plots """
    X = pd.DataFrame(Data)    
    'Puts the same names in X'
    X.columns = Data.columns
    'plots the data reescaled'
    X.plot(figsize=size)
    plt.show()
    'Computes n PCA components'
    pca = decomposition.PCA(n_components=n_pca)
    X2 = pca.fit_transform(X)
    'Makes a DF with every PCA'
    PCA_df = pd.DataFrame(X2)
    PCA_df.columns = ['PCA '+str(i+1) for i in range(len(X2[0]))]
    PCA_df.index = X.index
    'If graph is True'
    if graph == True:
        'Calls the graf fn'
        graf_PCA(X, PCA_df,n_pca, inv = inv )
        Graf_scatter(X,PCA_df)
    'Prints how much variance explain the number of PCAs'
    percentage =  pca.explained_variance_ratio_
    print('Percentage of variance explained by the PCA:')
    for i in range(len(percentage)):
        print('{0:.2f}% by the PCA '.format(percentage[i]*100),i+1,)
    return PCA_df

def graf_PCA(Data,PCA_DF,n_pca, size = (20,10), inv = False):
    """Plots the PCA with all the series"""
    'Center in Zero'
    Data["x0"] = 0
    'Shows the Data vs the PCA'
    for pca in PCA_DF.columns:
        fig, ax = plt.subplots()
        'Plot every column in grey color'
        for column in Data.columns:
            Data[column].plot(title= pca, figsize=size,color="grey")
        'if it is necesary flips the graph'
        if inv == True:
            PCA_DF[pca] *= -1
        'Plots the PCA in red'
        PCA_DF[pca].plot(figsize=size,color="red")
        plt.legend()
        plt.show()
        
def Graf_scatter(X,PCAs):
    'Makes a plot with every component vs each PCA'
    for PCAs_column in PCAs.columns:
        for X_column in X.columns:
            if X_column != "x0":
                plt.scatter(PCAs[PCAs_column],X[X_column],label=X_column)
                plt.ylabel(X_column)
                plt.xlabel(PCAs_column)
        plt.legend()
        plt.show()

# Main

'Object with the language and timezone '
pytrend = TrendReq(hl='en-US', tz=360)
'Keywords to search, 5 at most'
keywords = ['Cancun','Tulum','Los Cabos', 'Monterrey', 'Guadalajara']
'Stocks to compare'
stock_ticker = ['OMAB.MX','GAPB.MX','ASURB.MX']

# Makes the menu

option = 0
print('Que periodo quieres?\n 1.- 1 Año\n 5.- 5 Años')
while option != 1 or option != 5:
    option = int(input('R = '))
    if option==1:
        months,years = 12,0
        break
    elif option==5:
        months,years = 0,5
        break

# Gets the data

'Gets the info from Google trends'
data = google_trends(keywords, m=months,y=years)
'Does everything with every Ticker'
for Ticker in stock_ticker:
    # Creation of each DF needed
    'Gets the info from Yahoo finance'
    df = yahoo_finance(Ticker,m=months,y=years)
    'Creates a dataframe'
    df_complete = one_df(df,data.copy())
    columns_needed = [Ticker]+keywords
    df_complete = df_complete[columns_needed]
    'Create the df without the ticker(s)'
    df_withour_tickers = df_complete[ keywords ]
    
    # PCA compute
    Dataf = PCA_gen( df_withour_tickers, graph=True, inv = False, size = (10,5))
    'Ajusta el PCA a los precios'
    Dataf['PCA 1'] = Dataf['PCA 1'] + df_complete[Ticker].mean()
    'Grafica Accion vs Most accurate PCA'
    plt.figure(figsize=(15,8))
    plt.title(Ticker+' vs PCA',size=20)
    plt.plot( Dataf.index, Dataf['PCA 1'])
    plt.plot( Dataf.index,df_complete[Ticker] )
    plt.legend(['PCA 1',Ticker],prop={'size': 15})
    plt.xlabel('Date',size=15)
    plt.show()
    'Calculating Correlation'
    corr = np.corrcoef(df_complete[Ticker], Dataf['PCA 1'])
    print('The correlation between the PCA and'+Ticker+' is: ',round(corr[0,1]*100,4),'% \n')

# Category numbers: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
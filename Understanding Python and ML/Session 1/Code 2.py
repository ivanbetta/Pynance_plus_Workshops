
# Understanding Python

'The objective of this code is understand main libraries, functions, conditionals and loops'

# Libraries

'Always add libraries on the top and comments to your codes'

from datetime import date
import numpy as np
import pandas as pd
import pandas_datareader as dr
import matplotlib.pyplot as plt

# This function calculates the RSI

def rsi(prices, n=14):

    deltas = np.diff(prices)
    seed = deltas[:n]

    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n

    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n+1] = 100. - 100./(1.+rs)

    for i in range(n+1, len(prices)):
        delta = deltas[i-1]

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

# Main

'Read data from excel'
Names = pd.read_excel('investment_universe.xlsx').dropna()
print('\n --- Ticker List Loaded --- \n')

for i in range(len(Names)):
    
    'Get stock_ticker and calculate features'
    stock_ticker = Names['Ticker'][i]

    try:
        today = date.today()
        prices = dr.DataReader(stock_ticker,'yahoo',str(today.day)+'/'+str(today.month)+'/'+str(today.year-3))
        df = pd.DataFrame(prices)
        df['Date'] = df.index
        df.index = range(0,len(df))
    except:
        print('No data for ',stock_ticker )
        continue
    
    df['1D Return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1)).dropna()
    df['1D Return'].fillna(0,inplace=True)

    df['RSI14'] = rsi(df['Adj Close'])

    df['Traded Value'] = df['Adj Close'] * df['Volume']
    df['Traded Value DMA20'] = df['Traded Value'].rolling(window=20).mean()

    df['21DMA'] = df['Adj Close'].rolling(window=21).mean()
    df['55DMA'] = df['Adj Close'].rolling(window=55).mean()
    
    df['70'] = 70
    df['30'] = 30
    
    if df.iloc[len(df)-1,8] > 70 or df.iloc[len(df)-1,8] < 30:
        
        if  df.iloc[len(df)-1,9] > df.iloc[len(df)-1,10]*1.1:
                        
            print(stock_ticker)
            
            if df.iloc[len(df)-1,8] > 65:
                print('Overbought')
            elif df.iloc[len(df)-1,8] < 35:
                print('Oversold')

            print('RSI 14D: ', format(df.iloc[len(df)-1,8],',.2f'))
            print('Stock traded $',format(df.iloc[len(df)-1,9],',.2f'),'USD', format(((df.iloc[len(df)-1,9]/df.iloc[len(df)-1,10])-1)*100,',.2f'),'% above its 20D Average Traded Value:')
    
            # Chart Adj Close        
            df['Adj Close'].plot(grid=True,color='black',lw=1.2)
            df['21DMA'].plot(grid=True,color='blue',lw=1)
            df['55DMA'].plot(grid=True,color='orange',lw=1)
            plt.ylabel('Adj Close ' + stock_ticker)
            plt.gca().legend(('Adj Close','21DMA','55DMA'))
            plt.show()  
    
            # Chart RSI14                
            df['RSI14'].plot(grid=True,color='black',lw=1.2)
            df['70'].plot(grid=True,color='red',lw=1)
            df['30'].plot(grid=True,color='green',lw=1)
            plt.ylabel('RSI14 ' + stock_ticker)
            plt.show()                

# References

'https://docs.python.org/3/library/index.html'
'https://numpy.org/'
'https://pandas.pydata.org/'
'https://pandas-datareader.readthedocs.io/en/latest/cache.html'
'https://matplotlib.org/'


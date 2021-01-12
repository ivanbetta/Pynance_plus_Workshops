'Importing libraries that we are going to use'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

today_date=date.today()

shares = pd.DataFrame(data={})
mkt_cap= pd.DataFrame(data={})
sector= pd.DataFrame(data={})
df= pd.DataFrame(data={})
flag=2
ndatos=4

'Getting the raw data from our exel file'
for k in range(1,4):
    df=pd.read_csv("Ejemplo"+str(k)+".csv",header=1,delimiter=",",decimal=',', skiprows=[i for i in range(0,8)])
    df["Market Value"]=df["Market Value"].str.replace(',','')
    df["Market Value"]=pd.to_numeric(df["Market Value"], downcast="float")
    df = df[df["Market Value"] >10000]
    df = df[df["Asset Class"]=="Equity"]
    if(flag>1):
        sector[k]=df["Sector"]
        flag=-60

'With the flag we make sure that we only enter the sectors once'
shares[k]=df["Shares"]
mkt_cap[k]=df["Market Value"]

'We asifn the tickers of our original df to our new df and then we set those tickers as indexes'
sector["Ticker"]=df["Ticker"]
mkt_cap["Ticker"]=df["Ticker"]
shares["Ticker"]=df["Ticker"]

shares= shares.set_index('Ticker')
mkt_cap= mkt_cap.set_index('Ticker')
sector= sector.set_index('Ticker')

'Maybe we need to delete or replace all the nan values in our DF'
mkt_cap=mkt_cap.fillna(0)
sector=sector.fillna("-")

# Graphics
'First we need names for our columns to make easier access to them'
shares.columns= ['#Shares']
mkt_cap.columns= ['Mkt Cap']

'If you need to count the values of some column in a DF you can use this line'
sector_counts = df['Sector'].value_counts()

'First the most simple plot'
plt.plot( mkt_cap.index[:10],mkt_cap['Mkt Cap'].to_list()[:10])
plt.title('Market Cap Graphic')
plt.show()

'Now lets create a bar plot'
plt.bar(sector_counts.index,sector_counts.to_list())
plt.title('Sectors')
plt.show()

'You can create an horizontal bar plot, if its better for you'
plt.barh(sector_counts.index,sector_counts.to_list())
plt.title('Sectors')
plt.show()

'We can also make a Scatter plot with names in each point'
plt.scatter( shares.loc[shares.index[:5],'#Shares'] , mkt_cap.loc[shares.index[:5],'Mkt Cap'] )
for ticker in range(5):
    plt.annotate(str(shares.index[ticker]), (shares['#Shares'].iloc[ticker] , mkt_cap['Mkt Cap'].iloc[ticker] ))
plt.title('Market Cap vs Shares')
plt.xlabel('Shares')
plt.ylabel('Market Cap ')
plt.show()

'We conclude by transposing our df in order to graph them easier'
shares= shares.T
mkt_cap= mkt_cap.T
sector= sector.T




# Quandl

'The objective of this code is to download data from Quandl'

# Libraries

import matplotlib.pyplot as plt
import quandl
quandl.ApiConfig.api_key = "bsZGEZZrbU6VWLn_hVrZ"  # Key needed to get the data  
          
# Data from Quandl

'Make a chart of the Holdings of Equity Shares Issued by Mexican Enterprises'

s1 = quandl.get('BDM/SF103160')
print(s1.tail(1))

plt.plot(s1, label='USD Holdings in $bn')

'Adss a label with the text Date in the x axis'
plt.xlabel('Date')

'Adds a label with the text USD $bn in the y axis'
plt.ylabel('USD $bn')

'Adds the title Non—Residents Holdings of Equity Shares Issued by Mexican Enterprises above the plot'
plt.title('Non—Residents Holdings of Equity Shares Issued by Mexican Enterprises')

'Adds legend'
plt.legend()

'Show chart'
plt.show()

# References

'https://matplotlib.org/'
'https://www.quandl.com/'
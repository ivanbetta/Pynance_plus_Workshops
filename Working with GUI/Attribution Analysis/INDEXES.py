
import urllib.request
import pandas as pd
import numpy as np
from datetime import date

DateTODAY=date.today()
DateTODAY=str(DateTODAY)

def Download_file(Address, Path, Name, col_names):
    #Path is the direccion where the file is gonna be downloaded
    Path = Path+Name
    #Open de link where it will Download the file
    Web_Page = urllib.request.urlopen(Address)
    #Create the file in the path and puts its rights (wb or read & write)
    fichero = open(Path,"wb")
    #Write the file in the "fichero" variable
    fichero.write(Web_Page.read())
    #Close the file
    fichero.close()
    #Now it is gonna read the file created
    read_file = pd.read_csv(Path, names = col_names,skiprows = 10 )
    #And return that file
    return read_file

#Link of the File
Address = "https://www.blackrock.com/us/individual/products/239670/ishares-msci-mexico-capped-etf/1464253357814.ajax?fileType=csv&fileName=EWW_holdings&dataType=fund"
#Where is gonna be saved and readed
Path = "Bases\\ "
#Which name it will have
Name = "EWW_holdings_"+DateTODAY+".csv"
#The names of the columns
col_names=['Ticker','Name','Asset Class','Weight (%)','Price','Shares','Market Value',
           'Notional Value','Sector','SEDOL','ISIN','Exchange','Location','Currency',
           'Market Currency','FX Rate']
#Download the file
data_for_BM = Download_file(Address, Path, Name, col_names)
#Returns historical or expected

#Link of the File
Address = "https://www.blackrock.com/us/individual/products/239637/ishares-msci-emerging-markets-etf/1464253357814.ajax?fileType=csv&fileName=EEM_holdings&dataType=fund"
#Where is gonna be saved and readed
Path = "Bases\\ "
#Which name it will have
Name = "EEM_holdings_"+DateTODAY+".csv"
#The names of the columns
col_names=['Ticker','Name','Asset Class','Weight (%)','Price','Shares','Market Value',
           'Notional Value','Sector','SEDOL','ISIN','Exchange','Location','Currency',
           'Market Currency','FX Rate']
#Download the file
data_for_BM = Download_file(Address, Path, Name, col_names)
#Returns historical or expected

#Link of the File
Address = "https://www.blackrock.com/us/individual/products/239600/ishares-msci-acwi-etf/1464253357814.ajax?fileType=csv&fileName=ACWI_holdings&dataType=fund"
#Where is gonna be saved and readed
Path = "Bases\\ "
#Which name it will have
Name = "ACWI_holdings_"+DateTODAY+".csv"
#The names of the columns
col_names=['Ticker','Name','Asset Class','Weight (%)','Price','Shares','Market Value',
           'Notional Value','Sector','SEDOL','ISIN','Exchange','Location','Currency',
           'Market Currency','FX Rate']
#Download the file
data_for_BM = Download_file(Address, Path, Name, col_names)
#Returns historical or expected


#Link of the File
Address = "https://www.blackrock.com/us/individual/products/239726/ishares-core-sp-500-etf/1464253357814.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
#Where is gonna be saved and readed
Path = "Bases\\ "
#Which name it will have
Name = "IVV_holdings_"+DateTODAY+".csv"
#The names of the columns
col_names=['Ticker','Name','Asset Class','Weight (%)','Price','Shares','Market Value',
           'Notional Value','Sector','SEDOL','ISIN','Exchange','Location','Currency',
           'Market Currency','FX Rate']
#Download the file
data_for_BM = Download_file(Address, Path, Name, col_names)
#Returns historical or expected

#Link of the File
Address = "https://www.blackrock.com/us/individual/products/239659/ishares-msci-india-etf/1464253357814.ajax?fileType=csv&fileName=INDA_holdings&dataType=fund"
#Where is gonna be saved and readed
Path = "Bases\\ "
#Which name it will have
Name = "INDA_holdings_"+DateTODAY+".csv"
#The names of the columns
col_names=['Ticker','Name','Asset Class','Weight (%)','Price','Shares','Market Value',
           'Notional Value','Sector','SEDOL','ISIN','Exchange','Location','Currency',
           'Market Currency','FX Rate']
#Download the file
data_for_BM = Download_file(Address, Path, Name, col_names)
#Returns historical or expected
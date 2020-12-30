# =============================================================================
#                                  Libraries
# =============================================================================

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import date
import pathlib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
global etf_name
import matplotlib.ticker as ticker

# =============================================================================
#                                   Load Data
# =============================================================================

bandera=1
today_date=date.today()
#depending on how many data you have you can change the days
initial_date=date.today()- timedelta(days=40)

shares = pd.DataFrame(data={})
mkt_cap= pd.DataFrame(data={})
sector= pd.DataFrame(data={})

#We declare the dataframes with or that we will divide the historical files 
contador=1
findesem=False
etf_name="IVV_holdings_"
list_name=[etf_name]


while(initial_date<today_date):# we will find the files based on the dates we enter and adding up day by day 

 
    
    initial_date=initial_date+ timedelta(days=1)

    in_date=str(initial_date)
    
    bandera=bandera+1
    for i in range(len(list_name)):
        try:
                
            try:
                #First we try to download the file without counting the first rows and comma spreaders 

    
                df2=pd.read_csv(list_name[i]+in_date+".csv",header=1,delimiter=",",decimal=',', skiprows=[i for i in range(0,8)])
                df2["Market Value"]=df2["Market Value"].str.replace(',','')
                df2["Market Value"]=pd.to_numeric(df2["Market Value"], downcast="float")
                df2 = df2[df2["Market Value"] >10000]
                df2 = df2[df2["Asset Class"]=="Equity"]
                shares[in_date]=df2["Shares"]
                mkt_cap[in_date]=df2["Market Value"]
                 # We save different variables in weight, price, stock and market value dataframes 
                if(bandera>1):
                    sector[in_date]=df2["Sector"]
                    #WE save also the type of sector of each action, we only need to save it once 
                    bandera=-60
            
            except:
                df2=pd.read_csv(list_name[i]+in_date+".csv",header=1,delimiter=",",decimal=',', skiprows=[i for i in range(0,8)])
                df2["Valor de mercado"]=df2["Valor de mercado"].str.replace(',','')
                df2["Valor de mercado"]=pd.to_numeric(df2["Valor de mercado"], downcast="float")
                df2 = df2[df2["Valor de mercado"] >10000]
                df2 = df2.fillna(0)


                df2 = df2[df2["Clase de activo"]=="Equity"]
                shares[in_date]=df2["Acciones"]
                mkt_cap[in_date]=df2["Valor de mercado"]
                
            #Despues de tener los datos vamos a convertir en formato numerico, no sin antes eliminar todas la comas por completo
            #After having the data we will convert to numeric format, not without first deleting all commas completely 

                 

            #To be able to refer to the data that I want uniformly, we put all the dataframes their ticker 


        except:
            
            findesem=True
            #this will help us to prevent the program from breaking when you want to download non-existent data for weekend or holidays     if(findesem==True):
        findesem=False
    else:
        contador=contador+1

sector["Ticker"]=df2["Ticker"] 
mkt_cap["Ticker"]=df2["Ticker"]
shares["Ticker"]=df2["Ticker"]  
     
shares= shares.set_index('Ticker')
mkt_cap= mkt_cap.set_index('Ticker')
sector= sector.set_index('Ticker')

#then we apply the transpose of the matrices so that we can graph and call the data more effectively 

 
shares= shares.T
mkt_cap= mkt_cap.T
sector= sector.T
mkt_cap=mkt_cap.fillna(0)
sector=sector.fillna("-")


# By having them sorted we can get the total of different data 

# We declare global variables that we will use for the graphical interface 

 
global dispac,aelim,sect,secafil,tac
import numpy as np

dispac=[]
aelim=[]
tac=[]
sect=[]
secafil=[]
Nanencon=0
for name in mkt_cap.columns:
    if name!="-"or name!="Filtro"or name!="TOTAL":
        dispac.append(name)
#We also get the different sectors for the GUI 
    
tac=dispac
for name in sector.iloc[0, 0:].values:
    if name!="-"or name!="Filtro"or name!="TOTAL":
        try:
            sect.append(name)
        except:
            Nanencon=Nanencon+1
sect = list(dict.fromkeys(sect))


from tkinter import*
from tkinter.ttk import Combobox
#Con estas dos liberias podremos usar y desplegar la interfaz grafica

# =============================================================================
#                                   Function
# =============================================================================

def cambiardeindex():
    global shares,mkt_cap,sector
        
    bandera=1
    today_date=date.today()
    
    initial_date=date.today()- timedelta(days=22)
    
    shares= pd.DataFrame(data={})
    mkt_cap= pd.DataFrame(data={})
    sector= pd.DataFrame(data={})
    
    contador=1
    findesem=False
    etf_name=Cbox1.get()  #This is the only thing other than the main download because with this name we rely 
    list_name=[etf_name]
    while(initial_date<today_date):
        
        initial_date=initial_date+ timedelta(days=1)
        in_date=str(initial_date)
        
        bandera=bandera+1
        for i in range(len(list_name)):
            try:
                    
                try:
                  
                    df2=pd.read_csv(list_name[i]+in_date+".csv",header=1,delimiter=",",decimal=',', skiprows=[i for i in range(0,8)])
                    df2["Market Value"]=df2["Market Value"].str.replace(',','')
                    df2["Market Value"]=pd.to_numeric(df2["Market Value"], downcast="float")
                    df2 = df2[df2["Market Value"] >10000]

                    df2 = df2[df2["Asset Class"]=="Equity"]
                    shares[in_date]=df2["Shares"]
                    mkt_cap[in_date]=df2["Market Value"]
                    if(bandera>1):
                        sector[in_date]=df2["Sector"]
                        bandera=-60
                
                except:
                    df2=pd.read_csv(list_name[i]+in_date+".csv",header=1,delimiter=",",decimal=',', skiprows=[i for i in range(0,8)])
                    df2["Valor de mercado"]=df2["Valor de mercado"].str.replace(',','')
                    df2["Valor de mercado"]=pd.to_numeric(df2["Valor de mercado"], downcast="float")
                    df2 = df2[df2["Valor de mercado"] >10000]
                    df2 = df2.fillna(0)
  
                    df2 = df2[df2["Clase de activo"]=="Equity"]
                    shares[in_date]=df2["Acciones"]
                    mkt_cap[in_date]=df2["Valor de mercado"]

    
            except:
                
                findesem=True
        if(findesem==True):
            findesem=False
        else:
            contador=contador+1
  
    sector["Ticker"]=df2["Ticker"] 
    mkt_cap["Ticker"]=df2["Ticker"]
    shares["Ticker"]=df2["Ticker"]  
         
    shares= shares.set_index('Ticker')
    mkt_cap= mkt_cap.set_index('Ticker')
    sector= sector.set_index('Ticker')
    

    shares= shares.T
    mkt_cap= mkt_cap.T
    sector= sector.T
    mkt_cap=mkt_cap.fillna(0)
    sector=sector.fillna("-")

    
    global dispac,aelim,sect,secafil,tac
    import numpy as np
    dispac=[]
    aelim=[]
    tac=[]
    sect=[]
    secafil=[]
    Nanencon=0
    for name in mkt_cap.columns:
        if name!="-"or name!="Filtro"or name!="TOTAL":
            dispac.append(name)
            
    tac=dispac
    for name in sector.iloc[0, 0:].values:
        if name!="-"or name!="Filtro"or name!="TOTAL":
            try:
                sect.append(name)
            except:
                Nanencon=Nanencon+1
    sect = list(dict.fromkeys(sect))


def grafica():
    cont=1
    cambios=0
    contar=-1
    converf=[]
    convert=[]
    mkt_cap["TOTAL"]=0
    mkt_cap['Filtro']=0
    mkt_cap["TOTAL"] = mkt_cap.sum(axis=1)
# We reset the values of the total and the new filter to be applied     
    for name in aelim:
    
        if cont!=1:
            mkt_cap['Filtro']=mkt_cap['Filtro']-mkt_cap[name]
            cambios=cambios+1
        else:
            mkt_cap['Filtro']=mkt_cap['TOTAL'] -mkt_cap[name]
# if it's the first variable to delete we should save total and start subtracting it as a first statement            cambios=cambios+1
            cont=2
    if cambios==0:
        mkt_cap['Filtro']=mkt_cap["TOTAL"]
# To avoid errors with a lack of change we condition the final equality so as not to plot a line in 0    
    for i in mkt_cap['TOTAL']:
        contar=contar+1
        try:
            convert.append((convert[0]*i)/mkt_cap['TOTAL'][0])
        except:
            convert.append(100)
    mkt_cap["TOTAL %"]=convert
    contar=-1
    for i in mkt_cap['Filtro']:
        contar=contar+1
        try:
            converf.append((converf[0]*i)/mkt_cap['Filtro'][0])
        except:
            converf.append(100)
    mkt_cap["Filtro %"]=converf   
    
    fig, ax1 = plt.subplots()
    color = '#006d91'
    color2="#d1c000"
    ax1.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Performance', color=color)
    ax1.plot(mkt_cap.index.tolist(),mkt_cap["TOTAL %"],color=color)
    ax1.plot(mkt_cap.index.tolist(),mkt_cap["Filtro %"],color=color2)
    leyendas=[]
    if Cbox1.get()!="":
        leyendas.append(Cbox1.get())
    else:
        leyendas.append(etf_name)
    leyendas.append("Etf sin acciones elegidas ")
    plt.legend(leyendas)
    ax1.tick_params(axis='y')
    plt.title("Resultado")
    plt.show
#We plot, get more legends to deploy and color asign the graph 

 
def graficaSec():
	    # We do the same thing as in sectors only now we have certain actions to filter     
    cambios=0
    converf=[]
    convert=[]
    contar=-1
    mkt_cap["TOTAL"]=0
    mkt_cap['Filtro']=0
    mkt_cap['Filtro %']=0
    mkt_cap["TOTAL"] = mkt_cap.sum(axis=1)
    cont=1

    for name in tac:
        try:
            if sector[name].item()==Cboxs.get():
                if cont!=1:
                    mkt_cap['Filtro']=mkt_cap['Filtro']-mkt_cap[name]
                    cambios=cambios+1


                else:
                    mkt_cap['Filtro']=mkt_cap['TOTAL'] -mkt_cap[name]



                    
                    cont=2
                    cambios=cambios+1
            else:
                pass
        
        except:
            pass
    if cambios==0:
        mkt_cap['Filtro']=mkt_cap["TOTAL"]
        
        
    for i in mkt_cap['TOTAL']:
        contar=contar+1
        try:
            convert.append((convert[0]*i)/mkt_cap['TOTAL'][0])
        except:
            convert.append(100)
    mkt_cap["TOTAL %"]=convert
    contar=-1
    for i in mkt_cap['Filtro']:
        contar=contar+1
        try:
            converf.append((converf[0]*i)/mkt_cap['Filtro'][0])
        except:
            converf.append(100)
    mkt_cap["Filtro %"]=converf
            
  
    fig, ax1 = plt.subplots()
    
    color = '#006d91'
    color2="#d1c000"
    ax1.set_xlabel('Time')
    ax1.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax1.set_ylabel('Performance', color=color)
    ax1.plot(mkt_cap.index.tolist(),mkt_cap["TOTAL %"],color=color)
    ax1.plot(mkt_cap.index.tolist(),mkt_cap["Filtro %"],color=color2)
    leyendas=[]
    if Cbox1.get()!="":
        leyendas.append(Cbox1.get())
    else:
        leyendas.append(etf_name)
    leyendas.append("Etf sin "+ Cboxs.get())
    plt.legend(leyendas)
    ax1.tick_params(axis='y')
    plt.title("Resultado")
    plt.show
	
#---------------------------------------------------------------------------------------------------------------------------------
#                                             GRAPHIC USER INTERFACE (GUI)
#
#---------------------------------------------------------------------------------------------------------------------------------

def agregaracc():
    
  accion=Cbox.get()
  dispac.remove(accion)
  Cbox['values']=dispac
  aelim.append(accion)
  # We get the combobox action, remove it from the actions to be deployed so that we can not "remove it double" 

  #WITH this combobox values are reset 


def SectoresPant():
  global SectoresPant
  global Cboxs
  screen3 = Toplevel(menu)
  screen3.title("Sector filter")
  screen3.geometry("500x350")
  Label(screen3, text = "Enter Sector",bg = "grey",pady=10, width = "300", height = "2", font = ("Calibri", 13)).pack()
  Label(screen3, text = "").pack()
  #We define Screen3 in this case so that it is on top, we give it its size and to enter text we put it as label 

  Cboxs= Combobox(screen3)
  Cboxs['values']=sect
  Cboxs.pack(pady=20)
  #with pack we deploy an object to the iterface
  Button(screen3,text = "Apply filter", width = 15, height = 1,pady=10, command = graficaSec).pack()
  
def Eliminac():
    #We define our global variables that we will use when switching filters 
  global SectoresPant
  global Cbox
  #Deploy screen 2 above principal screen
  screen2 = Toplevel(menu)
  screen2.title("Stock filter")
  screen2.geometry("500x350")
  Label(screen2, text = "Enter stocks to filter",bg = "grey",pady=10, width = "300", height = "2", font = ("Calibri", 13)).pack()
  Label(screen2, text = "").pack()
  #we fill combobox content and define size height and padding
  Cbox= Combobox(screen2)
  Cbox['values']=dispac
  Cbox.pack(pady=20)
  
  Button(screen2,text = "Add", width = 15,pady=10, height = 1, command = agregaracc).pack()
  Button(screen2,text = "Apply filter", width = 15,pady=10, height = 1, command = grafica).pack()
#Each button calls distinct action


# =============================================================================
#                               Main Program
# =============================================================================

root=Tk()
#Our root allows us to run de GUI for an indetermined amount of time

menu=Menu(root)
root.config(menu=menu)
#we reference our menu root as menu
global Cbox1
subMenu=Menu(menu)
#aadding submenus un cascade calling diferent functions
menu.add_cascade(label="New Filter", menu=subMenu)
subMenu.add_command(label="Sector filter",command=SectoresPant)
subMenu.add_command(label="Filter certain stocks",command=Eliminac)
root.geometry("300x250")
root.title("Index tool")
#Defining labels and tittle
Label(text = "Index Tool", bg = "grey",pady=10, width = "300", height = "2", font = ("Calibri", 13)).pack()
Cbox1= Combobox()
Cbox1['values']=['EEM_holdings_','ACWI_holdings_','IVV_holdings_','EWW_holdings_','INDA_holdings_']
Cbox1.pack(pady=20)
Button(text = "Change index", width = 15, height = 1,pady=10,borderwidth=3, command = cambiardeindex).pack()
#Looping our root indefenitly
root.mainloop()

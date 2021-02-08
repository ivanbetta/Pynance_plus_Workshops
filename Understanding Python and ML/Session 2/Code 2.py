
# Data Curation

'The objective of this code is understand data visualization'

# Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from vega_datasets import data
import seaborn as sns
from plotly.offline import init_notebook_mode,  plot
from plotly import graph_objs as go

# MatplotLib

np.random.seed(100)
y = np.random.standard_normal(25)

'A simple x/y plot'
x = range (len(y)) # Match x and y
plt.plot(x,y)
plt.show()

'Plotting the Cumulative Sum of Data'
plt.plot(y.cumsum())
plt.show()

'A better plot representation'
plt.figure(figsize=(10,5)) # width and height
plt.plot(y.cumsum(), 'b', lw=1.5)
plt.plot(y.cumsum(), 'ro')
plt.grid(True)
plt.axis ('tight')
plt.xlabel('Time')
plt.ylabel('Index')
plt.title ('Representative Cumulative plot')
plt.show()

'Plot with labeled datasets¶'
np.random.seed(100)
y = np.random.standard_normal((25,2)).cumsum(axis=0)

plt.figure(figsize=(10,5))
plt.plot(y[:,0], lw=1.5, label = '1st DS')
plt.plot(y[:,1], lw=1.5, label = '2nd DS')
plt.plot(y,'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis ('tight')
plt.xlabel('Time')
plt.ylabel('Index')
plt.title ('Representative plot with two datasets')
plt.show()

'Plotting a histogram'
plt.figure(figsize=(10,5))
plt.hist(y, label = ['1st','2nd'], bins=25)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('Index Returns')
plt.ylabel ('Stock Returns')
plt.title ('Histogram')
plt.show()

y = np.random.standard_normal((100,2))

'Scatter Plots'
plt.figure (figsize = (7,5))
plt.scatter(y[:,0], y[:,1], marker='o')
plt.grid(True)
plt.xlabel ('1st dataset')
plt.ylabel ('2nd dataset')
plt.title('Scatter Plot')
plt.show()

# Seaborn

'Scatterplot Matrix'

sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")

'Linear regression'

sns.set_theme(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.jointplot(x="total_bill", y="tip", data=tips,
                  kind="reg", truncate=False,
                  xlim=(0, 60), ylim=(0, 12),
                  color="m", height=7)

# Altair

titanic = sns.load_dataset("titanic")

titanic_chart = alt.Chart(titanic).mark_bar().encode(
    x='class',
    y='count()'
)

titanic_chart.show()

'Simple Heatmap'

'Compute x^2 + y^2 across a 2D grid'
x, y = np.meshgrid(range(-5, 5), range(-5, 5))
z = x ** 2 + y ** 2

'Convert this grid to columnar data expected by Altair'
source = pd.DataFrame({'x': x.ravel(),
                     'y': y.ravel(),
                     'z': z.ravel()})

heatmap = alt.Chart(source).mark_rect().encode(
    x='x:O',
    y='y:O',
    color='z:Q'
)
heatmap.show()

'Bar Chart with rounded edges'

source = data.seattle_weather()

bar_chart = alt.Chart(source).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x='month(date):O',
    y='count():Q',
    color='weather:N'
)
bar_chart.show()

'Choropleth Map'

counties = alt.topo_feature(data.us_10m.url, 'counties')
source = data.unemployment.url

us_map = alt.Chart(counties).mark_geoshape().encode(
    color='rate:Q'
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'id', ['rate'])
).project(
    type='albersUsa'
).properties(
    width=500,
    height=300
)

us_map.show()

# Ploty

'Candlestick'

init_notebook_mode()

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],
                high=df['AAPL.High'],
                low=df['AAPL.Low'],
                close=df['AAPL.Close'])])
fig.write_html("candlestick.html")
plot(fig)  

# References

'https://quantra.quantinsti.com/'
'https://matplotlib.org/'
'https://altair-viz.github.io/gallery/index.html'
'https://datapane.com/u/leo/reports/python-visualisation-guide-814e8638/'
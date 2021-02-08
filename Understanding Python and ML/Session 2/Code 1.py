
# Data Curation

'The objective of this code is understand more details for Lists Methods, Numpy and Pandas'

# Libraries

import numpy as np
from numpy import pi
from numpy import random as rg
from matplotlib import pyplot as plt 
import pandas as pd

# Lists Methods

prices = [1,2,3]
prices.append((4,5))
print(prices)

prices = [1,2,3]
prices.extend((8,5))
print(prices)

prices = [1,2,3]
prices.insert(3,'valor')
print(prices)

prices = [1,2,3]
prices.pop(1)
print(prices)

prices = [1,2,3,'number']
prices.remove('number')
print(prices)

prices = [1,2,3]
prices.reverse()
print(prices)

prices = [1,2,3,4,5,6,7,8,9,10]
print(prices[2:4])

prices = [100,2,350,22]
prices.sort()
print(prices)

# NumPy quickstart

"""
NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). 
Changing the size of an ndarray will create a new array and delete the original.

The elements in a NumPy array are all required to be of the same data type, and thus will be 
the same size in memory. The exception: one can have arrays of (Python, including NumPy) objects, 
thereby allowing for arrays of different sized elements.

NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. 
Typically, such operations are executed more efficiently and with less code than is possible using 
Python’s built-in sequences.

A growing plethora of scientific and mathematical Python-based packages are using NumPy arrays; 
though these typically support Python-sequence input, they convert such input to NumPy arrays prior 
to processing, and they often output NumPy arrays. In other words, in order to efficiently use much 
(perhaps even most) of today’s scientific/mathematical Python-based software, just knowing how to use 
Python’s built-in sequence types is insufficient - one also needs to know how to use NumPy arrays.

"""

'Array creation'

a = np.array([2,3,4])
print(a)

'arange'

b = np.arange(20).reshape(4, 5)
print(b)

print(np.arange( 10, 30, 5 ))

print(np.arange( 0, 2, 0.3 )  )

'linespace'

print(np.linspace( 0, 2, 9 )  )

x = np.linspace( 0, 2*pi, 100 ) # useful to evaluate function at lots of points
f = np.sin(x)
plt.plot(f)
plt.show()

'Basic operations'

i = np.array([20,30,40])
j = np.array([2,3,4])

k = i-j
print(k)

print(i**2)

print(j<3)

A = np.array( [[1,1],
               [0,1]] )
B = np.array( [[2,0],
               [3,4]] )
print(A*B)

a = rg.random((2,3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

C = np.arange(3)
print(np.exp(C))

'Indexing, slicing and iterating'

def f(x,y):
    return 10*x+y
b = np.fromfunction(f,(5,4),dtype=int)
print(b)

print(b[2,3])

print(b[0:5, 1]) # each row in the second column of b

print(b[ : ,1]) # equivalent to the previous example

print(b[1:3, : ]) # each column in the second and third row of b

'Indexing with Boolean Arrays'

def mandelbrot( h,w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime
plt.imshow(mandelbrot(20,20))
plt.show()

'Stacking together different arrays'

a = np.floor(10*rg.random((2,2)))
print(a)

b = np.floor(10*rg.random((2,2)))
print(b)

print(np.vstack((a,b)))

print(np.hstack((a,b)))

'Simple Array Operations'

a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)

print(a.transpose())

print(np.linalg.inv(a))

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
print(u)

j = np.array([[0.0, -1.0], [1.0, 0.0]])
print(j @ j)        # matrix product

print(np.trace(u))  # trace

y = np.array([[5.], [7.]])
print(np.linalg.solve(a, y))

print(np.linalg.eig(j))

'Histograms'

rg = np.random.default_rng(1)

'Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2'
mu, sigma = 2, 0.5
v = rg.normal(mu,sigma,10000)

'Plot a normalized histogram with 50 bins'
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)

'Compute the histogram with numpy and then plot it'
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()

# 10 minutes to pandas

'Object creation'

series = pd.Series([1, 3, 5, np.nan, 6, 8])
print(series)

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

'Viewing data'

print(df.head())

print(df.tail(3))

print(df.index)

print(df.columns)

"""
DataFrame.to_numpy() gives a NumPy representation of the underlying data. 
Note that this can be an expensive operation when your DataFrame has columns with different data types, .
which comes down to a fundamental difference between pandas and NumPy: 
    NumPy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column. 
    When you call DataFrame.to_numpy(), pandas will find the NumPy dtype that can hold all of the dtypes 
    in the DataFrame. This may end up being object, which requires casting every value to a Python object.

For df, our DataFrame of all floating-point values, DataFrame.to_numpy() is fast and doesn’t require copying data.
"""

print( df.to_numpy())

'Getting'

print(df["A"])

print( df[0:3])

print(df["20130102":"20130104"])

'Selection by label'

print(df.loc[dates[0]])

print( df.loc[:, ["A", "B"]])

print(df.loc["20130102":"20130104", ["A", "B"]])

print(df.loc["20130102", ["A", "B"]])

print(df.loc[dates[0], "A"])

print(df.at[dates[0], "A"])

'Selection by position'

print(df.iloc[3])

print(df.iloc[3:5, 0:2])

print(df.iloc[:, 1:3])

print(df.iat[1, 1])

print (df.iloc[-2:])

'Boolean indexing'

print(df[df["A"] > 0])

print(df[df > 0])

df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print(df2)

'Setting'

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
print(s1)

df["F"] = s1
df.at[dates[0], "A"] = 0
df.iat[0, 1] = 0
df.loc[:, "D"] = np.array([5] * len(df))
print(df)

'Missing data'

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0] : dates[1], "E"] = 1
print(df1)

print(df1.dropna(how="any"))

print(df1.fillna(value=5))

print(pd.isna(df1))

'Stats'

print(df.mean())

print(df.mean(1))

'Apply'

print(df)

print(df.apply(np.cumsum))

print(df.apply(lambda x: x.max() - x.min()))

'Histogramming'

s = pd.Series(np.random.randint(0, 7, size=10))
print(s)

print( s.value_counts())

'Concat'

df = pd.DataFrame(np.random.randn(10, 4))
print(df)

pieces = [df[:3], df[3:7], df[7:]]
print(pieces)

print(pd.concat(pieces))

'Pivot tables'

df = pd.DataFrame(
    {
         "A": ["one", "one", "two", "three"] * 3,
         "B": ["A", "B", "C"] * 4,
         "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
         "D": np.random.randn(12),
         "E": np.random.randn(12),
    }
)
print(df)

print(pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]))

'Time series'

rng = pd.date_range("1/1/2012", periods=100, freq="S")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

print(ts_utc.tz_convert("US/Eastern"))

rng = pd.date_range("1/1/2012", periods=5, freq="M")
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)

ps = ts.to_period()
print(ps)

print(ps.to_timestamp())

duration = pd.date_range('1/1/2018', periods=100, freq='Min')
ts = np.random.seed(100)
ts = pd.Series(np.random.randint(0, 5000, len(duration)), index=duration)
ts = ts.asfreq('10Min', method='pad')
print(ts)

'Plotting'

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

'CSV'

df.to_csv("foo.csv")

print(pd.read_csv("foo.csv"))

'Excel'

df.to_excel("foo.xlsx", sheet_name="Sheet1")

print(pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"]))

# References

'https://www.w3schools.com/python/python_ref_list.asp'
'https://numpy.org/doc/stable/user/index.html'
'https://numpy.org/doc/stable/reference/index.html'
'https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html'
'https://pandas.pydata.org/pandas-docs/stable/reference/index.html'
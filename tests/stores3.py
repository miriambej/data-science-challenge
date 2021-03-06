# The data (Weekly_Sales, Size, Population_under_18, Population) are for each store.
# Weekly_Sales = sales for the store
# Size = the size of the store
# Population_under_18 = whether the week is a special holiday week
# Population = Population in the region

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('storespop.xls')
X = df.as_matrix()

#plot the data
plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

plt.scatter(X[:,3], X[:,0])
plt.show()

df['ones'] = 1
Y = df['Weekly_Sales']
X = df[['Size', 'Population_under_18', 'Population', 'ones']]
Sizeonly = df[['Size', 'ones']]
Population_under_18only = df[['Population_under_18', 'ones']]
Populationonly = df[['Population', 'ones']]

#function
def get_r2(X, Y):
    w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print "r2 for Size only:", get_r2(Sizeonly, Y)
print "r2 for Population_under_18 only:", get_r2(Population_under_18only, Y)
print "r2 for Population only:", get_r2(Populationonly, Y)
print "r2 for all:", get_r2(X, Y)

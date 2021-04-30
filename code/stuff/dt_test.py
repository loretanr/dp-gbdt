import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# example of useless DT, validation data is not related to traning data

x = np.linspace(0,1)
y = x + np.random.uniform(-0.2,0.2,x.shape)
plt.scatter(x,y)


# train, validation set split
x_trn, x_val = x[:40,None], x[40:,None]
y_trn, y_val = y[:40,None], y[40:,None]

# fit a model
m = DecisionTreeRegressor(max_depth=6).fit(x_trn, y_trn)

plt.scatter(x_val,m.predict(x_val),color='blue',label='Prediction')
plt.scatter(x_val,y_val,color='red',label='Actual')
plt.scatter(x_trn,m.predict(x_trn),color='blue')
plt.scatter(x_trn,y_trn,color='red')
plt.legend(loc='upper left')
plt.show()

def bla():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression

    plt.figure(figsize=(20,10))
    x = np.linspace(0,2,10000)
    y = 1+ 3*x
    plt.scatter(x,y)
    plt.show()
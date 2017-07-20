import numpy as np

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=42423423):
    np.random.seed(seed)
    dimension = 10
    size= 1000 * dimension
    x_data = np.random.random(size).reshape(dimension,-1).T
    y_data = 2 * x_data + 3 #linear function
    return x_data, y_data[:,0]


def best4RT(seed=42423423):
    np.random.seed(seed)
    dimension = 2
    size=1000 * dimension
    X = np.random.random(size).reshape(dimension,-1).T
    Y =X ** 3 # non-linear function 
    return X, Y[:,0]

if __name__ == "__main__":

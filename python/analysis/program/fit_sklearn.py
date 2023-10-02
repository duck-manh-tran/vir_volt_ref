import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def generateData(n = 30):
    # similar to peaks() function in MATLAB
    g = np.linspace(-3.0, 3.0, n)
    X, Y = np.meshgrid(g, g)
    X, Y = X.reshape(-1,1), Y.reshape(-1,1)
    Z = 3 * (1 - X)**2 * np.exp(- X**2 - (Y+1)**2) \
        - 10 * (X/5 - X**3 - Y**5) * np.exp(- X**2 - Y**2) \
        - 1/3 * np.exp(- (X+1)**2 - Y**2)
    return X, Y, Z

def names2model(names):
    # C[i] * X^n * Y^m
    return ' + '.join([
        f"C[{i}]*{n.replace(' ','*')}"
        for i,n in enumerate(names)])

# generate some random 3-dim points
X, Y, Z = generateData()

# 1=linear, 2=quadratic, 3=cubic, ..., nth degree
order = 11

# best-fit polynomial surface
model = make_pipeline(
    PolynomialFeatures(degree=order),
    LinearRegression(fit_intercept=False))
model.fit(np.c_[X, Y], Z)

m = names2model(model[0].get_feature_names_out(['X', 'Y']))
C = model[1].coef_.T  # coefficients
r2 = model.score(np.c_[X, Y], Z)  # R-squared

# print summary
print(f'data = {Z.size}x3')
print(f'model = {m}')
print(f'coefficients =\n{C}')
print(f'R2 = {r2}')

# uniform grid covering the domain of the data
XX,YY = np.meshgrid(np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20))

# evaluate model on grid
ZZ = model.predict(np.c_[XX.flatten(), YY.flatten()]).reshape(XX.shape)

# plot points and fitted surface
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X, Y, Z, c='r', s=2)
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
ax.axis('tight')
ax.view_init(azim=-60.0, elev=30.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
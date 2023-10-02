import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def generateData(n = 20):
    # similar to peaks() function in MATLAB
    g = np.linspace(-3.0, 3.0, n)
    X, Y = np.meshgrid(g, g)
    X, Y = X.reshape(-1,1), Y.reshape(-1,1)
    Z = 3 * (1 - X)**2 * np.exp(- X**2 - (Y+1)**2) \
        - 10 * (X/5 - X**3 - Y**5) * np.exp(- X**2 - Y**2) \
        - 1/3 * np.exp(- (X+1)**2 - Y**2)
    return X, Y, Z

def exp2model(e):
    # C[i] * X^n * Y^m
    return ' + '.join([
        f'C[{i}]' +
        ('*' if x>0 or y>0 else '') +
        (f'X^{x}' if x>1 else 'X' if x==1 else '') +
        ('*' if x>0 and y>0 else '') +
        (f'Y^{y}' if y>1 else 'Y' if y==1 else '')
        for i,(x,y) in enumerate(e)
    ])

# generate some random 3-dim points
# X, Y, Z = generateData()

data = np.loadtxt(f"data.txt", delimiter='\t')
X = data[:, 0]
X = X.reshape(-1, 1)
Y = data[:, 1]
Y = Y.reshape(-1, 1)
Z = data[:, 2]
Z = Z.reshape(-1, 1)

# 1=linear, 2=quadratic, 3=cubic, ..., nth degree
order = 5

# calculate exponents of design matrix
#e = [(x,y) for x in range(0,order+1) for y in range(0,order-x+1)]
e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
eX = np.asarray([[x] for x,_ in e]).T
eY = np.asarray([[y] for _,y in e]).T

# best-fit polynomial surface
A = (X ** eX) * (Y ** eY)
C,resid,_,_ = lstsq(A, Z)    # coefficients
print(resid)
# calculate R-squared from residual error
# r2 = 1 - resid[0] / (Z.size * Z.var())

# print summary
print(f'data = {Z.size}x3')
print(f'model = {exp2model(e)}')
print(f'coefficients =\n{C}')
# print(f'R2 = {r2}')

# uniform grid covering the domain of the data
XX,YY = np.meshgrid(np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20))

# evaluate model on grid
A = (XX.reshape(-1,1) ** eX) * (YY.reshape(-1,1) ** eY)
ZZ = np.dot(A, C).reshape(XX.shape)

B = (X ** eX) * (Y ** eY)

np.savetxt(f"C_coeff.txt", C, delimiter='\t')
Z_cap = np.dot(B, C).reshape(X.shape)
LR_e = [(100 * abs(Z_cap[iz] - Z[iz])/Z[iz]) for iz in range(len(Z_cap))]
LR_e = np.concatenate( LR_e, axis=0 )
print (LR_e)
bx = plt.figure().add_subplot()
bx.plot(LR_e)

# plot points and fitted surface
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X, Y, Z, c='r', s=2)
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
ax.axis('tight')
ax.view_init(azim=-60.0, elev=30.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.show()

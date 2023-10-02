import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math

def main():
# Measurement process

    filename = 'time2vol.txt'
    data = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype=str)
    index = list(map(int, data[:,0]))
    time = list(map(float, data[:,1]))
    capa = list(map(float, data[:,2]))
    
    # setup initial points and stop points
    ipoint = [];
    VDD = [];
    k_thres = 0.25
    k = [0.4, 0.3, 0.25]
    m_ = [0, 0, 0]
    m = [0, 0, 0]
    n = [0, 0, 0]
    y = []
    r = []
    r_cap = []
    for iv in range(21):  #(41):
        VDD.append(0.4 + 0.02 * iv)
        for ir in range(26):     #(126):
            r.append(round(1/VDD[iv], 8))
            r_cap.append(round (1.25 + 0.05 * ir, 4))
            for im in range(3):
                m[im] = round (pow(2, 16) * r_cap[ir] * k[im])
                m_[im] = m[im]/pow(2, 16)
                v_start = VDD[iv] * m_[im]
                v_stop = VDD[iv] * k_thres
                n[im] = round (pow(10, 6) * meas_discharge_time (time, capa, v_start, v_stop))
            y_ = math.exp( (n[0] - n[1])/(n[1] - n[2]) + 4 )
            y.append(round (y_, 2))
    y_bar = [ 1/item for item in y ]
    # yt = np.reshape (y, (5, 26))
    # print (yt)
    result = list(zip(r_cap, y, r))
    np.savetxt(f"data.txt", result, fmt='%10.6f', delimiter='\t')
    plotting3d(r_cap, y, r)

# curve fitting process

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
    
    # uniform grid covering the domain of the data
    XX,YY = np.meshgrid(np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20))
    
    # evaluate model on grid
    A = (XX.reshape(-1,1) ** eX) * (YY.reshape(-1,1) ** eY)
    ZZ = np.dot(A, C).reshape(XX.shape)
    
    C_ = np.around(C, 15)
    print (C_)
    B = (X ** eX) * (Y ** eY)
    np.savetxt(f"C_coeff.txt", C, delimiter='\t')
    Z_cap = np.dot(B, C_).reshape(X.shape)
    LR_e = [(100 * abs(Z_cap[iz] - Z[iz])/Z[iz]) for iz in range(len(Z_cap))]
    LR_e = np.concatenate( LR_e, axis=0 )
    #print (LR_e)
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
    plt.show()

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

def meas_discharge_time (time, data, start_val, stop_val):
    t_start = meas_when (time, data, start_val)
    t_stop = meas_when (time, data, stop_val)
    t_diff = t_stop - t_start
    return t_diff

def meas_when (time, value, point):
    value_ = [ abs( item - point ) for item in value ]
    minpos = value_.index(min(value_))
#    print (minpos)
    return time[minpos]

def plotting2d (x, y):
#    plt.rc('lines', linewidth=0.5)
    fig, ax = plt.subplots()
    plot_y = ax.plot(x, y, 'o')
    plt.grid (visible=True, which='major', axis='both')
    plt.show()

def plotting3d (x, y, z):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()

main()

import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    # load data from txt file to python

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

'''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(r_cap, y, 'o')
    ax2.plot(r_cap, y_bar, 'o')
    plt.grid (visible=True, which='major', axis='both')
    plt.show()
'''
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

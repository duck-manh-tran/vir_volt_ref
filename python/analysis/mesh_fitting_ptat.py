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
    for iv in range(4):  #(41):
        VDD.append(0.7 + 0.01 * iv)
        for ir in range(10):     #(126):
            r.append(1/VDD[iv])
            r_cap.append(2.2 + 0.01 * ir)
            for im in range(3):
                m[im] = round (pow(2, 16) * r_cap[ir] * k[im])
                m_[im] = m[im]/pow(2, 16)
                v_start = VDD[iv] * m_[im]
                v_stop = VDD[iv] * k_thres
                n[im] = round (pow(10, 6) * meas_discharge_time (time, capa, v_start, v_stop))
            y_ = math.exp( (n[0] - n[1])/(n[2] - 1) - 2 ) 
            y.append(y_)
            print (m)
            print (n)
            print (y_)


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

def plotting2d (t_d, vol, p_len):
#    plt.rc('lines', linewidth=0.5)
    fig, ax = plt.subplots()
    line = []
    for i in range (p_len):
        line.append(i)
        line[i], = ax.plot(vol, t_d[i], 'o')
    plt.grid (visible=True, which='major', axis='both')
    plt.show()

main()

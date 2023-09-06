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
    t_dis = []
    for iv in range(41):  #(41):
        VDD.append(0.4 + 0.01 * iv)

    for ip in range(8):
        ipoint.append (0.3 + 0.1 * ip)
        t_dis.append([])
        for iv in range(41):  #(41):
                v_start = VDD[iv] * ipoint[ip]
                v_stop = VDD[iv] * k_thres
                t_dis[ip].append( (round (pow(10, 6) * meas_discharge_time (time, capa, v_start, v_stop))))
    plotting2d (t_dis, VDD, len(t_dis))

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
        line[i], = ax.plot(vol, t_d[i])
    plt.grid (visible=True, which='major', axis='both')
    plt.show()

main()

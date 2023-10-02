import matplotlib.pyplot as plt
import numpy as np
import math
def main():
	 for ix in range(file_range):
		  infile = "result/temp/text/discharge_nwk_" + str(20+ix*2) + "C.txt"
		  cmd="sed -e '/\o14/d' -e '/sch_path:/d' -e '/---/d' -e '/Index/d' -e '/Transient/d' -i " + infile
		  os.system(cmd)
	 

    # load data from txt file to python

    filename = 'time2vol.txt'
    data = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype=str)
    index = list(map(int, data[:,0]))
    time = list(map(float, data[:,1]))
    capa = list(map(float, data[:,2]))
    
    # setup initial points and stop points
    k = [0.4, 0.3, 0.25]
    m_ = [0, 0, 0]
    m = [0, 0, 0]
    n = [0, 0, 0]
    y = []
    r = []

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
    result = list(zip(r_cap, y, r))
    np.savetxt(f"data.txt", result, fmt='%10.6f', delimiter='\t')
    plotting3d(r_cap, y, r)



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

# import matplotlib.pyplot as plt
import numpy as np
import math
import os
def main():
# cleaning the data files
	for ix in range(5):
		infile = "result/temp/text/discharge_nwk_" + str(20+ix*2) + "C.txt"
		outfile = "result/temp/text/discharge_nwk_" + str(20+ix*2) + "C_data.txt"
		cmd="sed -e '/\o14/d' -e '/sch_path:/d' -e '/---/d' -e '/Index/d' -e '/Transient/d' -i " + infile
		os.system(cmd)
# measure on each of files
		filename = infile
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
		n1 = []
		n2 = []
		n3 = [] 	
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
				n1.append(n[0])
				n2.append(n[1])
				n3.append(n[2])
				y_ = math.exp( (n[0] - n[1])/(n[1] - n[2]) + 4 )
				y.append(round (y_, 2))
		y_bar = [ 1/item for item in y ]
		# yt = np.reshape (y, (5, 26))
		# print (yt)
		result = list(zip(r_cap, y, n1, n2, n3, r))
		np.savetxt(outfile, result, fmt='%10.6f', delimiter='\t')
#    plotting3d(r_cap, y, r)

def meas_discharge_time (time, data, start_val, stop_val):
	t_start = meas_when (time, data, start_val)
	t_stop = meas_when (time, data, stop_val)
	t_diff = t_stop - t_start
	return t_diff

def meas_when (time, value, point):
	value_ = [ abs( item - point ) for item in value ]
	minpos = value_.index(min(value_))
#   print (minpos)
	return time[minpos]

main()

#import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math
from functions import *


def main():

	for ix in range(5):
		outfile = "sweep_ipoint_volt0" + str(ix+4) + ".txt"
		get_discharge_time_at_all_temps_sweep_points((ix+4)/10, outfile)


def get_discharge_time_at_all_temps_sweep_points(VDD, outfile):
	temp = []
	t_dis = []
	ipoint = [0.3+ix*0.05 for ix in range(15)]

	for it in range(50):
		t_dis.append([])
		temp.append(it*2)
		if (temp[it] < 10):
			filename = "result/temp/l180nm/nf_6/discharge_curve_0" + str(temp[it]) + "C.txt"
		else:
			filename = "result/temp/l180nm/nf_6/discharge_curve_" + str(temp[it]) + "C.txt"
		print (filename)
		data = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype=str)
		index = list(map(int, data[:,0]))
		time = list(map(float, data[:,1]))
		capa = list(map(float, data[:,2]))
		k_thres = 0.25
		
		v_stop = VDD * k_thres
		t_stop = meas_when(time, capa, v_stop)
		for ip in range(len(ipoint)):
			v_start = VDD * ipoint[ip]
			t_dis[it].append(round(meas_discharge_time2(time, capa, v_start, t_stop), 6))
	t_dis = np.array(t_dis)	
	outfile = "result/temp/l180nm/nf_6/" + outfile
	np.savetxt(outfile, t_dis, delimiter='\t', fmt='%10.6f')

main()

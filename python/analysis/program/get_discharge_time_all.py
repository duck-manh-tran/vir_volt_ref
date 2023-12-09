import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math
from functions import *

def main():
	get_discharge_time_all_temps_sweep_points()
	get_discharge_time_all_VDDs_sweep_points(10)
	get_discharge_time_all_VDDs_sweep_points(30)
	get_discharge_time_all_VDDs_sweep_points(60)
	get_discharge_time_all_VDDs_sweep_points(90)

def get_discharge_time_all_temps_sweep_points():
	temp 		= [it*8 for it in range(13)]
	VDD 		= [0.4 + 0.05*ix for ix in range(9)]
	ipoint 	= [0.3+ix*0.05 for ix in range(15)]
	t_dis = []
	temps = []
	VDDs  = []
	ipoints = []
	k_thres = 0.25

	for it in range(len(temp)):	#13
		if (temp[it] < 10):
			filename = "result/temp/l180nm/nf_6/discharge_curve_0" + str(temp[it]) + "C.txt"
		else:
			filename = "result/temp/l180nm/nf_6/discharge_curve_" + str(temp[it]) + "C.txt"
		print (filename)
		data = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype=str)
		index = list(map(int, data[:,0]))
		time = list(map(float, data[:,1]))
		capa = list(map(float, data[:,2]))
	
		for ix in range(len(VDD)):
			v_stop = VDD[ix]*k_thres
			t_stop = meas_when(time, capa, v_stop)
			t_dis_vdd = round(meas_discharge_time2(time, capa, VDD[ix], t_stop), 6)
			for ip in range(len(ipoint)):
				v_start = VDD[ix]*ipoint[ip]
				t_dis.append(t_dis_vdd/round(meas_discharge_time2(time, capa, v_start, t_stop), 6))
				temps.append(temp[it])
				VDDs.append(VDD[ix])
				ipoints.append(ipoint[ip])
	t_dis = np.array(t_dis)	
	temps = np.array(temps)
	VDDs  = np.array(VDDs)
	ipoints= np.array(ipoints)
	stacked_data = np.column_stack ((t_dis, temps, VDDs, ipoints)) 
	filtered_data = sorted(stacked_data, key=lambda x: x[3])	
	outfile = "result/temp/l180nm/nf_6/filtered_data.txt"
	np.savetxt(outfile, filtered_data, delimiter='\t', fmt='%10.6f')

def get_discharge_time_all_VDDs_sweep_points(temp):

	infile = "result/temp/l180nm/nf_6/discharge_curve_" + str(temp) + "C.txt"
	
	data = np.loadtxt(infile, delimiter='\t', skiprows=1, dtype=str)
	index = list(map(int, data[:,0]))
	time = list(map(float, data[:,1]))
	capa = list(map(float, data[:,2]))
	
	VDD = [0.4 + 0.05*ix for ix in range(9)]
	ipoint = [0.3+ix*0.05 for ix in range(15)]
	t_dis = []
	k_thres = 0.25
	
	for ix in range(len(VDD)):
		v_stop = VDD[ix]*k_thres	
		t_stop = meas_when(time, capa, v_stop)
		t_dis.append([])
		for ip in range(len(ipoint)):
			v_start = VDD[ix]*ipoint[ip]
			t_dis[ix].append(round(meas_discharge_time2(time, capa, v_start, t_stop), 6))
	t_dis = np.array(t_dis)
	VDD = np.array(VDD)
	result = np.column_stack((t_dis, VDD))
	print (result)
	outfile = "result/temp/l180nm/nf_6/sweep_ipoint_temp_" + str(temp) + "C.txt"
	np.savetxt(outfile, result, delimiter='\t', fmt='%10.6f')

main()

import numpy as np
import os
#time_cap = [5200, 432, 192, 84, 48, 24, 12, 6, 3.6, 2.4, 1.44]
time_cap = [2810, 1800, 750, 310, 135, 59, 27.5, 13.5, 6.9, 3.65, 2.03]
temp_cap = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
stop_time = []
number_points = 200000
for ix in range(10):
	for iy in range(5):
#		print ((ix*10+iy*2))
		stop_time.append((time_cap[ix]*(temp_cap[ix+1] - (ix*10+iy*2)) + time_cap[ix+1]*((ix*10+iy*2) - temp_cap[ix]))/10)
#		print (round(stop_time[5*ix+iy]))			
step_time = [round(10**6 * stop_time[ix]/number_points) for ix in range(len(stop_time))]
# print (np.array(step_time))

end_dir = " result/temp/l180nm"
prefix = "discharge_curve_"
for ix in range(10):
	cmd = 'ngspice discharge_nwk_180nm.spice'
	temp = ix*2
	
	arg1 = " -D step_time=" + str(step_time[ix]) + "n"
	arg2 = " -D stop_time=" + str(round(stop_time[ix], 1)) + "m"
	arg3 = " -D temp_c=" + str(temp)
	if (temp < 10): 
		outfile = prefix + "0" + str(temp) + "C.txt"
	else:
		outfile = prefix + str(temp) + "C.txt"
	arg4 = " -D outfile=" + outfile
	cmd = cmd + arg1 + arg2 + arg3 +arg4
	print(cmd)
	os.system(cmd)
# ngspice discharge_nwk_test.spice -D step_time=5u -D stop_time=1.060 -D temp=0 -D file1=file2.txt





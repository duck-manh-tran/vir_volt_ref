import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math

def main():

	analysis()
	plt.show()

def analysis():
	data_temp_10 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_temp_10C.txt", delimiter='\t')
	data_temp_30 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_temp_30C.txt", delimiter='\t')
	data_temp_60 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_temp_60C.txt", delimiter='\t')
	data_temp_90 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_temp_90C.txt", delimiter='\t')

	data = data_temp_90

	ipoint = [0.3+ix*0.05 for ix in range(15)]
	VDD = [0.4 + 0.05*ix for ix in range(9)]
# plot discharge time at different ipoints and temp changing
	ax = plt.figure().add_subplot()
	for ix in range(15):
		ax.plot(VDD, data[:, ix], label=str(round(ipoint[ix], 2), ))   
	ax.grid()	
	plt.title("Analysis discharge time of different initial points in the voltage domain")
	plt.xlabel("VDD (V)")
	plt.ylabel("Discharge time (s)")
	leg = plt.legend(loc='upper right')

# plot rate of t_dis at different ipoints with full t_dis in temp domain
	bx = plt.figure().add_subplot()
	for ix in range(14):
		rate = [data[ir, 14]/data[ir, ix] for ir in range(len(VDD))] 
		bx.plot(VDD, rate, label=str(round(ix*0.05 + 0.3, 2)))   
	bx.grid()	
	plt.title("Analysis rate of full discharge time vs discharge time of different initial points in voltage domain")
	plt.xlabel("VDD (V)")
	plt.ylabel("discharge rate")
	leg = plt.legend(loc='upper right')

# form a variable to analysis
	ip1 = 14
	ip2 = 10		
	ip4 = 11
	ip3 = ip1

	#var = [(data[ir, ip1] - data[ir, ip2])/(data[ir, ip3] - data[ir, ip4]) for ir in range(len(temp))] 
	var = [(data[ir, ip1])/(data[ir, ip3] - data[ir, ip4]) for ir in range(len(VDD))] 
#	var = [math.exp((data[ir, ip1])/(data[ir, ip3] - data[ir, ip4])) for ir in range(len(VDD))] 
	
	cx = plt.figure().add_subplot()
	cx.plot(VDD, var, label=str(round(ix*0.05 + 0.3, 2)))   
	plt.title("analysis any variable in the volt domain")
	plt.xlabel("VDD (V)")
	plt.ylabel("value")
	cx.grid()	
	leg = plt.legend(loc='upper right')





main()

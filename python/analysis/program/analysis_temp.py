import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math
import pandas as pd

def main():

	analysis_sweep_ipoint()
#	undefined_analysis()
	plt.show()

def undefined_analysis(): # developing
	data = np.loadtxt("result/temp/l180nm/nf_6/filtered_data.txt", delimiter='\t')
	tdis_rate = data[:, 0]
	temp = data[:, 1]
	volt = data[:, 2]
	tdis_rate_ = tdis_rate.reshape(int(len(data)/117), 117).T
	reshaped_data = np.column_stack((tdis_rate_, temp[0:117], volt[0:117]))
	np.savetxt(f"data_tmp.txt", reshaped_data, delimiter='\t', fmt='%10.6f')
	sorted_data = sorted(reshaped_data, key=lambda x: x[1])
	
	np.savetxt(f"result/temp/l180nm/nf_6/sorted_data.txt", sorted_data, delimiter='\t', fmt='%10.6f')
	ax = plt.figure().add_subplot()
	sorted_data = np.array(sorted_data)
	print (sorted_data[:, 0 ])
	ax.plot(sorted_data[:, 1 ])
	ax.plot(sorted_data[:, 2 ]) 
	ax.grid()
	plt.title("analysis any variable in the temp domain")
	plt.xlabel("temp (C)")
	plt.ylabel("value")

#	ax.plot(sorted_data[:, 15])
#	ax.plot(sorted_data[:, 16]) 
	
	plt.show()
#	plotting3d (temp, volt, tdis_rate)

def analysis_sweep_ipoint ():
	data_volt04 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_volt04.txt", delimiter='\t')
	data_volt05 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_volt05.txt", delimiter='\t')
	data_volt06 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_volt06.txt", delimiter='\t')
	data_volt07 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_volt07.txt", delimiter='\t')
	data_volt08 = np.loadtxt("result/temp/l180nm/nf_6/sweep_ipoint_volt08.txt", delimiter='\t')
#	df = pd.DataFrame(data_volt06 , columns=['0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00'])
#	print (df)
#	
	data = data_volt06	# set VDD = 0.6V
	
	ipoint = [0.3+ix*0.05 for ix in range(12)]
	temp = [ix*2 for ix in range(50)]
# 	plot discharge time at different ipoints and temp changing
	ax = plt.figure().add_subplot()
	for ix in range(15):
		ax.plot(temp, data[:, ix], label=str(round(ix*0.05 + 0.3, 2)))
	ax.grid()
	plt.title("Analysis discharge time of different initial points in the temperature domain")
	plt.xlabel("Temperature (C)")
	plt.ylabel("Discharge time log scale (s)")
	plt.yscale("log")
	leg = plt.legend(loc='upper right')
	
# 	plot rate of t_dis at different ipoints with full t_dis in temp domain
	bx = plt.figure().add_subplot()
	for ix in range(14):
		rate = [data[ir, 14]/data[ir, ix] for ir in range(len(temp))] 
		bx.plot(temp, rate, label=str(round(ix*0.05 + 0.3, 2)))   
	bx.grid()
	plt.title("Analysis rate of full discharge time vs discharge time of different initial points in temperature domain")
	plt.xlabel("Temperature (C)")
	plt.ylabel("Discharge rate")
	leg = plt.legend(loc='upper right')

# form a variable to analysis
	
# plot rate of t_dis at different ipoints with full t_dis in temp domain
	ip1 = 14
	ip2 = 10		
	ip4 = 11
	ip3 = ip1

#  var = [(data[ir, ip1] - data[ir, ip2])/(data[ir, ip3] - data[ir, ip4]) for ir in range(len(temp))] 
#	var = [(data[ir, ip1] - data[ir, ip2])/(data[ir, ip3] - data[ir, ip4]) for ir in range(len(temp))] 
	var = [(data[ir, ip1])/(data[ir, ip3] - data[ir, ip4]) for ir in range(len(temp))] 
	cx = plt.figure().add_subplot()
	cx.plot(temp, var, label=str(round(ix*0.05 + 0.3, 2)))   
	cx.grid()
	plt.title("finding a variable in the temp domain")
	plt.xlabel("temp (C)")
	plt.ylabel("value")
	leg = plt.legend(loc='upper right')

def form_vars(n):
	print (n)
	y = [np.exp((n[ix, 0] - n[ix, 1])/(n[ix, 1] - n[ix, 2]) + 4) for ix in range(len(n))]
	return y

main()

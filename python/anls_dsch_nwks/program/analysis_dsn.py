import numpy as np
import os
import matplotlib.pyplot as plt

def main():
	model = 'dschr_nwk_2t_type_b'
	vdds = np.arange(0.4, 0.81, 0.05)
	inits = np.arange(0.3, 1.01, 0.1)
#	simulation(model, 0.8, 0.8, 1000, 110)
	multi_sims(model, vdds, inits)	
	show_graph(model, vdds, inits)
	plt.show()


def multi_sims(model, vdds, inits):
	stop = 105		#ms
	step = 200		#ns
	os.system ('rm tmp.txt')
	for vdd in vdds:
		for init in inits:
			simulation(model, vdd, vdd*init, step, stop)
	cmd = 'cp tmp.txt results/' + model + '.txt'
	os.system(cmd)	


def show_graph(model, vdds, inits):
	filename_ = 'results/' + model + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	print (data)
	data3d = data.reshape((len(vdds), len(inits), 3))
	print (data3d)	
	for ix in range(len(vdds)):
		t_dis = data3d [ix,:,2]
		plt.figure(1)
		plt.plot(inits, t_dis)	
		plt.grid()

def simulation (model, vdd, v_init, step, stop):
	cmd = 'ngspice '
	model = ' netlists/' + model + '_tb.spice'
	arg1 = " -D vdd=" + str(vdd)
	arg2 = " -D v_init=" + str(v_init)
	arg3 = " -D step_time=" + str(step) + "n"
	arg4 = " -D stop_time=" + str(stop) + "m"
	cmd = cmd + model + arg1 + arg2 + arg3 + arg4
	print (cmd)
	os.system (cmd)


main()


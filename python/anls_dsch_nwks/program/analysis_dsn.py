import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def main():
	lib = 'tsmc65_hspice'
#	lib = 'sky130a_ngspice'
#	lib = 'sky130a_spectre'	

	model = 'dschr_nwk_type_a'
	vdds = np.arange(0.4, 0.81, 0.05)
	inits = np.arange(0.3, 1.01, 0.1)

#	simulation(model, 0.8, 0.8, 1000, 110)

	if (os.path.exists('results/' + lib + '/' + model + '.txt')):
		print ('load data from result file')
	else:
		print ('result file is not exist, run simulations')
		if (lib == 'tsmc65_hspice'):
			hspice_sim(model)
		elif (lib == 'sky130a_ngspice'):
			multi_sims(model, vdds, inits)			
		elif (lib == 'sky130a_spectre'):
			print ('spectre is running')
	show_graph(lib, model, vdds, inits)
	plt.show()


def spectre_sim():
	print ('spectre')

def hspice_sim (model):
	lib = 'tsmc65_hspice'
	cmd = 'hspice '
	inpfile = 'netlists/' + lib + '/' + model + '_tb.spice'
	outfile = 'results/' + lib + '/' + model + '.txt'
	rawfile = 'results/' + lib + '/raw/' + model + '_tb'
	arg1 = " -i " + inpfile
	arg2 = " -o " + rawfile
	cmd = cmd + arg1 + arg2
	print (cmd)
	os.system (cmd)

	print ('processing RAW file')	
	rawfile = rawfile+'.mt0.csv'
	array = []
	arr = []
	with open(rawfile, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if ((len(row) == 6) & (row[0] != 'index')):
				val = np.array(row)
				val = val.astype(np.float)
				array.append(val)

	array = np.array(array)
	arr = np.vstack ((array[:,1], array[:,2], array[:, 3])).T
	np.savetxt(outfile, arr)	


def get_sim_time(model):
	vdd = 0.4
	v_init = 0.8
	step = 1000
	stop = 100
	
	os.system ('rm tmp.txt')
	simulation(model, vdd, v_init, step, stop)
	data = np.loadtxt('tmp.txt', dtype=float)
	sim_time = data[2];
	print (data)


def multi_sims(model, vdds, inits):
	stop = 105		#ms
	step = 200		#ns
	os.system ('rm tmp.txt')
	for vdd in vdds:
		for init in inits:
			simulation(model, vdd, vdd*init, step, stop)
	cmd = 'cp tmp.txt results/' + model + '.txt'
	os.system(cmd)	


def show_graph(lib, model, vdds, inits):
	filename_ = 'results/' + lib + '/' + model + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	print (data)
	data3d = data.reshape((len(vdds), len(inits), 3))
	print (data3d)	
	for ix in range(len(vdds)):
		t_dis = data3d [ix,:,2]
		t_dis = t_dis * 1e3
		plt.figure(1)
		plt.plot(inits, t_dis)	
		plt.grid()
	plt.title('Discharge time analysis of the ' + model + ' ' + lib)
	plt.xlabel('V_init / VDD')
	plt.ylabel('Discharge time (ms)')

def simulation (model, vdd, v_init, step, stop):
	cmd = 'ngspice '
	model = ' netlists/sky130a_ngspice/' + model + '_tb.spice'
	arg1 = " -D vdd=" + str(vdd)
	arg2 = " -D v_init=" + str(v_init)
	arg3 = " -D step_time=" + str(step) + "n"
	arg4 = " -D stop_time=" + str(stop) + "m"
	cmd = cmd + model + arg1 + arg2 + arg3 + arg4
	print (cmd)
	os.system (cmd)


main()


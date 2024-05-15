import numpy as np
import os
import csv
import matplotlib.pyplot as plt


def main():

#	analysis('volt')
	analysis('temp')



def analysis(sim_type):
	if sim_type == 'volt':
		print('Running analysis of supply voltage variation\n')
	elif sim_type == 'temp':
		print('Running analysis of temperature variation\n')

	lib = 'tsmc65_hspice'
#	lib = 'sky130a_ngspice'
#	lib = 'sky130a_spectre'	

	model = 'dschr_nwk_type_h'
	temps = np.arange(-40, 101, 10)
	vdds = np.arange(0.4, 0.81, 0.05)
	inits = np.arange(0.3, 1.01, 0.1)

	if (os.path.exists('results/' + sim_type + '/' + lib + '/' + model + '.txt')):
		print ('load data from result file')
	else:
		print ('result file is not exist, run simulations')
		if (lib == 'tsmc65_hspice'):
			hspice_sim(model, sim_type)
		elif (lib == 'sky130a_ngspice'):
			multi_sims(model, vdds, inits)			
		elif (lib == 'sky130a_spectre'):
			spectre_sim()

	if sim_type	== 'volt':
		show_volt_graph(lib, model, vdds, inits)
	elif sim_type == 'temp':
		show_temp_graph(lib, model, temps, inits)
	plt.show()

def spectre_sim():
	print ('spectre')

def hspice_sim (model, sim_type):
	lib = 'tsmc65_hspice'
	ctrl_volt_file = 'netlists/tsmc65_hspice/ctrl_volt.spice'
	ctrl_temp_file = 'netlists/tsmc65_hspice/ctrl_temp.spice'
	ctrl_sims_file = 'netlists/tsmc65_hspice/ctrl_sims.spice'

	if sim_type == 'volt':
		os.system ('cp ' + ctrl_volt_file + ' ' + ctrl_sims_file)
	elif sim_type == 'temp':
		os.system ('cp ' + ctrl_temp_file + ' ' + ctrl_sims_file)

	cmd = 'hspice '
	inpfile = 'netlists/' + lib + '/' + model + '_tb.spice'
	outfile = 'results/' + sim_type + '/' + lib + '/' + model + '.txt'
	rawfile = 'results/' + sim_type + '/' + lib + '/raw/' + model + '_tb'
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
	np.savetxt(outfile, arr, fmt = '%3.3e')

def get_sim_time(model):
	vdd = 0.4
	v_init = 0.8
	step = 1000
	stop = 100
	
	os.system ('rm tmp.txt')
	ngspice_sim(model, vdd, v_init, step, stop)
	data = np.loadtxt('tmp.txt', dtype=float)
	sim_time = data[2];
	print (data)

def multi_sims(model, vdds, inits):
	stop = 105		#ms
	step = 200		#ns
	os.system ('rm tmp.txt')
	for vdd in vdds:
		for init in inits:
			ngspice_sim(model, vdd, vdd*init, step, stop)
	cmd = 'cp tmp.txt results/volt/' + model + '.txt'
	os.system(cmd)	

def import_data(sim_type, lib, model, vars_1, vars_2):
	filename_ = 'results/' + sim_type + '/' + lib + '/' + model + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	print (data)
	data3d = data.reshape((len(vars_1), len(vars_2), 3))
	print (data3d)	
	return data3d

def show_temp_graph(lib, model, temps, inits):
	data3d = import_data('temp', lib, model, inits, temps)

	t2_dis = data3d[len(inits)-1, :, 2]
	for ix in range(len(inits)-2):
		t1_dis = data3d[ix, :, 2]
		t_dis = t1_dis * 1e3
		plt.figure(1)
		plt.subplot(2, 1, 1)
		plt.semilogy(temps, t_dis)	
		
		plt.subplot(2, 1, 2)
		plt.plot(temps, t2_dis/t1_dis)
	

	plt.subplot(2, 1, 1)
	plt.title('Discharge time analysis of the ' + model + ' ' + lib)
	plt.xlabel('temperatures')
	plt.ylabel('Discharge time t1 (ms)')

	plt.subplot(2, 1, 2)
	plt.xlabel('temperatures')
	plt.ylabel('Discharge ratio t1/t2')
	plt.tight_layout()
	

def show_volt_graph(lib, model, vdds, inits):
	data3d = import_data('volt', lib, model, vdds, inits)
	for ix in range(len(vdds)):
		t_dis = data3d [ix,:,2]
		t_dis = t_dis * 1e3
		plt.figure(1)
		plt.plot(inits, t_dis)	
		plt.grid()
	plt.title('Discharge time analysis of the ' + model + ' ' + lib)
	plt.xlabel('V_init / VDD')
	plt.ylabel('Discharge time (ms)')

def ngspice_sim(model, vdd, v_init, step, stop):
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


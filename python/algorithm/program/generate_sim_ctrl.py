import numpy as np
import os
import matplotlib.pyplot as plt

def main():
	model = 'dschr_nwk_1t_type_a'
	vdds = np.arange(0.6, 1.21, 0.02)
	temps = np.arange(-40, 101, 10)
	inits = np.arange(0.45, 1.01, 0.05)
#	vdd = 0.6
#	gen_input_temp(temps, inits)
#	gen_input_volt(vdds, inits)
#	show_graph(vdds, inits)
#	gen_input_volt_r_cap()
	gen_input_temp_r_cap()



def gen_input_volt_r_cap():
	vdds = np.arange(0.4, 0.81, 0.02)
	r_caps = np.arange(1.25, 2.501, 0.025)
	os.system ('rm ./netlists/tsmc65_hspice/tmp.txt')
	input_arr = []
	cnt = 0
	for vdd in vdds:
		for r_cap in r_caps:
			vec = [vdd, r_cap]
			input_arr.append(vec)
			cnt = cnt+1
	print ('number of sims: ', cnt)
	np.savetxt('./netlists/tsmc65_hspice/tmp.txt', input_arr, fmt='%2.3f')


def gen_input_temp_r_cap():
	os.system ('rm ./netlists/tsmc65_hspice/tmp.txt')
	vdds = np.arange(0.4, 0.81, 0.1)
	temps = np.arange(0, 101, 2)
	r_caps = np.arange(1.25, 2.501, 0.025)

#	vdds = np.arange(0.4, 0.81, 0.2)
#	temps = np.arange(0, 101, 10)
#	r_caps = np.arange(1.5, 2.501, 1)

	cnt = 0
	input_arr = []
	for vdd in vdds:
		for temp in temps:
			for r_cap in r_caps:
				vec = [temp, vdd, r_cap]
				input_arr.append(vec)
				cnt = cnt+1
	print ('number of sims: ', cnt)
	gen_in_data(input_arr)

def gen_input_temp(temps, inits):
	os.system ('rm ./netlists/tsmc65_hspice/tmp.txt')
	input_arr = []
	for init in inits:
		for temp in temps:
			vec = [temp, init]
			input_arr.append(vec)
	np.savetxt('./netlists/tsmc65_hspice/tmp.txt', input_arr, fmt='%2.1f')


def gen_input_volt(vdds, inits):
	os.system ('rm ./netlists/tsmc65_hspice/tmp.txt')
	input_arr = []
	for vdd in vdds:
		for init in inits:
			vec = [vdd, init*vdd]
			input_arr.append(vec)
	np.savetxt('./netlists/tsmc65_hspice/tmp.txt', input_arr, fmt='%.3f')


def show_graph(vdds, inits):
	filename_ = 'tmp.txt'
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
	plt.title('Discharge time analysis of the ')
	plt.xlabel('V_init / VDD')
	plt.ylabel('Discharge time (ms)')

def gen_in_data(data, lib='tsmc65_hspice'):
	template_file = 'netlists/' + lib + '/template.spice'
	ctrl_sims_file = 'netlists/' + lib + '/ctrl_sims.spice'
	cp_cmd = 'cp ' + template_file + ' ' + ctrl_sims_file
	os.system(cp_cmd)
	val_names = '+ temp_val vdd r_cap'
	with open(ctrl_sims_file, 'a') as f:
		f.write(val_names)
		f.write('\n')
	
	def list2str(list):
		string = ''
		for ele in list:
			string = string + ' ' + str(round(ele, 3))
		string = '+' + string
		return string
	
	with open(ctrl_sims_file, 'a') as f:
		for element in data: 
			f.write(list2str(element)) 
			f.write('\n') 
		f.write('.ENDDATA')
#
#
#def hspice_pal_sim(lib, dsnwk):
#	sim_type = 'temps'
#	vdds = np.arrange (0.4, 0.81, 0.1)
#	temps = np.arange (0, 101, 2)
#	r_caps = np.arange(1.25, 2.501, 0.025)
#
#	cp_cmd = 'cp ' + template_file + ' ' + ctrl_sims_file
#	
#	processes = []
#	for vdd in vdds:
#		data = []
#		for temp in temps:
#			for r_cap in r_caps:
#				line = [temp, vdd, r_cap]
#				data.append(line)
#
#		gen_in_data(data, template_file, ctrl_sims_file)
#		p = Process(target=hspice_sim, args=('temp', 'tsmc65_hspice', 'dsn_h_12t', str(vdd), ))
#		p.start()
#		processes.append(p)
#		time.sleep (10)
#	
#	for p in processes:
#		p.join()
#		time.sleep(10)
#

main()


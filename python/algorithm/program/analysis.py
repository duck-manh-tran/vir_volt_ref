import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import math
import time
import pandas as pd

## option flags to run the analysis

## Overwrite the simulation files
#ovr_flag = False
ovr_flag = True

def main():

## Choose the type of the analysis
	sim_type = 'volt'
#	sim_type = 'temp'

## Choose the library-simulator of the analysis
	lib = 'tsmc65_hspice'
#	lib = 'sky130a_ngspice'
#	lib = 'sky130a_spectre'	

## Choose the discharge network for the analysis
	dsnwk = 'dsn_h_12t'

	analysis(sim_type, lib, dsnwk)
#	volt_anls_1(lib, dsnwk)
#	CTAT_var_anls1(lib, dsnwk)
#	temp_anls_1(lib, dsnwk)
#	CTAT_var_anls2(lib, dsnwk)


def analysis(sim_type, lib, dsnwk):

	if (~ovr_flag & os.path.exists('results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt')):
		print ('load data from result file')
	else:
		print ('result file is not exist, run simulations')
		if (lib == 'tsmc65_hspice'):
			hspice_sim(sim_type, lib, dsnwk)
		elif (lib == 'sky130a_ngspice'):
			ngspice_sim()
		elif (lib == 'sky130a_spectre'):
			spectre_sim()

	if sim_type	== 'volt':
		print('Running analysis of supply voltage variation\n')
		volt_anls_1(lib, dsnwk)
	elif sim_type == 'temp':
		print('Running analysis of temperature variation\n')
		temp_anls_1(lib, dsnwk)
#		CTAT_var_anls(lib, dsnwk)
#		CTAT_var_anls2(lib, dsnwk)
	plt.show()

def spectre_sim():
	print ('spectre')

def ngspice_sim():
	print ('ngspice')


def hspice_sim (sim_type='volt', lib='tsmc65_hspice', dsnwk='dsn_h_12t', subfix=''):
#	lib = 'tsmc65_hspice'
	cmd = 'hspice -dp 10'
	inpfile = 'netlists/' + lib + '/' + dsnwk + '_tb.spice'
	outfile = 'results/' + sim_type + '/' + lib + '/' + dsnwk + subfix +'.txt'
	rawfile = 'results/' + sim_type + '/' + lib + '/raw/' + dsnwk + subfix + '_tb'
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
		tmp = csv.reader(csvfile)
		num_of_val = len(list(tmp)[10])
		print ('number of values: ', num_of_val)

	with open(rawfile, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if ((len(row) == num_of_val) & (row[0] != 'index')):
				val = np.array(row)
				val = val.astype(np.float)
				array.append(val)

	array = np.array(array)
	print ('length of data: ', len(array))

	arr = array[:, 1]
	for ix in range (num_of_val-4):
		arr = np.vstack((arr, array[:, ix+2]))
	
	arr = arr.T
	np.savetxt(outfile, arr, fmt = '%3.6e')


def process_dataframe(sim_type, lib, dsnwk):
	filename_ = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	if (sim_type == 'temp'):
		df_cols = ['temp', 'vdd', 'r_cap', 'n1', 'n2', 'n3']
	elif (sim_type == 'volt'):
		df_cols = ['vdd', 'r_cap', 'n1', 'n2', 'n3']

	df =  pd.DataFrame(data, columns = df_cols)
	df['1_vdd'] = 1/df['vdd']
	df['n1'] = df['n1'] * 1e6
	df['n2'] = df['n2'] * 1e6
	df['n3'] = df['n3'] * 1e6

	df['d_12'] = df['n1'] - df['n2']
	df['d_23'] = df['n2'] - df['n3']
	df['d_13'] = df['n1'] - df['n3']
	df['d_123'] = df['n1'] + df['n3'] - 2*df['n2']

	df['div_1'] = df.d_12/df.d_23
	df['div_2'] = df.d_13/df.d_23
	df['div_3'] = df.d_13/df.d_12
	df['div_4'] = (df.d_13-500)/df.d_12

	df['y_1'] = df.div_1 + df.div_2 - 2
	df['y_2'] = df.div_1 + df.div_2 + df.div_3 - 3
	df['y_3'] = df.div_1 + df.div_2 - df.div_3
	df['y_4'] = df.div_2 - df.div_3
	
	df['ey_1'] = np.exp(df.y_1)
	df['ey_2'] = np.exp(df.y_2)
	df['ey_3'] = np.exp(df.y_3)
	df['ey_4'] = np.exp(df.y_4)

#	df['ediv_1'] = np.exp(df.div_1)
#	df['ediv_2'] = np.exp(df.div_2)
#	df['ediv_3'] = np.exp(df.div_3)
	
	df['CTAT_1'] = np.cbrt(1.5 - (df.d_13-500)/df.d_12)

	r_cap = df['r_cap'].to_numpy()
	ey_val =	df['ey_3'].to_numpy()

	C = np.loadtxt('C_coeff.txt', dtype=float)
	order = 1
	len_order = 3
	while len(C) > len_order:
		order = order+1
		len_order = 0.5*order**2 + 1.5*order + 1

	e = []	
	for n in range (order+1):
		for y in range (n+1):
			for x in range (n+1):
				if x+y==n:
					e.append([x, y])
	eX = np.array([x for x,_ in e]).T
	eY = np.array([y for _,y in e]).T
	r_cal = []
	for ix in range(len(ey_val)):
		A = (r_cap[ix] ** eX) * (ey_val[ix] ** eY)
		r_cal.append(np.dot(A, C))
	df['r_cal'] = r_cal
	return df

def show_graph (grph_name, data_df, data_y, data_z):

	fig = plt.figure(grph_name)
	list_obj = []
	n_rows = 2
	n_cols = 3

	for obj in range(n_rows * n_cols):	
		list_obj.append(fig.add_subplot(n_rows, n_cols, obj+1, projection='3d'))
	
	def setup_axes(obj, data_x, data_y, data_z, dat_x_label, axis_x_label):
		colors = ['blue', 'orange', 'green', 'red', 'purple']	
		list_obj[obj].set_xlabel(axis_x_label, fontsize = 16)
		num_of_var = len(data_x)
		for ix in range(num_of_var):
			list_obj[obj].scatter(data_df[data_x[ix]], data_df[data_y], data_df[data_z], c=colors[ix], label=dat_x_label[ix])

	names = ['$n_1$', '$n_2$', '$n_3$']
	cols = ['n1', 'n2', 'n3']
	setup_axes (obj=0, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='n')

	names = ['$d_{12}=n_1-n_2$', '$d_{23}=n_2-n_3$', '$d_{13}=n_1-n_3$', '$d_{123}$']
#	cols = ['d_12', 'd_23', 'd_13', 'd_123']
	cols = ['d_12', 'd_23', 'd_13']
	setup_axes (obj=1, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='diff')

	names = ['$dv_1=d_{12}/d_{23}$', '$dv_2=d_{13}/d_{23}$', '$dv_3=d_{13}/d_{12}$', '$dv_4$']
#	cols = ['div_1', 'div_2', 'div_3', 'div_4']
	cols = ['div_1', 'div_2', 'div_3']
	setup_axes (obj=2, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='div')

	names = ['$y_1=dv_1+dv_2-2$', '$y_2=dv_1+dv_2+dv_3-4$', '$y_3=dv_1+dv_2-dv3$', '$y_4$']
#	cols = ['y_1', 'y_2', 'y_3', 'y_4']
	cols = ['y_1', 'y_2', 'y_3']
	setup_axes (obj=3, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='$y_{value}$')

	names = ['$e^{y_1}$', '$e^{y_2}$', '$e^{y_3}$', '$e^{y_4}$']
#	cols = ['ey_1', 'ey_2', 'ey_3', 'ey_4']
	cols = ['ey_1', 'ey_2', 'ey_3']
	setup_axes (obj=4, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='$e^y_{value}$')

#	names = ['$e^{div_1}$', '$e^{div_2}$', '$e^{div_3}$']
#	cols = ['ediv_1', 'ediv_2', 'ediv_3']
#	setup_axes (obj=5, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='$e^{div}$')

#	names = ['$CTAT_1$', '$e^{div_2}$', '$e^{div_3}$']
	names = ['the $ey$ value']
	cols = ['ey_3']
	setup_axes (obj=5, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='$ey$')
#	setup_axes (obj=5, data_x=cols, data_y=data_y, data_z=data_z, dat_x_label=names, axis_x_label='$e^{div}$')	
	return fig, list_obj


def temp_anls_1(lib, dsnwk):
	processed_df = process_dataframe('temp', lib, dsnwk)

	fig = plt.figure('temp')
	n_rows = 2
	n_cols = 3
	colors = ['navy', 'blue', 'dodgerblue']
	list_obj = []
	def r_temp_fit(func, vdd):
		temp_range = np.arange (0, 101, 1)
		r_temp_fit = func[0]*temp_range + func[1]
		r_temp_inv = -func[0]*temp_range - func[1] + 1/vdd
		return temp_range, r_temp_fit, r_temp_inv

	vdds = np.arange(4, 8.1, 1)/10
	for idd in range(len(vdds)):
		filt_df = processed_df[processed_df['vdd'] == vdds[idd]]
		func = np.polyfit(filt_df['temp'], filt_df['r_cal'], 1)
		list_obj.append(fig.add_subplot(n_rows, n_cols, idd+1))
		list_obj[idd].scatter(filt_df['temp'], filt_df['r_cal'])
		list_obj[idd].plot([0, 100], [1/vdds[idd], 1/vdds[idd]])
		X, Y, Y_inv = r_temp_fit(func, vdds[idd])
		list_obj[idd].plot(X, Y)
		list_obj[idd].plot(X, Y_inv)
		list_obj[idd].set_title('VDD = ' + str(vdds[idd]) + ' @ ' + str(int(func[0]*1e4)) + '$e^{-4}/^oC$')
		list_obj[idd].grid()	

	colors = ['blue', 'orange', 'green', 'red', 'purple']
	list_obj.append(fig.add_subplot(n_rows, n_cols, 6))
	for idd in range(len(vdds)):	
		filt_df = processed_df[processed_df['vdd'] == vdds[idd]]
		list_obj[5].scatter(filt_df['temp'], filt_df['r_cal'], c = colors[idd], label='VDD='+str(vdds[idd]))
		list_obj[5].plot([0, 100], [1/vdds[idd], 1/vdds[idd]], c = colors[idd])
		func = np.polyfit(filt_df['temp'], filt_df['r_cal'], 1)
		X, Y, Y_inv = r_temp_fit(func, vdds[idd])
		list_obj[5].plot(X, Y)

	list_obj[5].set_xlabel('Temperature', fontsize=16)
	list_obj[5].set_ylabel('r=f($r_{cap}$, ey)', fontsize=16)
	
	list_obj[5].legend()	
	list_obj[5].grid()	
	
#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)

def volt_anls_1 (lib, dsnwk):
	data_df = process_dataframe('volt', lib, dsnwk)

	fig, list_obj = show_graph('volt_analysis', data_df, 'r_cap', '1_vdd')
	for ix in range(6):
		list_obj[ix].set_ylabel('$r_{cap}$', fontsize = 16)
		list_obj[ix].set_zlabel('$r=1/VDD$', fontsize = 16)
		list_obj[ix].legend()
		list_obj[ix].grid(True)	

#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()	
	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)


def CTAT_var_anls1(lib, dsnwk):
	processed_df = process_dataframe('temp', lib, dsnwk)

	filt_df = processed_df[processed_df['vdd'] == 0.6]

	def r_temp_compen(df):
		func = np.polyfit(df['temp'], df['r_cal'], 1)
		df['comp'] = -func[0]*df['temp'] - func[1] + 1/df['vdd']
		return df
	filt_df = r_temp_compen(filt_df)

	fig, list_obj = show_graph('CTAT_var_analysis_1', filt_df, 'r_cap', 'temp')

	for ix in range(6):
		list_obj[ix].set_ylabel('$r_{cap}$', fontsize = 16)
		list_obj[ix].set_zlabel('$temp$', fontsize = 16)
		list_obj[ix].legend()	
		list_obj[ix].grid(True)	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)


def CTAT_var_anls2(lib, dsnwk):
	processed_df = process_dataframe('temp', lib, dsnwk)

	def cal_fit_line(func):
		temp_range = np.arange (0, 101, 1)
		fit_val = func[0]*temp_range + func[1]
		return temp_range, fit_val

	list_obj = []
	fig = plt.figure('temp_var_anls2')
	colors = ['blue', 'orange', 'green', 'red', 'purple']	
	n_rows=1
	n_cols=2

	list_obj.append(fig.add_subplot(n_rows, n_cols, 1))
	list_obj.append(fig.add_subplot(n_rows, n_cols, 2))
#	ctat_1 = np.sqrt(-filt_df['CTAT_1'] + 2) 
#	ctat_2 = np.cbrt(-filt_df['CTAT_1'] + 1.5) 
	
	
	vdds = np.arange(4, 8.1, 1)/10
	for idd in range(len(vdds)):
		filt_df = processed_df[processed_df['vdd'] == vdds[idd]]
		ctat = 0.37 - 4*filt_df['CTAT_1']/(filt_df['r_cap']+9.5)
		func1 = np.polyfit(filt_df['temp'], ctat, 1)
		temp_range, line1 = cal_fit_line(func1)
		list_obj[0].scatter(filt_df['temp'], ctat, c=colors[idd], label=('VDD='+str(vdds[idd])))
		list_obj[0].plot(temp_range, line1, c=colors[idd], label=('slope='+str(int(func1[0] * 1e4))+'E-4'))
		ctat = 0.37 - 4*filt_df['CTAT_1']/(filt_df['r_cap']+9.5)
		list_obj[1].scatter(filt_df['temp'], filt_df['r_cal'] + ctat, c=colors[idd], label=('VDD='+str(vdds[idd])))
		func2 = np.polyfit(filt_df['temp'], filt_df['r_cal'] + ctat, 1)
		temp_range, line2 = cal_fit_line(func2)
		list_obj[1].plot(temp_range, line2, c=colors[idd], label=('slope='+str(int(func2[0] * 1e6))+'E-4'))
		list_obj[1].plot([0, 100], [1/vdds[idd], 1/vdds[idd]], c=colors[idd])

	volt_df = process_dataframe('volt', lib, dsnwk)
	
	v_variation = 0.37 - 8*volt_df['CTAT_1']/(volt_df['r_cap']+21)

#	list_obj[1].scatter(volt_df['vdd'], v_variation, c=colors[1], label='$CTAT_4$')

	#filt_df = processed_df[processed_df['vdd'] == 0.5]

	list_obj[0].set_xlabel('Temperature', fontsize=16)
	list_obj[0].set_ylabel('$y_{ctat}$', fontsize=16)
	list_obj[1].set_xlabel('Temperature', fontsize=16)
	list_obj[1].set_ylabel('$r=f(r_{cap}, ey)+y_{ctat}$', fontsize=16)
	list_obj[0].tick_params(axis='both', which='major', labelsize=14)
	
	for ix in range(2):
		list_obj[ix].legend()	
		list_obj[ix].grid(True)	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)

main()


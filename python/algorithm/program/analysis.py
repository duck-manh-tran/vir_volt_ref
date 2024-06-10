import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import math
import time
import pandas as pd

## option flags to run the analysis

## Overwrite the simulation files
ovr_flag = False
#ovr_flag = True

def main():

## Choose the type of the analysis
#	sim_type = 'volt'
	sim_type = 'temp'

## Choose the library-simulator of the analysis
	lib = 'tsmc65_hspice'
#	lib = 'sky130a_ngspice'
#	lib = 'sky130a_spectre'	

## Choose the discharge network for the analysis
	dsnwk = 'dsn_h_12t'

	analysis(sim_type, lib, dsnwk)


def analysis(sim_type, lib, dsnwk):
	if sim_type == 'volt':
		print('Running analysis of supply voltage variation\n')
	elif sim_type == 'temp':
		print('Running analysis of temperature variation\n')
	vdds = np.arange(0.6, 1.21, 0.02)
	temps = np.arange(-40, 101, 10)
	inits = np.arange(0.45, 1.01, 0.05)

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
		volt_anls_1(lib, dsnwk)
	elif sim_type == 'temp':
		temp_anls_1(lib, dsnwk)
		CTAT_var_anls(lib, dsnwk)
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
	#os.system (cmd)

	print ('processing RAW file')	
	rawfile = rawfile+'.mt0.csv'
	array = []
	arr = []
	with open(rawfile, 'r') as csvfile:
		tmp = csv.reader(csvfile)
		num_of_val = len(list(tmp)[10])
		print ('bnumber of values: ', num_of_val)
	
	with open(rawfile, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if ((len(row) == num_of_val) & (row[0] != 'index')):
				val = np.array(row)
				val = val.astype(np.float)
				array.append(val)

	array = np.array(array)
	print ('length of data: ', len(array))

	arr = []
	for ix in range (num_of_val-3):
		arr = np.vstack((arr, array[:, ix+1]))
	
	arr = arr.T
	np.savetxt(outfile, arr, fmt = '%3.6e')

def temp_anls_1(lib, dsnwk):
	sim_type = 'temp'
	filename_ = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	df_cols = ['temp', 'vdd', 'r_cap', 'n1', 'n2', 'n3']
	raw_df =  pd.DataFrame(data, columns = df_cols)

	def virtual_ref (df):
		df['d_12'] = df['n1'] - df['n2']
		df['d_23'] = df['n2'] - df['n3']
		df['d_13'] = df['n1'] - df['n3']
		df['div_1'] = df.d_12/df.d_23
		df['div_2'] = df.d_13/df.d_23
		df['div_3'] = df.d_13/df.d_12
		df['y_val'] = df.div_1 + df.div_2 - df.div_3
		df['ey_val'] = np.exp(df.y_val)

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
		for ind in df.index:
			A = (df.r_cap[ind] ** eX) * (df.ey_val[ind] ** eY)
			r_cal.append(np.dot(A, C))
		df['r_cal'] = r_cal
		return df

	processed_df = virtual_ref(raw_df)

	fig = plt.figure('temp')
	n_rows = 2
	n_cols = 3
	colors = ['navy', 'blue', 'dodgerblue']
	list_obj = []
	def r_temp_fit(func):
		temp_range = np.arange (0, 101, 1)
		r_temp_fit = func[0]*temp_range + func[1]
		return temp_range, r_temp_fit

	vdds = np.arange(4, 8.1, 1)/10
	for idd in range(len(vdds)):
		filt_df = processed_df[processed_df['vdd'] == vdds[idd]]
		func = np.polyfit(filt_df['temp'], filt_df['r_cal'], 1)
		list_obj.append(fig.add_subplot(n_rows, n_cols, idd+1))
		list_obj[idd].scatter(filt_df['temp'], filt_df['r_cal'])
		list_obj[idd].plot([0, 100], [1/vdds[idd], 1/vdds[idd]])
		X, Y = r_temp_fit(func)
		list_obj[idd].plot(X, Y)
		list_obj[idd].set_title('VDD = ' + str(vdds[idd]) + ' @ ' + str(int(func[0]*1e4)) + '$e^{-4}/^oC$')
		list_obj[idd].grid()	

	colors = ['blue', 'orange', 'green', 'red', 'purple']
	list_obj.append(fig.add_subplot(n_rows, n_cols, 6))
	for idd in range(len(vdds)):	
		filt_df = processed_df[processed_df['vdd'] == vdds[idd]]
		list_obj[5].scatter(filt_df['temp'], filt_df['r_cal'], c = colors[idd], label=str(vdds[idd]))
		list_obj[5].plot([0, 100], [1/vdds[idd], 1/vdds[idd]], c = colors[idd])
		list_obj[5].legend()	
		list_obj[5].grid()	
	
#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)

def variable_gen(df):
	df['d_12'] = df['n1'] - df['n2']
	df['d_23'] = df['n2'] - df['n3']
	df['d_13'] = df['n1'] - df['n3']

	df['div_1'] = df.d_12/df.d_23
	df['div_2'] = df.d_13/df.d_23
	df['div_3'] = df.d_13/df.d_12

	df['y_1'] = df.div_1 + df.div_2 - 2
	df['y_2'] = df.div_1 + df.div_2 + df.div_3 - 4
	df['y_3'] = df.div_1 + df.div_2 - df.div_3
	
	df['ey_1'] = np.exp(df.y_1)
	df['ey_2'] = np.exp(df.y_2)
	df['ey_3'] = np.exp(df.y_3)
	return df

def CTAT_var_anls(lib, dsnwk):
	sim_type = 'temp'
	filename_ = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	
	df_cols = ['temp', 'vdd', 'r_cap', 'n1', 'n2', 'n3']
	raw_df =  pd.DataFrame(data, columns = df_cols)
	
	processed_df = variable_gen(raw_df)
	filt_df = processed_df[processed_df['vdd'] == 0.4]
	
	r_caps = [1.3, 1.6, 1.9, 2.1, 2.4]
	list_obj = []
	fig = plt.figure('temp_var_anls')
	n_rows = 2
	n_cols = 3
	colors = ['blue', 'orange', 'green', 'red', 'purple']	

	list_obj.append(fig.add_subplot(n_rows, n_cols, 1, projection='3d'))
	names = ['$n_1$', '$n_2$', '$n_3$']
	cols = ['n1', 'n2', 'n3']
	for ix in range(3):
		list_obj[0].scatter(filt_df['temp'], filt_df['r_cap'], filt_df[cols[ix]], \
				c=colors[ix], label=names[ix])

	list_obj.append(fig.add_subplot(n_rows, n_cols, 2, projection='3d'))
	names = ['$d_{12}=n_1-n_2$', '$d_{23}=n_2-n_3$', '$d_{13}=n_1-n_3$']
	cols = ['d_12', 'd_23', 'd_13']
	for ix in range(3):
		list_obj[1].scatter(filt_df['temp'], filt_df['r_cap'], filt_df[cols[ix]], \
				c=colors[ix], label=names[ix])

	filt_rcap_mk  = filt_df['r_cap'].isin(r_caps)
	filt_rcap  = filt_df[filt_rcap_mk]

	list_obj.append(fig.add_subplot(n_rows, n_cols, 3))
	names = ['$dv_1=d_{12}/d_{23}$', '$dv_2=d_{13}/d_{23}$', '$dv_3=d_{13}/d_{12}$']
	cols = ['div_1', 'div_2', 'div_3']
	for ix in range(3):
		list_obj[2].scatter(filt_rcap['temp'], filt_rcap[cols[ix]], c=colors[ix], label=names[ix])

	list_obj.append(fig.add_subplot(n_rows, n_cols, 4))
	names = ['$y_1=dv_1+dv_2-2$', '$y_2=dv_1+dv_2+dv_3-4$', '$y_3=dv_1+dv_2-dv3$']
	cols = ['y_1', 'y_2', 'y_3']
	for ix in range(3):
		list_obj[3].scatter(filt_rcap['temp'], filt_rcap[cols[ix]], c=colors[ix], label=names[ix])

	list_obj.append(fig.add_subplot(n_rows, n_cols, 5))
	list_obj[4].scatter(filt_rcap['temp'], filt_rcap['div_3'], c=colors[0], label='$dv_3=d_{13}/d_{12}$')

	for ix in range(5):
		list_obj[ix].legend()	
		list_obj[ix].grid(True)	
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)


def volt_anls_1 (lib, dsnwk):
	sim_type = 'volt'
	filename_ = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'
	data = np.loadtxt(filename_, dtype=float)
	df_cols = ['vdd', 'r_cap', 'n1', 'n2', 'n3']
	raw_df =  pd.DataFrame(data, columns = df_cols)
	
	var_df = variable_gen(raw_df)

	fig2 = plt.figure('volt2')
	list_obj = []
	n_rows = 2
	n_cols = 3
	colors = ['navy', 'blue', 'dodgerblue']
	
	list_obj.append(fig2.add_subplot(n_rows, n_cols, 1, projection='3d'))
	names = ['$n_1$', '$n_2$', '$n_3$']
	cols = ['n1', 'n2', 'n3']
	for ix in range(3):
		list_obj[0].scatter(var_df[cols[ix]], var_df['r_cap'], 1/var_df['vdd'], c=colors[ix], label=names[ix])

	list_obj.append(fig2.add_subplot(n_rows, n_cols, 2, projection='3d'))
	names = ['$d_{12}=n_1-n_2$', '$d_{23}=n_2-n_3$', '$d_{13}=n_1-n_3$']
	cols = ['d_12', 'd_23', 'd_13']
	for ix in range(3):
		list_obj[1].scatter(var_df[cols[ix]], var_df['r_cap'], 1/var_df['vdd'], c=colors[ix], label=names[ix])

	list_obj.append(fig2.add_subplot(n_rows, n_cols, 3, projection='3d'))
	names = ['$dv_1=d_{12}/d_{23}$', '$dv_2=d_{13}/d_{23}$', '$dv_3=d_{13}/d_{12}$']
	cols = ['div_1', 'div_2', 'div_3']
	for ix in range(3):
		list_obj[2].scatter(var_df[cols[ix]], var_df['r_cap'], 1/var_df['vdd'], c=colors[ix], label=names[ix])

	list_obj.append(fig2.add_subplot(n_rows, n_cols, 4, projection='3d'))
	names = ['$y_1=dv_1+dv_2-2$', '$y_2=dv_1+dv_2+dv_3-4$', '$y_3=dv_1+dv_2-dv3$']
	cols = ['y_1', 'y_2', 'y_3']
	for ix in range(3):
		list_obj[3].scatter(var_df[cols[ix]], var_df['r_cap'], 1/var_df['vdd'], c=colors[ix], label=names[ix])

	list_obj.append(fig2.add_subplot(n_rows, n_cols, 5, projection='3d'))
	names = ['$e^{y_1}$', '$e^{y_2}$', '$e^{y_3}$']
	cols = ['ey_1', 'ey_2', 'ey_3']
	for ix in range(3):
		list_obj[4].scatter(var_df[cols[ix]], var_df['r_cap'], 1/var_df['vdd'], c=colors[ix], label=names[ix])

	for ix in range(5):
		list_obj[ix].legend()	
		list_obj[ix].grid(True)	

#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()	
	
	fig2.tight_layout(pad=0, w_pad=0, h_pad=0)

main()


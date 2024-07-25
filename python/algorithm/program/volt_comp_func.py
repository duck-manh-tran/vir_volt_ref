import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from functions import *
import numpy as np
import math
import sys 
import csv
import pandas as pd

def main():
	get_volt_func()
	plt.show()


def get_va_model_data():
	rawfile = 'data.csv'
	array = []
	with open(rawfile, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			val = np.array(row)
			val = val.astype(np.float)
			array.append(val)

	array = np.array(array)
	df = pd.DataFrame(array, columns = ['vdd', 'r_cap', 'n1', 'n2', 'n3'])
	df['y_val'] = (df['n1'] - df['n2'])/(df['n2'] - df['n3']) \
					+ (df['n1'] - df['n3'])/(df['n2'] - df['n3']) \
					- (df['n1'] - df['n3'])/(df['n1'] - df['n2']) 
	df['ey_val'] = np.exp(df['y_val']) 
	df_sorted = df.sort_values(by=['vdd', 'r_cap'])
	df_sorted.to_csv('va_model.csv', index='false') 
	return df_sorted

def get_ananlysis_data(sim_type, lib, dsnwk):
	file_name = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'
	data = np.loadtxt(file_name, dtype=float)
	
	if sim_type == 'volt':
		df = pd.DataFrame(data, columns = ['vdd', 'r_cap', 'n1', 'n2', 'n3'])
		df['n1'] = df['n1']*1e8
		df['n2'] = df['n2']*1e8
		df['n3'] = df['n3']*1e8
		df['n1'] = df['n1'].round(0)
		df['n2'] = df['n2'].round(0)
		df['n3'] = df['n3'].round(0)
		df['y_val'] = (df['n1'] - df['n2'])/(df['n2'] - df['n3']) \
						+ (df['n1'] - df['n3'])/(df['n2'] - df['n3']) \
						- (df['n1'] - df['n3'])/(df['n1'] - df['n2']) 
		df['ey_val'] = np.exp(df['y_val']) 
		df_sorted = df.sort_values(by=['vdd', 'r_cap'])
		print (df)
		df_sorted.to_csv('analysis.csv', index='false') 
		return df_sorted

	elif sim_type == 'temp':
		print ('TBD')
		return 1, 1, 1

	else:
		print ('please enter right sim_type')
		return 0, 0, 0

def get_volt_func():
#	df = get_ananlysis_data('volt', 'tsmc65_hspice', 'dsn_h_12t')
	df = get_va_model_data()
	order = 5

	var_x = np.array(df['r_cap'])
	var_y = np.array(df['ey_val'])
	var_z = np.array(10/df['vdd'])
	
	C_coeff = curve_fitting(order,  var_x, var_y, var_z)
	C_coeff = np.around(C_coeff, 12)
	np.savetxt('C_coeff.txt', C_coeff, delimiter=', ', fmt='%.12f')
#	plot_fitting_mesh(order, C_coeff, var_x, var_y, var_z)
	df = get_va_model_data()
	print (df)
	print (df.sort_values(by='ey_val'))
	var_x1 = np.array(df['r_cap'])
	var_y1 = np.array(df['ey_val'])
	var_z1 = np.array(10/df['vdd'])

	plot_fitting_mesh(order, C_coeff, var_x1, var_y1, var_z1)

def curve_fitting(order, X, Y, Z):
	# 1=linear, 2=quadratic, 3=cubic, ..., nth degree
	X = X.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	Z = Z.reshape(-1, 1)
	
	# calculate exponents of design matrix
	e = []
	for n in range (order+1):
		for y in range (n+1):
			for x in range (n+1):
				if x+y==n:
					e.append((x, y))
	
	eX = np.array([xi for xi,_ in e]).T
	eY = np.array([yi for _,yi in e]).T
	print ("ey:", eY)
	# best-fit polynomial surface
	A = (X ** eX) * (Y ** eY)
	C,resid,_,_ = lstsq(A, Z)    # coefficients
	
	# print summary
	print(f'data = {Z.size}x3')
	print(f'model = {exp2model(e)}')
	print(f'coefficients =\n{C}')
	print('length of C: ', len(C))
	return C

def plot_fitting_mesh (order, C, X, Y, Z):
	C_ = C
	X = X.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	Z = Z.reshape(-1, 1)

	e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
	eX = np.asarray([[x] for x,_ in e]).T
	eY = np.asarray([[y] for _,y in e]).T

	# uniform grid covering the domain of the data
	XX,YY = np.meshgrid(np.linspace(X.min(), X.max(), 20), np.linspace(Y.min(), Y.max(), 20))
	
	# evaluate model on grid
	A = (XX.reshape(-1,1) ** eX) * (YY.reshape(-1,1) ** eY)
	ZZ = np.dot(A, C).reshape(XX.shape)
	B = (X ** eX) * (Y ** eY)
	Z_cap = np.dot(B, C_).reshape(X.shape)

	LR_e = [(100 * abs(Z_cap[iz] - Z[iz])/Z[iz]) for iz in range(len(Z_cap))]
	LR_e = np.concatenate( LR_e, axis=0 )


	fig = plt.figure('Forming voltage compensated function')	
	ax1 = fig.add_subplot(1, 2, 2)
	ax1.plot(LR_e)
	ax1.set_xlabel('points', fontsize=16)
	ax1.set_ylabel('fitting error(%)', fontsize=16)
	ax1.grid(True)
	
	ax2 = fig.add_subplot(1, 2, 1, projection ="3d")	
	ax2.scatter(X, Y, Z, c='r', s=2)
	ax2.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
	ax2.set_xlabel('$r_{cap}$', fontsize=16)
	ax2.set_ylabel('$ey$', fontsize=16)
	ax2.set_zlabel('$r=1/VDD$', fontsize=16)

#	plt.xticks(fontsize=16, rotation=90)
#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)

main()

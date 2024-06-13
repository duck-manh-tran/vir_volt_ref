import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from functions import *
import numpy as np
import math
import sys 


def main():
	get_volt_func()

	plt.show()


def get_data(sim_type, lib, dsnwk):
	file_name = 'results/' + sim_type + '/' + lib + '/' + dsnwk + '.txt'	
	data = np.loadtxt(file_name, dtype=float)
	
	if sim_type == 'volt':
		vdds = data[:, 0]
		r_caps = data[:, 1]
		n_vals = data[:, 2:5]*1e6
		n_vals = n_vals.T
		return vdds, r_caps, n_vals

	elif sim_type == 'temp':
		print ('TBD')
		return 1, 1, 1

	else:
		print ('please enter right sim_type')
		return 0, 0, 0

def get_volt_func():
	vdds, r_caps, n_vals = get_data('volt', 'tsmc65_hspice', 'dsn_h_12t')
	order = 5
	r_reals = 1/vdds	

	d_12 = n_vals[0] - n_vals[1]
	d_23 = n_vals[1] - n_vals[2]
	d_13 = n_vals[0] - n_vals[2]

	div_1 = d_12/d_23
	div_2 = d_13/d_23
	div_3 = d_13/d_12

#	ey_vals = np.exp(div_1 + div_2 - div_3)
	ey_vals = np.exp(div_2 - div_3 + 2)
#	ey_vals = 4*(div_1 + div_2 - div_3)

	print (len(ey_vals))
	
	C_coeff = curve_fitting(order,  r_caps, ey_vals, r_reals)
	np.savetxt('C_coeff.txt', C_coeff, delimiter='\t')
	plot_fitting_mesh(order, C_coeff, r_caps, ey_vals, r_reals)

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
	C_ = np.around(C, 15)
	B = (X ** eX) * (Y ** eY)
	Z_cap = np.dot(B, C_).reshape(X.shape)


	LR_e = [(100 * abs(Z_cap[iz] - Z[iz])/Z[iz]) for iz in range(len(Z_cap))]
	LR_e = np.concatenate( LR_e, axis=0 )


	fig = plt.figure('error rate')	
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.plot(LR_e)
	ax1.set_xlabel('points')
	ax1.set_ylabel('mesh error(%)')
	ax1.grid(True)
	
	ax2 = fig.add_subplot(1, 2, 2, projection ="3d")	
	ax2.scatter(X, Y, Z, c='r', s=2)
	ax2.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
	ax2.set_xlabel('$r_{cap}$')
	ax2.set_ylabel('$y_{val}$')
	ax2.set_zlabel('$r=1/VDD$')

#	figManager = plt.get_current_fig_manager()
#	figManager.window.showMaximized()
	fig.tight_layout(pad=0, w_pad=0, h_pad=0)

main()

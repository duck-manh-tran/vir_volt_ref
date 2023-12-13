import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import numpy as np
import math
from functions import * 
import sys 
def main():
	if sys.argv[1] == 'get_points':
		# Measurement process
		filename = 'result/temp/l180nm/nf_6/discharge_curve_28C.txt'
		get_ctat_points(filename)

	elif sys.argv[1] == 'fit_mesh':
		# curve fitting process
		data = np.loadtxt(f"data.txt", delimiter='\t')
		X = data[:, 0]
		Y1 = data[:, 1]
		Y2 = data[:, 2]
		Z = data[:, 3]
		order = 5	
		C_coeff = curve_fitting(order, X, Y1, Z)
		np.savetxt(f"C_coeff.txt", C_coeff, delimiter='\t')
		plot_fitting_mesh(order, C_coeff, X, Y1, Z)
	
		plt.figure(1)
		plt.title("Y_ctat = exp((n[0]-n[1])/(n[2]-1)-2)")

		plt.figure(2)
		plt.title("Y_ctat = exp((n[0]-n[1])/(n[2]-1)-2)")

		C_coeff_2 = curve_fitting(order, X, Y2, Z)
		plot_fitting_mesh(order, C_coeff_2, X, Y2, Z)

		plt.figure(3)
		plt.title("Y_ctat=exp((n[0]-n[1])/(n[1]-n[2])+4)")	
	
		plt.figure(4)
		plt.title("Y_ctat=exp((n[0]-n[1])/(n[1]-n[2])+4)")	
	
		plt.show()
	else:
		print ("Please select your option:")
		print ("option_1:get_points")
		print ('option_2:fit_mesh')

def get_ctat_points(filename):
	data = np.loadtxt(filename, delimiter='\t', skiprows=1, dtype=str)
	index = list(map(int, data[:,0]))
	time = list(map(float, data[:,1]))
	capa = list(map(float, data[:,2]))
	
	# setup initial points and stop points
	ipoint = [];
	VDD = [];
	k_thres = 0.25
	k = [0.4, 0.3, 0.25]
	m_ = [0, 0, 0]
	m = [0, 0, 0]
	n = [0, 0, 0]
	y1 = []
	y2 = []
	r = []
	r_cap = []
	for iv in range(21):  #(41):
		VDD.append(0.4 + 0.02 * iv)
		for ir in range(26):     #(126):
			r.append(round(1/VDD[iv], 8))
			r_cap.append(round (1.25 + 0.05 * ir, 4))
			for im in range(3):
				m[im] = round (pow(2, 16) * r_cap[ir] * k[im])
				m_[im] = m[im]/pow(2, 16)
				v_start = VDD[iv] * m_[im]
				v_stop = VDD[iv] * k_thres
				n[im] = round (pow(10, 6) * meas_discharge_time1 (time, capa, v_start, v_stop))
			y_1 = math.exp( (n[0] - n[1])/(n[2] - 1) - 2)
			y_2 = math.exp( (n[0] - n[1])/(n[1]-n[2]) + 4 )
			y1.append(round (y_1, 2))
			y2.append(round (y_2, 2))
	 
	result = list(zip(r_cap, y1, y2, r))
	np.savetxt(f"data.txt", result, fmt='%10.6f', delimiter='\t')

def curve_fitting(order, X, Y, Z):
	# 1=linear, 2=quadratic, 3=cubic, ..., nth degree
	X = X.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	Z = Z.reshape(-1, 1)
	
	# calculate exponents of design matrix
	e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
	eX = np.asarray([[x] for x,_ in e]).T
	eY = np.asarray([[y] for _,y in e]).T
	
	# best-fit polynomial surface
	A = (X ** eX) * (Y ** eY)
	C,resid,_,_ = lstsq(A, Z)    # coefficients
	print(resid)
	
	# print summary
	print(f'data = {Z.size}x3')
	print(f'model = {exp2model(e)}')
	print(f'coefficients =\n{C}')
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
	plt.figure()	
	plt.plot(LR_e)
	plt.xlabel('points')
	plt.ylabel('mesh error(%)')
	
	plt.figure()	
	ax = plt.axes(projection ="3d")	
	ax.scatter(X, Y, Z, c='r', s=2)
	ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2, linewidth=0.5, edgecolor='b')
	ax.axis('tight')
	ax.view_init(azim=-60.0, elev=30.0)
	ax.set_xlabel('r_cap')
	ax.set_ylabel('Yctat')
	ax.set_zlabel('r = 1/VDD')

main()

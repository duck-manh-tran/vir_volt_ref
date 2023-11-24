#/bin/bash python
import sys
import os
import numpy as np
import math

def main():
	r = 1.5
	N = 12
	k = np.array([0.4, 0.3, 0.25])
	m = np.array([0, 0, 0])
	if (sys.argv[1] == "0"):
		print(round(r * k[0] * 2**N))
#		print(format(round(r * k[0] * 2**N), '012b'))
	if (sys.argv[1] == "1"):
		print(round(r * k[1] * 2**N))
#		print(format(round(r * k[1] * 2**N), '012b'))
	if (sys.argv[1] == "2"):
		print(round(r * k[2] * 2**N))
#		print(format(round(r * k[2] * 2**N), '012b'))
	if (sys.argv[1] == "3"):
		m_ref = get_m_set(get_discharge_val(), r, N)
		print(m_ref)

def get_m_set(n_val, r_, N):
	Y = math.exp( (n_val[0] - n_val[1])/(n_val[1] - n_val[2]) + 4 )
	X = r_
	# perform function
	filename = 'C_coeff.txt'
	data = np.loadtxt(filename, delimiter='\t', dtype=str)
	C = list(map(float, data))
	
	order = 5
	e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
	eX = np.asarray([[x] for x,_ in e]).T
	eY = np.asarray([[y] for _,y in e]).T
	A = (X ** eX) * (Y ** eY)
	Z_cap = np.dot(A, C)
	# print (1/Z_cap)
	m_out = round((0.35*Z_cap[0])*(2**N))
	return m_out
	
	# r_CTAT = C[0] \
	# 		 + C[1]*X + C[2]*Y \
	# 		 + C[3]*X**2 + C[4]*X*Y + C[5]*Y**2 \
	# 		 + C[6]*X**3 + C[7]*X**2*Y + C[8]*X*Y**2 + C[9]*Y**3 \
	# 		 + C[10]*X**4 + C[11]*X**3*Y + C[12]*X**2*Y**2 + C[13]*X*Y**3 + C[14]*Y**4 \
	# 		 + C[15]*X**5 + C[16]*X**4*Y + C[17]*X**3*Y**2 + C[18]*X**2*Y**3 + C[19]*X*Y**4 + C[20]*Y**5 \
	# 		 + C[21]*X**6 + C[22]*X**5*Y + C[23]*X**4*Y**2 + C[24]*X**3*Y**3 + C[25]*X**2*Y**4 + C[26]*X*Y**5 + C[27]*Y**6 \
	# 

def get_discharge_val():
	infile = [" stop1.txt", " stop2.txt", " stop3.txt"]
	outfile = " tmp.txt"
	pos_ = []
	for ix in range(3):
		cmd1 = "awk \'{if($2 == \"1s\" || $2 == \"0s\"){$1=\"\"; print $0;}}\'"
		cmd1 = cmd1 + infile[ix] + " > " + outfile
		# print (cmd1)
		os.system(cmd1)
		cmd2 = "sed -i -e \'s/s//g\' -e \'s/ //g\'"
		cmd2 = cmd2 + outfile 
		# print (cmd2)
		os.system(cmd2)
		input1 =os.popen("wc -l tmp.txt").read()
		NoL = int(input1.split(' ')[0])
		pos_.append(NoL-1)
		
	# print (pos_)
	n_val = []
	for ix in range(3):
		cmd = "sed -n " + str(pos_[ix]) + "p tmp.txt"
		# print (cmd)
		input2 = os.popen(cmd).read()
		n_val.append(int(input2, 2))
	np.savetxt('discharge_cnt.txt', n_val, fmt='%d')
	return n_val

main()

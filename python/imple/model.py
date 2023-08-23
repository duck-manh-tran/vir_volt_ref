import numpy as np
import os
import math
v_dd = 0.4675
r_hat = 1.5
r = r_hat
N = 16
k = [0.4, 0.3, 0.25]
m = [0, 0, 0]

set_m =[]
for im in range(3):
	m[im] = round (r * k[im] * 2 ** N)
	chain = ' -D m' + str(im+1) + '=' + str(m[im])
	set_m.append(chain)
set_vdd = ' -D v_dd=' + str(v_dd)
cmd = 'ngspice -b -i d_bgr_model.spice'
cmd = cmd + set_vdd + set_m[0] + set_m[1] + set_m[2]
print (cmd)
os.system(cmd)

# load data from txt file to python
cmd2 = "sed -i -e \"1,5d\" discharge_time.txt"
print (cmd2)
os.system(cmd2)
filename = 'discharge_time.txt'
data = np.loadtxt(filename, delimiter='\t', dtype=str)
time = list(map(float, data[:,1]))
n = []
for ix in range(3):
	n.append(round(time[ix] * 1e6))
print (n)
Y = math.exp( (n[0] - n[1])/(n[1] - n[2]) + 4 )
X = r

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
print (1/Z_cap)
m_out = round((0.35*Z_cap[0])*(2**N))
print (m_out)

set_op = ' -D set=1'
m_set = ' -D m_set=' + str(m_out)
cmd3 = 'ngspice -b -i d_bgr_model.spice'
cmd3 = cmd3 + set_vdd + set_op + m_set
print (cmd3)
os.system(cmd3)

# r_CTAT = C[0] \
# 		 + C[1]*X + C[2]*Y \
# 		 + C[3]*X**2 + C[4]*X*Y + C[5]*Y**2 \
# 		 + C[6]*X**3 + C[7]*X**2*Y + C[8]*X*Y**2 + C[9]*Y**3 \
# 		 + C[10]*X**4 + C[11]*X**3*Y + C[12]*X**2*Y**2 + C[13]*X*Y**3 + C[14]*Y**4 \
# 		 + C[15]*X**5 + C[16]*X**4*Y + C[17]*X**3*Y**2 + C[18]*X**2*Y**3 + C[19]*X*Y**4 + C[20]*Y**5 \
# 		 + C[21]*X**6 + C[22]*X**5*Y + C[23]*X**4*Y**2 + C[24]*X**3*Y**3 + C[25]*X**2*Y**4 + C[26]*X*Y**5 + C[27]*Y**6 \
# 



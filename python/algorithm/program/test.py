import os
import numpy as np


def main():
	test2()



def test2():
	order = 4
	e_1 = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
	
	e_2 = []
	for ni in range (order+1):
		for yi in range (ni+1):
			for xi in range (ni+1):
				if xi+yi==ni:
					e_2.append([xi, yi])

	eX = np.asarray([[x] for x,_ in e_1]).T
	eY = np.asarray([[y] for _,y in e_1]).T

	e2X = np.array([xi for xi,_ in e_2]).T
	e2Y = np.array([yi for _,yi in e_2]).T
	
	print(e_1)
	print(e_2)
	print(type(e_1), type(e_2))
	if e_1 == e_2:
		print ('same')

	print(eX)
	print(eY)

	print(e2X)
	print(e2Y)
	
	print (type(e2Y))
	print (type(eY))
	
	X = 1.25
	Y = 10.5
	A = (X ** eX) * (Y ** eY)
	
	C = np.loadtxt('C_coeff.txt', dtype=float)
#	Z = np.dot(A, C)
#	print(Z)
	order = 1
	len_order = 3
	print ('Length of C: ', len(C))
	while len(C) > len_order:
		order = order+1
		len_order = 0.5*order**2 + 1.5*order + 1

	print('number of order: ', order)
	

def test1():
	vdds = np.arange(0.4, 0.81, 0.1)
	temps = np.arange(0, 100, 20)
	inits = np.arange(1.25, 2, 0.025)
	
	data = []
	for vdd in vdds:
		for temp in temps:
			for init in inits:
				element = [temp, vdd, init]
				data.append(element)
	
	# file.txt should be replaced with
	# the actual text file name
	def gen_in_data(data):
		os.system('cp tmp.txt des.txt')
		val_names = '+ temp_val vdd init'
		with open('des.txt', 'a') as f:
			f.write(val_names)
			f.write('\n')
		
		def list2str(list):
			string = ''
			for ele in list:
				string = string + ' ' + str(round(ele, 3))
			string = '+' + string
			return string
		
		with open('des.txt', 'a') as f:
			for element in data: 
				f.write(list2str(element)) 
				f.write('\n') 
			f.write('.ENDDATA')
	
	gen_in_data(data)
	os.system('cat des.txt')

main()

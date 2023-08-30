#/bin/bash python
import sys
import numpy as np

r=1.5
N = 16
k = np.array([0.4, 0.3, 0.25])
m = np.array([0, 0, 0])
if (sys.argv[1] == "0"):
	print(round(r * k[0] * 2**N))
if (sys.argv[1] == "1"):
	print(round(r * k[1] * 2**N))
if (sys.argv[1] == "2"):
	print(round(r * k[2] * 2**N))
if (sys.argv[1] == "3"):
	for i in range(3):
		m[i] = sys.argv[i+2]
	print (32768)


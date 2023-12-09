import os
import numpy as np

home_dir = "result/temp/l180nm/nf_6/discharge_curve_"
for ix in range(50):
	if (2*ix < 10):
		infile = home_dir + "0" + str(ix*2) + "C.txt"
	else:
		infile = home_dir + str(ix*2) + "C.txt"
	cmd="sed -e '/\o14/d' -e '/sch_path:/d' -e '/---/d' -e '/Index/d' -e '/Transient/d' -i " + infile
	os.system(cmd)
	print (cmd)

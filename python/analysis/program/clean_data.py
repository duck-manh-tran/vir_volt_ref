import os

for ix in range(1):
	if (2*ix < 10):
		infile = "result/temp/text/discharge_nwk_0" + str(ix*2) + "C.txt"
	else:
		infile = "result/temp/text/discharge_nwk_" + str(ix*2) + "C.txt"
	cmd="sed -e '/\o14/d' -e '/sch_path:/d' -e '/---/d' -e '/Index/d' -e '/Transient/d' -i " + infile
	os.system(cmd)


import os
import numpy as np

def clear_log():
   cmd = "echo  > clean_files.log"
   os.system(cmd)

def write_log (file):
   cmd = "echo ========================================================== >> clean_files.log"
   os.system(cmd)
   cmd = "cat " + file + " >> clean_files.log"
   os.system(cmd)
home_dir = "result/temp/l180nm/nf_6/discharge_curve_"

for ix in range(50):
	if (2*ix < 10):
		infile = home_dir + "0" + str(ix*2) + "C.txt"
	else:
		infile = home_dir + str(ix*2) + "C.txt"
	cmd="sed -e '/\o14/d' -e '/sch_path:/d' -e '/---/d' -e '/Index/d' -e '/Transient/d' -i " + infile
	os.system(cmd)

# get Voltage from
clear_log()

mylist = ".tmp"
mylist1 = ".tmp2"

top_vol = " \"8\.010\" "
bot_vol = " \"9\.900\" "


print ("get positions of the top voltage")

cmd = "grep -i " + top_vol + home_dir + "* > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "awk '{$2=\"\"; print $0}' " + mylist + " > " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "grep -i \"e-01\" " + mylist1 + " > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "grep -i " + top_vol + mylist + " > " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "sed -i 's/:/\t/g' " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "awk '!a[$1]++' " + mylist1 + " > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "awk '{print($2)}' " + mylist + " > head_line"
os.system(cmd)

print ("get positions of the bottom voltage")

## get Voltage to
cmd = "grep -i" + bot_vol + " " + home_dir + "* > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "awk '{$2=\"\"; print $0}' " + mylist + " > " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "grep -i \"e-02\" " + mylist1 + " > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "grep -i " + bot_vol +  mylist + " > " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "sed -i 's/:/\t/g' " + mylist1
os.system(cmd)
write_log(mylist1)

cmd = "awk '!a[$1]++' " + mylist1 + " > " + mylist
os.system(cmd)
write_log(mylist)

cmd = "awk '{print($2)}' " + mylist + " > tail_line"
os.system(cmd)

head_line = np.loadtxt(f"head_line")
tail_line = np.loadtxt(f"tail_line")

for ix in range(50):
   if (2*ix < 10):
      infile = " " + home_dir + "0" + str(ix*2) + "C.txt"
      outfile = " > result/temp/l180nm/nf_6/cleaned_data/discharge_curve_0" + str(ix*2) + "C.txt"
   else:
      infile = " " + home_dir + str(ix*2) + "C.txt"
      outfile = " > result/temp/l180nm/nf_6/cleaned_data/discharge_curve_" + str(ix*2) + "C.txt"

   cmd = "sed -n '" + str(int(head_line[ix]+1)) + "," + str(int(tail_line[ix]+1)) + "p'" + infile + outfile
   os.system(cmd)
   print (cmd)

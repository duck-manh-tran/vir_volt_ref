
vcs:
#	vcs -full64 -R -ad -timescale=1ns/1ps -f vlist -l vcs.log
	vcs -full64 -ad -timescale=1ns/1ps -debug_access+all -f vlist -l vcs.log
#	vcs -full64 -R -ad -timescale=1ns/1ps -debug_access+all -f vlist -l vcs.log

run_simv:
	./simv -ucli -i ucli.cmd

wave:
	wv top.fsdb &
	
clean:
	-rm -rf cosim* nova* csrc simv* sxcmd.log* \
		ucli.key verdiLog vfastLog DVEfiles *log* *fpi \
		AN.DB
clean_wave:
	-rm -rf *fsdb *vpd

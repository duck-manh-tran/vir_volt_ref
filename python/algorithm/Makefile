HSPICE_DIR=$(VVR_HOME)/python/algorithm/netlists/tsmc65_hspice
VA_MODEL=$(VVR_HOME)/synopsys/vir_volt_ref/system/veriloga/veriloga.va
NETLIST=$(HSPICE_DIR)/system_test.spice
OUTPUT=$(HSPICE_DIR)/simulation

normal_mode:
	rm -f result.csv
	hspice -mt 4 -i $(NETLIST) -o $(OUTPUT)
	python3 program/analysis_va_model.py
	mv veriloga.log $(OUTPUT)

collect_mode:
	rm -f data.csv
	hspice -dp 10 -i $(NETLIST) -o $(OUTPUT)
	python3 program/volt_comp_func.py
	
env:
	source ../../env.sh

va:
	vim $(VA_MODEL) 

spi:
	vim $(NETLIST)

debug:
	vim simulation/system_test.lis

all: normal_mode

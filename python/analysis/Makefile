simulate:
	mkdir -p result/temp/l180nm/nf_6/cleaned_data
	python3 program/ctrl_simulations.py 

filtering_data:
	python3 program/clean_data.py
	python3 program/get_discharge_time_at_all_temps_sweep_points.py
	python3 program/get_discharge_time_all.py	

volt:
	python3 program/analysis_volt.py
temp:
	python3 program/analysis_temp.py
clean: 

.PHONY: clean simulate filtering_data volt temp

 

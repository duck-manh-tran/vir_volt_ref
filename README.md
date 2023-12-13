# virtual_voltage_reference
This is the repo of the project of the vitual voltage reference 

I.Analyze discharge behavior
please go to the dir: ./python/analysis

1. Run discharge_network simulations:
	make simulate

2. Filter data from result files
	make filtering_data

3. Analyze discharge_network by Python program:
	make volt
	make temp

4. Generate CTAT points and fit to a mesh
	python3 program/get_ctat_coeff.py get_points
	python3 program/get_ctat_coeff.py fit_mesh


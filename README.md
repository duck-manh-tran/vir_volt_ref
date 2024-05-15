# Virtual_Voltage_Reference
This is the repo of the project of the virtual voltage reference 

## Analysis of discharge networks 
This analysis shows voltage and temperature variation 
To run these analyses, please go to the analysis directory:
	
 	cd ./python/anls_dsch_nwks

There are 2 types of analyses (voltage, temperature), 3 library_simulator options (), and 8 discharge networks (a, b, c, d, e, f, g, h). Before running the analysis program, let's choose a few options in the file './program/analysis_dsn.py' as below:

1. Choose 1 of 2 analysis types:
* sim_type = 'volt'
* sim_type = 'temp'

2. Choose 1 in 3 library-simulator options:
* lib = 'tsmc65_hspice'	(Full voltage and temperature analyses)
* lib = 'sky130a_ngspice' (only voltage analyses are available)
* lib = 'sky130a_spectre' (In the developing)

3. Choose 1 in 8 discharge networks:
* dsnwk = 'dschr_nwk_type_a'
* dsnwk = 'dschr_nwk_type_b'
* dsnwk = 'dschr_nwk_type_c'
* dsnwk = 'dschr_nwk_type_d'
* dsnwk = 'dschr_nwk_type_e'
* dsnwk = 'dschr_nwk_type_f'
* dsnwk = 'dschr_nwk_type_g'
* dsnwk = 'dschr_nwk_type_h'

Finally, to show the analysis results, please run the following command:

	python3 ./program/analysis_dsn.py

<b>Notice:</b> The processed files are included in this repo, therefore you can show the result graphs without any simulation. If you want to re-run any simulation, please turn on the flag 'ovr_flag' in the analysis program.
* ovr_flag = True
  
## The following section is defining
  
I.Analyze discharge behavior  
	please go to the dir: ./python/analysis
	$ cd ./python/analysis

1. Run discharge_network simulations:  
	$ make simulate

2. Filter data from result files:  
	$ make filtering_data

3. Analyze discharge_network by Python program:  
	$ make volt  
	$ make temp  
	
4. Generate CTAT points and fit to a mesh:  
	python3 program/get_ctat_coeff.py get_points  
	python3 program/get_ctat_coeff.py fit_mesh  
   	Or  
   	$ make filtering_data  
   	$ make fit_mesh


	If the result file exists, the program only shows the graph.
	If the result file does not exist, the program runs simulations and then shows the graph.
	If you want to re-simulate a discharge network, please remove a result file and run the analysis_dsn.py program
		Example: $ rm results/dschr_nwk_1t_type_a
					$ python3 program/analysis_dsn.py

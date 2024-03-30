# virtual_voltage_reference
This is the repo of the project of the vitual voltage reference 

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


II. Analysis various discharge networks
1. change model name of the discharge network
	From repo dir, go to directory "./python/anls_dsch_nwks/"
	$ cd ./python/anls_dsch_nwks/

	Edit analysis control program "program/analysis_dns.py"
	$ vi program/analysis_dsn.py

	change discharge network by edit 'model' variable to names below:
		dschr_nwk_1t_type_a  
		dschr_nwk_1t_type_c  
		dschr_nwk_1t_type_e  
		dschr_nwk_1t_type_f  
		dschr_nwk_1t_type_g  
		dschr_nwk_2t_type_b  
		dschr_nwk_2t_type_d	

	Run simulation or show graph: 
	$ python3 ./program/analysis_dsn.py

	If the result file is exist, the program only shows the graph.
	If the result file is not exist, the program runs simulations and then shows the graph.
	If you want to re-simulate a discharge network, please remove a result file and run the analysis_dsn.py program
		Example: $ rm results/dschr_nwk_1t_type_a
					$ python3 program/analysis_dsn.py
		

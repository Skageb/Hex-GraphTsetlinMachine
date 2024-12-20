# Hex-GraphTsetlinMachine
Repository for the Hex winner prediction project using Graph-Tsetline-Machine in IKT457-Learning Systems. 

# Repo structure

* main.ipynb: Notebook containing graph initalization, training and testing of the GTM.

* main.py: Same content as main.ipynb converted to python script with an argparser. Used to run a queue of different configurations.

* run_multiple_board_sizes.sh: Used to run main.py with different configurations.

* plot.py: Plotting functionality for the report.

* hex_dataset_generation.py: is used to generate hex datasets and write them to hex_datasets/ folder.

* results: The best result for each board size and number of turns left is logged including performance metrics and hyper parameters.
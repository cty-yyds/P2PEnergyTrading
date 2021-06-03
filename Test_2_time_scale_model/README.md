# File content

2018price.pkl is the P2P electricity price data file.

res_mg_1hour.pkl and res_mg_15min.pkl are the generation and demand data file of the MEMG at one-hour and 15-minutes resolution.

Residential_MES.py is the environment of a MEMG which includes the reward functions and system transition models.

two_timescale_TD3.py is the MATD3 agents file which includes NN models and implementing ideas of centralised training and decentrailsed execution.

test_2timescale_true_state.py is the main file to set parameters and train the agents.

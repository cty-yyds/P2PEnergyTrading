# P2P Energy Trading
In this project, we investigate the external P2P energy trading problem and internal energy conversion problem for a
multi-energy microgrid (MEMG). These two problems are multi-timescale and complex decision-making problems with
enormous high-dimensional data and uncertainty, so a multi-agent deep reinforcement learning approach combining the
multi-agent actor-critic algorithm with the twin delayed deep deterministic policy gradient algorithm is proposed. The
proposed approach can handle the high-dimensional continuous action space and two timescale between P2P energy
trading and energy conversion. Simulation results based on real-world MG datasets show that the proposed approach
significantly reduces the MEMG’s average hourly operation cost. The impact of carbon tax pricing is also considered.

## File content
2018price.pkl is the P2P electricity price data file.

res_mg_1hour.pkl and res_mg_15min.pkl are the generation and demand data file of the MEMG at one-hour and 15-minutes resolution.

Residential_MES.py is the environment of a MEMG which includes the reward functions and system transition models.

two_timescale_TD3.py is the MATD3 agents file which includes NN models and implementing ideas of centralised training and decentrailsed execution.

test_2timescale_true_state.py is the main file to set parameters and train the agents.

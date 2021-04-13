import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def check_1h_15min_data(env):
    for i in range(24):
        solar_1h = env.generation_demand_1hour.iloc[i, 0]
        e_load_1h = env.generation_demand_1hour.iloc[i, 1]
        h_load_1h = env.generation_demand_1hour.iloc[i, 2]
        solar_15min = env.generation_demand_15min.iloc[i*4:i*4+4, 0].sum()
        e_load_15min = env.generation_demand_15min.iloc[i*4:i*4+4, 1].sum()
        h_load_15min = env.generation_demand_15min.iloc[i*4:i*4+4, 2].sum()
        assert solar_1h == solar_15min, f"solar not met at {i} hour"
        assert e_load_1h == e_load_15min, f"e load not met at {i} hour"
        assert h_load_1h == h_load_15min, f"h load not met at {i} hour"
    print("We are good")


class ResidentialMicrogrid:
    def __init__(self, B_e, B_h2, pes_max, HT_p_max):
        self.n_features_1h = 6  # state: solar,E-H_demand,battery,h2_level,E_price
        self.n_features_15min = 7  # state: solar,E-H_demand,battery,h2_level, energy trading amount_E-H
        self.WE_max = 1  # Water Electrolyser input max
        self.FC_max = 1  # Fuel cell input max
        self.HB_max = 1  # Boiler hydrogen input max
        self.Etrade_max = 1  # Electricity trading max
        self.Gtrade_max = 1  # Natural gas trading (m3) max
        self.n_actions_1h = 2  # Y_elec,Y_gas
        self.n_actions_15min = 3  # WE,FC,HB
        self.battery = 0.0  # battery level
        self.hydrogen = 0.0  # Hydrogen storage level
        self.B_e = B_e  # Battery capacity
        self.B_h2 = B_h2  # Hydrogen storage capacity
        self.pes_max = pes_max  # battery power limit
        self.HT_p_max = HT_p_max  # hydrogen tank flow limit
        self.step_1h = 0
        self.step_15m = 0
        self.build_MG()

    def build_MG(self):
        self.generation_demand_1hour = pd.read_pickle('res_mg_1hour.pkl')['2019-5-15':'2019-05-16 00:00:00	']
        self.generation_demand_15min = pd.read_pickle('res_mg_15min.pkl')['2019-5-15':'2019-05-16 00:00:00	']
        self.price = pd.read_pickle('2018price.pkl')['2018-05-16':'2018-05-17 00:00:00	']  # Electricity price ($/Kwh)
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        scaler3 = StandardScaler()
        self.normalized_gen_dem_1hour = scaler1.fit_transform(self.generation_demand_1hour.values)
        self.normalized_gen_dem_15min = scaler2.fit_transform(self.generation_demand_15min.values)
        self.normalized_price = scaler3.fit_transform(self.price.values.reshape(-1, 1))
        self.gas_price = np.array([0.15820997, 0.17198271, 0.14196519, 0.13808057, 0.1345491, 0.13348966,
                                   0.13313651, 0.12995819, 0.13278337, 0.14267149, 0.15962256, 0.19352469])

        # dollar per cubic meter
        # self.E_trading, self.H_trading = 0, 0  # The energy trading amount

    def reset(self):
        self.battery = 0.0
        self.hydrogen = 0.0
        # self.E_trading = np.random.uniform(-1, 1)
        # self.H_trading = np.random.uniform(-1, 1)
        # reset state (reduce scale for DNN)
        state_1hour = np.hstack((self.normalized_gen_dem_1hour[0, 0],  # solar
                                 self.normalized_gen_dem_1hour[0, 1],  # electricity
                                 self.normalized_gen_dem_1hour[0, 2],  # heat
                                 self.battery / self.B_e, self.hydrogen / self.B_h2,
                                 self.normalized_price[0, 0]))

        state_15m = np.hstack((self.normalized_gen_dem_15min[0, 0],
                               self.normalized_gen_dem_15min[0, 1],
                               self.normalized_gen_dem_15min[0, 2],
                               self.battery / self.B_e, self.hydrogen / self.B_h2))
        # reset step
        self.step_1h = 0
        self.step_15m = 0
        return state_1hour, state_15m

    def sample_trading(self):
        # random select trading actions
        y_elec = np.random.uniform(0, self.Etrade_max)  # *2-1==>(-1, 1) sigmoid to tanh
        y_gas = np.random.uniform(0, self.Gtrade_max)

        return np.array([y_elec, y_gas])

    def sample_conversion(self):
        # random select conversion actions
        a_WE = np.random.uniform(0, self.WE_max)
        a_FC = np.random.uniform(0, self.FC_max)
        a_HB = np.random.uniform(0, self.HB_max)

        return np.array([a_WE, a_FC, a_HB])

    def convert_energy(self, trading_action, conversion_action):
        # -----------------------parameters----------------------------
        p_P2P_e = self.price.iloc[self.step_1h]  # p2p electricity price
        p_ng = self.gas_price[4]  # natural gas price (May)
        p_h2 = 5  # hydrogen gas price ($/kg)

        s_bg = 1.2  # price ratio of buying from grid to p2p
        s_sg = 0.8  # price ratio of selling from grid to p2p

        k_we = 0.8  # water electrolyser efficiency
        k_fc_e = 0.6  # fuel cell to electricity efficiency
        k_fc_h = 0.25  # fuel cell to heat efficiency
        k_gb = 0.9  # gas boiler efficiency
        k_ng2q = 8.816  # natural gas(m3) to Q(KWh) ratio
        k_h2q = 33.33  # hydrogen(kg) to Q(KWh) ratio

        c_p = 3 * p_P2P_e  # electricity penalty coefficient
        c_h = 2 * p_ng  # heat penalty coefficient

        beta_gas = 0.245  # carbon intensity kg/kwh
        beta_elec = 0.683  # carbon intensity kg/kwh
        alpha_co2 = 0.0316  # carbon tax $/kg

        reward = 0
        E_dif = 0  # electricity balance difference (Kwh)
        h2_dif = 0  # h2 balance difference (kg)
        H_dif = 0  # heat balance difference (m3)
        co2_emission = 0

        # ---------------------energy conversion----------------------
        a_WE = conversion_action[0] * 200 / 4  # kwh
        a_FC = conversion_action[1] * 8 / 4  # kg
        a_HB = conversion_action[2]  # kg
        y_elec = (trading_action[0] * 2 - 1) * 100 / 4  # kwh
        if y_elec > 0:
            co2_emission += y_elec * beta_elec
        y_gas = trading_action[1] * 10 / 4  # m3
        co2_emission += y_gas * k_ng2q * beta_gas

        E_FC = a_FC * k_h2q * k_fc_e  # electricity (kwh) output from FC
        H_FC = a_FC * k_h2q * k_fc_h  # heat (kwh) output from FC
        h2_WE = a_WE * k_we / k_h2q  # hydrogen output (kg) from WE
        H_GB = y_gas * k_ng2q * k_gb  # heat output from natural gas boiler (Kwh)
        H_HB = a_HB * k_h2q * k_gb  # heat output from hydrogen gas boiler (Kwh)

        # ---------------------energy balance-------------------------
        solar = self.generation_demand_15min.iloc[self.step_15m, 0]  # solar output
        E_demand = self.generation_demand_15min.iloc[self.step_15m, 1]  # electricity load
        H_demand = self.generation_demand_15min.iloc[self.step_15m, 2]  # heat load
        # reward += pL1 * pp * 1.8  # retail profit

        E_battery = solar + y_elec + E_FC - a_WE - E_demand  # battery charging (if>0) amount by electricity balance
        if E_battery < -self.pes_max:  # battery level and penalty
            self.battery += -self.pes_max / 0.9
            E_dif += -self.pes_max - E_battery
        if -self.pes_max <= E_battery < 0:
            self.battery += E_battery / 0.9
        if 0 < E_battery <= self.pes_max:
            self.battery += E_battery * 0.9
        if E_battery > self.pes_max:
            self.battery += self.pes_max * 0.9  # no penalty as described in the paper
        if self.battery > self.B_e:
            self.battery = self.B_e  # no penalty as described in the paper
        if self.battery < 0:
            E_dif += -self.battery
            self.battery = 0

        h2_HT = h2_WE - a_FC - a_HB  # hydrogen tank inflow amount (if>0) kg by hydrogen balance
        if h2_HT < -self.HT_p_max:  # hydrogen tank level and cost
            self.hydrogen += -self.HT_p_max / 0.95
            h2_dif += -self.HT_p_max - h2_HT
        if -self.HT_p_max <= h2_HT < 0:
            self.hydrogen += h2_HT / 0.95
        if 0 < h2_HT <= self.HT_p_max:
            self.hydrogen += h2_HT * 0.95
        if h2_HT > self.HT_p_max:
            self.hydrogen += self.HT_p_max * 0.95
            # don't sell surplus hydrogen as we don't want to use hydrogen in this way
        if self.hydrogen > self.B_h2:
            self.hydrogen = self.B_h2  # as above
        if self.hydrogen < 0:
            h2_dif += -self.hydrogen
            self.hydrogen = 0

        if H_GB + H_FC + H_HB < H_demand:  # heat balance
            H_dif += (H_demand - H_GB - H_FC - H_HB) / k_ng2q

        # -------------------calculating reward-----------------------
        # reward -= E_trading * p_P2P_e  # E trading profit/cost
        if y_elec >= 0:
            reward -= y_elec * p_P2P_e * s_bg  # buy from grid
        else:
            reward -= y_elec * p_P2P_e * s_sg
        reward -= y_gas * p_ng  # buy natural gas

        reward -= E_dif * c_p + ((H_dif/10) ** 2 + H_dif) * c_h  # penalty
        reward -= h2_dif * p_h2  # buy needed hydrogen
        reward -= co2_emission * alpha_co2  # carbon tax

        # ------------------------next state--------------------------
        self.step_15m += 1  # next step
        if self.step_15m % 4 == 0:
            self.step_1h += 1

        s1h_ = np.hstack((self.normalized_gen_dem_1hour[self.step_1h, 0],
                          self.normalized_gen_dem_1hour[self.step_1h, 1],
                          self.normalized_gen_dem_1hour[self.step_1h, 2],
                          self.battery / self.B_e, self.hydrogen / self.B_h2,
                          self.normalized_price[self.step_1h, 0]))

        s15m_ = np.hstack((self.normalized_gen_dem_15min[self.step_15m, 0],
                           self.normalized_gen_dem_15min[self.step_15m, 1],
                           self.normalized_gen_dem_15min[self.step_15m, 2],
                           self.battery / self.B_e, self.hydrogen / self.B_h2))
        # need to concatenate trading action into 15 min states

        return reward / 10, s15m_, s1h_

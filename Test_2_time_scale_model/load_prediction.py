import numpy as np
from collections import deque
from Residential_MES import ResidentialMicrogrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def create_data_for_regression():
    np.random.seed(42)
    env = ResidentialMicrogrid(B_e=300, B_h2=30, pes_max=25, HT_p_max=3)
    data_list = deque()

    for test_episode in range(500):
        s_1h, s_15min_6states, _ = env.reset()
        trading_a = env.sample_trading()

        for i in range(96):
            solar = env.generation_demand_15min.iloc[i, 0]
            conversion_a = env.sample_conversion()
            E_demand = env.generation_demand_15min.iloc[i, 1]

            # Step the env
            reward, s2_15min_6states, s2_1h, _ = env.convert_energy(trading_a, conversion_a)
            delta_battery = s2_15min_6states[3] - s_15min_6states[3]
            s_15min_6states = s2_15min_6states

            # store the data y_elec, a_WE, a_FC
            data_list.append([delta_battery, solar, trading_a[0], conversion_a[0], conversion_a[1], E_demand])

            if i % 4 == 3:
                trading_a = env.sample_trading()

    data_for_regression = np.array(data_list)
    input_data = data_for_regression[:, :-1]
    target = data_for_regression[:, -1]
    return input_data, target


if __name__ == "__main__":
    X, y = create_data_for_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_predict = lin_reg.predict(X_test)
    accuracy = np.abs(y_predict-y_test)/y_test
    print(f'Test Accuracy is {1-np.mean(accuracy)}')

import os
import time
import pickle

import numpy as np
import spiceypy
from tqdm import tqdm

from .optimize_times import optimize_times


def generate_optimized_data(asteroid_list, m_1, m_2, m_3,
                            launch_utc_min, launch_utc_max):
    num_asteroids = len(asteroid_list)
    total_operations = num_asteroids ** 3

    elapsed_times = np.zeros(total_operations)

    default_entry = {
        'delta_v_launch': np.inf,
        'delta_v_A1_arrive': np.inf,
        'delta_v_A1_leave': np.inf,
        'delta_v_A2_arrive': np.inf,
        'delta_v_A2_leave': np.inf,
        'delta_v_A3_arrive': np.inf,
        'delta_v_total': np.inf,
        'et_launch': np.inf,
        'et_arrive_1': np.inf,
        'et_stay_1': np.inf,
        'et_arrive_2': np.inf,
        'et_stay_2': np.inf,
        'et_arrive_3': np.inf,
    }

    asteroid_optimized_data = [
        [[dict(default_entry) for _ in range(num_asteroids)]
         for _ in range(num_asteroids)]
        for _ in range(num_asteroids)
    ]

    et_launch_min = spiceypy.str2et(launch_utc_min)
    et_launch_max = spiceypy.str2et(launch_utc_max)

    launch_dates = [et_launch_min, et_launch_max]

    pbar = tqdm(total=total_operations,
                desc=f'Ms: {m_1}, {m_2}, {m_3}')

    for i in range(num_asteroids):
        for j in range(num_asteroids):
            for k in range(num_asteroids):
                current_operation = k + j * num_asteroids + i * num_asteroids ** 2
                t_start = time.time()

                a_id_1 = asteroid_list[i]['ID']
                a_id_2 = asteroid_list[j]['ID']
                a_id_3 = asteroid_list[k]['ID']

                if a_id_1 == a_id_2 or a_id_2 == a_id_3 or a_id_3 == a_id_1:
                    elapsed_times[current_operation] = time.time() - t_start
                    pbar.update(1)
                    continue

                a_id_1_str = str(int(asteroid_list[i]['ID']))
                a_id_2_str = str(int(asteroid_list[j]['ID']))
                a_id_3_str = str(int(asteroid_list[k]['ID']))

                asteroid_optimized_data[i][j][k] = optimize_times(
                    a_id_1_str, a_id_2_str, a_id_3_str,
                    launch_dates, m_1, m_2, m_3)

                elapsed_times[current_operation] = time.time() - t_start
                pbar.update(1)

    pbar.close()

    os.makedirs('./optimal_asteroid_paths', exist_ok=True)
    save_path = f'./optimal_asteroid_paths/asteroid_data_{m_1}_{m_2}_{m_3}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(asteroid_optimized_data, f)

    return asteroid_optimized_data

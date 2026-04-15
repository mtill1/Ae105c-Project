"""Score function for greedy path optimization."""

from ..constants import YEAR
from .compute_path_deltav_greedy import compute_path_deltav_greedy


def score_paths_greedy(time_vector, launch_range, launch_body,
                       landing_body, goal_body, m_1, m_2):
    """Evaluate delta-v for a greedy path given a time parameter vector.

    Parameters
    ----------
    time_vector : array-like of length 3
        [launch_offset, transfer_1_time, transfer_2_time] in units of YEAR.
    launch_range : array-like of length 2
        [et_min, et_max] launch window in ephemeris time.
    launch_body : str
        NAIF ID of departure body.
    landing_body : str
        NAIF ID of flyby body, or "-1" for direct transfer.
    goal_body : str
        NAIF ID of destination body.
    m_1 : float
        Lambert revolution parameter for the first leg.
    m_2 : float
        Lambert revolution parameter for the second leg.

    Returns
    -------
    dv_total : float
        Total delta-v in km/s.
    """
    launch_date = time_vector[0] * YEAR + launch_range[0]
    arrival_date = time_vector[1] * YEAR + launch_date
    goal_date = time_vector[2] * YEAR + arrival_date

    _, _, _, dv_total, _, _, _, _ = compute_path_deltav_greedy(
        launch_body, landing_body, goal_body,
        launch_date, arrival_date, goal_date, m_1, m_2)

    return dv_total

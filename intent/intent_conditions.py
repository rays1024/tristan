import json
from collections import OrderedDict
from typing import Any, Tuple

import cv2
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from shapely.geometry import LinearRing, LineString, Point, Polygon, MultiPolygon
import math
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

import intent.IntentOSM as intent_osm

osm = intent_osm.IntentOSM()

try:
    osm.load_all()
except Exception as e:
    osm = None
    print("No OSM cache:\n{}".format(repr(e)))


class TokenEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.number):
            return o.item()
        else:
            return json.JSONEncoder.default(self, o)


def dist_sqr(pos1, pos2):
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]

    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def is_condition_log_id(res, past_horizon=23, future_horizon=23, tlog_hash=None, time_range=None):
    time_range = None
    result = True
    if not (tlog_hash is None):
        result = res["filehash"] == tlog_hash
        # if (result):

    if not (time_range is None):
        timestamp = res["timestamp"][len(res["past_positions"])]
        result = result and timestamp > time_range[0] and timestamp <= time_range[1]
    return result


def is_condition_tested_worst_cases(obj, statistics_json, slice_names, group_type="nll_highest"):
    """
    Compares the json object's fname and timestamp to statistics_json, getting the worst examples according to
    group type, for the slice names (e.g, nll_highest for all slices)
    :param obj:
    :param statistics_json:
    :param slice_names:
    :param group_type:
    :return:
    """
    result = False
    assert "slice_results" in statistics_json
    slices = statistics_json["slice_results"]
    if slice_names is None or not slice_names:
        slice_names = list(slices.keys())

    assert "fname_meta" in obj
    assert "timestamp" in obj
    fname = obj["fname_meta"]
    timestamp = obj["timestamp"]
    for slice_name in slice_names:
        slice_result = slices[slice_name]

        if group_type in slice_result:

            group = slice_result[group_type]
            for itm in group[1]:
                if itm["fname"] == fname and itm["timestamp"] == timestamp:
                    return True

    return result


### Functors to identify driving conditions
def is_condition_stop_f(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the future
    future_traj = np.array(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    dist_future_sqr = dist_sqr(future_traj[0, :], future_traj[-1, :])

    return dist_future_sqr < 4.0


def is_condition_stop_p(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the past
    past_traj = np.array(res["past_positions"])
    past_traj = past_traj[-past_horizon:]
    dist_past_sqr = dist_sqr(past_traj[0, :], past_traj[-1, :])

    return dist_past_sqr < 4.0


def is_condition_stop_m(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the middle of a motion
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    future_traj_sqr_diff = [dist_sqr(future_traj[i], future_traj[i + 1]) for i in np.arange(future_horizon - 1)]
    # print("future_traj_sqr_diff")
    # print(future_traj_sqr_diff)
    return np.count_nonzero(future_traj_sqr_diff) < future_horizon * 4 / 5


def is_condition_stop(res, past_horizon=23, future_horizon=23):
    # Check for complete and partial stops
    return (
        is_condition_stop_f(res, past_horizon, future_horizon)
        or is_condition_stop_p(res, past_horizon, future_horizon)
        or is_condition_stop_m(res, past_horizon, future_horizon)
    )


def create_maneuver_condition(maneuver_name, adovehicle_data_dir):
    def __is_good_maneuver(res):
        from attic.data.adovehicle_database import AdovehicleDatabase

        adovehicle_db = AdovehicleDatabase(adovehicle_data_dir)
        try:
            from attic.adovehicle import imerit

            annotation = adovehicle_db.get_imerit_annotation(res["uuid"])
            return imerit.is_good_maneuver(annotation, maneuver_name)
        except KeyError:
            import traceback

            print("Annotation is not found.")
            traceback.print_exc()
            return False

    return __is_good_maneuver


def is_condition_nonstop(res, past_horizon=23, future_horizon=23):
    # Check for movements
    return not is_condition_stop(res, past_horizon, future_horizon)


def is_condition_forward(res, past_horizon=23, future_horizon=23):
    # Check for nonbackward movements

    # Make sure the vehicle moves
    # if is_condition_stop_f(res, past_horizon=23, future_horizon=23): return False
    if is_condition_stop_m(res, past_horizon=23, future_horizon=23):
        return False

    # check if vehicle moves forwards
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    past_traj = np.transpose(res["past_positions"])
    past_traj = past_traj[-past_horizon:]

    x_last = future_traj[-1][0]
    y_last = future_traj[-1][1]
    # Why 5?
    # x_first = past_traj[-5][0]
    # y_first = past_traj[-5][1]
    # Change to last element, for now
    x_first = past_traj[-1][0]
    y_first = past_traj[-1][1]

    # make sure vehicle does not go too fast
    if x_last > 400:
        return False

    # make sure the vehicle goes forward in x for at least 0.1 meters
    # and moves for at least 0.5 meters
    return x_first <= 0 and x_last > 0.1 and x_last**2 + y_last**2 > 0.25


def is_condition_forward_and_slow(res, past_horizon=23, future_horizon=23):
    # check if velocity is reasonbale
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    vel = res["velocity"]
    vel_norm = np.linalg.norm(vel)

    return vel_norm <= 10.0


def is_condition_straight(res, past_horizon=23, future_horizon=23):
    # Check for straight movements

    # Make sure the vehicle moves forward
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    # Check if the future trajectory is straight
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    # check linearity of the trajectory
    slope, intercept, r_value, p_value, std_err = stats.linregress(future_traj[:, 0], future_traj[:, 1])
    return np.abs(r_value**2) > 0.99


def is_condition_city(res, past_horizon=23, future_horizon=23):
    # Check for city samples

    # Make sure the vehicle moves forward
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    non_city_filehashes = ["c04cc49555b774a93bfeb487228bb7f81418b5f6"]
    return not res["filehash"] in non_city_filehashes


def is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
    # Check if the sample is close enough to the intersection

    # Make sure the data is collected in city
    if not is_condition_city(res, past_horizon=23, future_horizon=23):
        return False

    latlon = res["position_lla"]
    # middle = len(latlon) // 2
    # latlon = latlon[middle-1]
    dist_now = osm.dist_to_intersection(latlon)

    return np.abs(dist_now) <= 5

    # latlon_future = latlon[middle+5]
    # dist_future = osm.dist_to_intersection(latlon_future)

    # return dist_now > dist_future


def is_condition_turning(res, past_horizon=23, future_horizon=23):
    # Check if the vehicle is turning left or right

    # Make sure the data is collected in city
    if not is_condition_city(res, past_horizon=23, future_horizon=23):
        return False
    if not is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
        return False

    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    # check linearity of the trajectory
    slope, intercept, r_value, p_value, std_err = stats.linregress(future_traj[:, 0], future_traj[:, 1])

    y_last = future_traj[-1][1]
    x_last = future_traj[-1][0]
    return np.abs(y_last) > 3.0 and np.abs(x_last) < 30


def is_condition_stop_and_turn(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the past and turns in the future

    # Make sure the data is collected in city
    # if not is_condition_city(res, past_horizon=23, future_horizon=23):
    #     return False
    # if not is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
    #     return False
    non_city_filehashes = ["c04cc49555b774a93bfeb487228bb7f81418b5f6"]
    if "filehash" in res and res["filehash"] in non_city_filehashes:
        return False

    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    past_traj = np.transpose(res["past_positions"])
    past_traj = past_traj[-past_horizon:]

    x_last = future_traj[-1][0]
    y_last = future_traj[-1][1]
    x_first = past_traj[-5][0]
    y_first = past_traj[-5][1]

    return x_first <= 0 and x_first**2 + y_first**2 <= 0.01 and x_last > 0 and x_last**2 + y_last**2 > 0.25


def is_condition_all(res, past_horizon=23, future_horizon=23):
    # Dummy condition checker always returns true
    return True


def predicate_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filters the given agent trajectories

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t). The first value of the dictionary
        is the main car for the filter. Some filters may require more than one car as they are binary relations.
        For now, we assume fixed time-step samples, and the answer relates to the center of the timeline
        (e.g. is the car stopping at the central timepoint?)

    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
        map_elements: dict,
            A map from element id to a dictionary with fields:
              'type':string,
              'positions':np.array Mx2 of points for that map element. currently just center lines.

    params: dict
        The parameter dictionary for the filter,

    Returns
    -------
    out: list
        A list of dictionaries indicating when the predicate is true and the interacting agents.
    """
    return None


def create_duration_label(name: str, start_time: float, end_time: float, subject: str = None):
    """Create the output dictionary given the label and duration for the label.

    Parameters
    ----------
    name: str
        The name of the label being true.
    start_time: float
        The start timestamp of the label.
    end_time: flaot
        The end time stamp of the label.
    subject: str
        The subject of the label, e.g. the agent being yield to or the agent being followed.

    Returns
    -------
    out: dict
        The output dictionary for a filter.
    """
    out = {"label": name, "start_time": start_time/10, "end_time": end_time/10}
    if subject is not None:
        out["subject"] = subject
    return out

def interpolate_invalid_states(agent_traj):
    traj_copy = agent_traj.copy()

    invalid_mask = (traj_copy[:,0] == -1) & (traj_copy[:,1] == -1)

    for i in range(1, len(traj_copy)-1):
        if invalid_mask[i]:
            if not invalid_mask[i-1] and not invalid_mask[i+1]:
                x_prev, y_prev = traj_copy[i-1, 0], traj_copy[i-1, 1]
                x_next, y_next = traj_copy[i+1, 0], traj_copy[i+1, 1]
                traj_copy[i, 0] = (x_prev + x_next) / 2.0
                traj_copy[i, 1] = (y_prev + y_next) / 2.0
                invalid_mask[i] = False  # Now it's valid

    return traj_copy

def turn_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent is turning left or right.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t).
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        turn_threshold: float
            The threshold to determine if it is a turn.
        window_size: int
            The number of time steps to consider before and after the current time step to estimate the turns.
        window_std: float
            A point is too noisy and we don't estimate the turn if the standard deviation of angular changes in a window
            is larger than this number.

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for the turns.
    """
    turn_threshold = params["turn_threshold"]
    window_size = params["window_size"]
    window_std = params["window_std"]
    agent_traj = list(agent_trajectories.values())[0]
    # Compute heading.
    theta = np.arctan2(agent_traj[1:, 1] - agent_traj[:-1, 1], agent_traj[1:, 0] - agent_traj[:-1, 0])
    theta = np.hstack((theta[:1], theta))
    theta = np.degrees(theta)
    # Compute angular velocity.
    theta_diff = theta[1:] - theta[:-1]
    theta_diff = np.hstack((theta_diff, theta_diff[-1:]))
    theta_diff[theta_diff < -180] += 360
    theta_diff[theta_diff > 180] -= 360
    # Compute turns for all time steps.
    tokens = []
    current_token = None
    timestamps = []
    for t in range(theta_diff.shape[0]):
        t_start = t - window_size if t - window_size > 0 else 0
        t_end = t + window_size if t + window_size < theta.shape[0] else theta.shape[0]
        theta_diff_before = theta_diff[t_start:t]
        theta_diff_after = theta_diff[t:t_end]
        if theta_diff_before.shape[0] < window_size or theta_diff_after.shape[0] < window_size:
            continue
        mean_before = theta_diff_before.mean()
        std_bfore = theta_diff_before.std()
        mean_after = theta_diff_after.mean()
        std_after = theta_diff_after.std()
        if std_bfore > window_std or std_after > window_std:
            # Uncertain if the angular velocity changes too much.
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
        elif mean_before >= 1.0 * turn_threshold and mean_after >= 1.0 * turn_threshold:
            # Turning left if keep moving toward left.
            if current_token != "TurnLeft" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "TurnLeft"
            timestamps.append(agent_traj[t][2].item())
        elif mean_before <= -1.0 * turn_threshold and mean_after <= -1.0 * turn_threshold:
            # Turning left if keep moving toward right.
            if current_token != "TurnRight" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "TurnRight"

            timestamps.append(agent_traj[t][2].item())
        else:
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def velocity_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent is moving fast/slow or stopping.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        vel_threshold: tuple
            The thresholds (fast_min, slow_min) to determine if the agent moves fast/slow.

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for velocity mode.
    """

    vel_threshold = params["vel_threshold"]
    agent_traj = list(agent_trajectories.values())[0]
    agent_traj = interpolate_invalid_states(agent_traj)
    
    # Compute velocity.
    velocity = []
    for t in range(len(agent_traj)):
        x, y = agent_traj[t, 0], agent_traj[t, 1]
        if t == 0:
            # For the first timestep, we can't compute a velocity yet, just put a placeholder (0 or NaN)
            if x == -1 and y == -1:
                velocity.append(np.nan)  # invalid
            else:
                velocity.append(0.0)  # no previous frame
        else:
            x_prev, y_prev = agent_traj[t-1, 0], agent_traj[t-1, 1]
            if (x == -1 and y == -1) or (x_prev == -1 and y_prev == -1):
                velocity.append(np.nan)  # invalid due to invalid data
            else:
                dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                velocity.append(dist)
    velocity = np.array(velocity)

    tokens = []
    current_token = None
    timestamps = []
    for t in range(velocity.shape[0]):
        if np.isnan(velocity[t]):
            continue
        if velocity[t] >= vel_threshold[0]:
            if current_token != "MoveFast" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "MoveFast"
        elif velocity[t] >= vel_threshold[1]:
            if current_token != "MoveSlow" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "MoveSlow"
        elif velocity[t] < vel_threshold[1]:
            if current_token != "Stop" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "Stop"
        timestamps.append(agent_traj[t][2].item())
    if current_token is not None and len(timestamps) >= 3:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def acceleration_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:

    def smooth_signal(signal, window_size=5):
        if window_size == 0:
            return signal
        smoothed = np.copy(signal)
        n = len(signal)
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            if start >= end or np.all(np.isnan(signal[start:end])):
                smoothed[i] = np.nan
            else:
                smoothed[i] = np.nanmean(signal[start:end])        
        return smoothed

    acc_threshold = params.get("acc_threshold", 0.02)      # Threshold for acceleration/deceleration
    smoothing_window = params.get("smoothing_window", 5) # Half-window for smoothing

    agent_traj = list(agent_trajectories.values())[0]
    agent_traj = interpolate_invalid_states(agent_traj)
    
    velocity = []
    for t in range(len(agent_traj)):
        x, y = agent_traj[t, 0], agent_traj[t, 1]
        if t == 0:
            if x == -1 and y == -1:
                velocity.append(np.nan)
            else:
                velocity.append(0.0)
        else:
            x_prev, y_prev = agent_traj[t-1, 0], agent_traj[t-1, 1]
            if (x == -1 and y == -1) or (x_prev == -1 and y_prev == -1):
                velocity.append(np.nan)
            else:
                dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                velocity.append(dist)
    velocity = np.array(velocity)

    # Compute acceleration
    acceleration = []
    for t in range(len(velocity)):
        if t == 0:
            acceleration.append(np.nan)
        else:
            v_curr = velocity[t]
            v_prev = velocity[t-1]
            if np.isnan(v_curr) or np.isnan(v_prev):
                acceleration.append(np.nan)
            else:
                acceleration.append(v_curr - v_prev)
    acceleration = np.array(acceleration)

    # Smooth the acceleration to reduce jitter.
    acceleration = smooth_signal(acceleration, smoothing_window)

    labels = []
    for a in acceleration:
        if np.isnan(a):
            labels.append("ConstantSpeed")
        elif a > acc_threshold:
            labels.append("SpeedUp")
        elif a < -acc_threshold:
            labels.append("SlowDown")
        else:
            labels.append("ConstantSpeed")

    intervals = []
    current_label = labels[0]
    start_idx = 0

    for i, value in enumerate(labels):
        if i > 0 and i < len(labels)-1 and value != labels[i-1] and value != labels[i+1]:
            labels[i] = labels[i-1]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # Found a boundary
            intervals.append((current_label, start_idx, i-1))
            current_label = labels[i]
            start_idx = i
    intervals.append((current_label, start_idx, len(labels)-1))

    # Convert intervals to tokens
    tokens = []
    for (label, s, e) in intervals:
        start_time = agent_traj[s][2].item()
        end_time = agent_traj[e][2].item()
        tokens.append(create_duration_label(label, start_time, end_time))

    return tokens


def get_normal_and_tangential_distance_point(
    point: Point, centerline_ls: LineString, delta: float = 0.01, last: bool = False
) -> Tuple[float, float]:
    """Get normal (offset from centerline) and tangential (distance along centerline) for the given point,
    along the given centerline

    Parameters
    ----------
    point: Point
        A point (x,y)-coordinate in map frame
    centerline: LineString
        centerline along which n-t is to be computed
    delta: float
        Used in computing offset direction
    last: bool
        True if point is the last coordinate of the trajectory

    Returns
    -------
    (tang_dist, norm_dist): Tuple[float, float]
        tangential and normal distances
    """
    tang_dist = centerline_ls.project(point)
    norm_dist = point.distance(centerline_ls)
    point_on_cl = centerline_ls.interpolate(tang_dist)

    # Deal with last coordinate differently. Helped in dealing with floating point precision errors.
    if not last:
        pt1 = point_on_cl.coords[0]
        pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
        pt3 = point.coords[0]

    else:
        pt1 = centerline_ls.interpolate(tang_dist - delta).coords[0]
        pt2 = point_on_cl.coords[0]
        pt3 = point.coords[0]

    lr_coords = []
    lr_coords.extend([pt1, pt2, pt3])
    lr = LinearRing(lr_coords)

    # Left positive, right negative
    if lr.is_ccw:
        return (tang_dist, norm_dist)
    return (tang_dist, -norm_dist)


def lane_change_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the agent switches lanes.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
        map_elements: dict
            The dictionary of map info. Contains
            'lane_centers': list of lane center lines for the lanes in the map.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        lane_threshold: float
            The threshold to consider as closest lane. If the distances of an agent to a lane compared to another lane
            is within this threshold, we consider both lanes as the closest lane.
        skip_threshold: float
            If the distance to the closet lane exceeds this threshold, we don't consider it as a valid lane change,
            It is likely that the lane center ends earlier.

    Returns
    -------
    tokens: list
        A list of dictionaries that include the label and timestamps indicating the duration for lane changes.
    """
    lane_threshold = params["lane_threshold"]
    skip_threshold = params["skip_threshold"]
    window_size = params["window_size"]
    agent_traj = list(agent_trajectories.values())[0]
    traj_length = len(agent_traj)
    # Check for lane change maneuvers based on distance to centerlines.
    # First obtain all candidate centerlines.
    map_elements = scene_information["map_elements"]
    candidate_centerlines = map_elements["lane_centers"]
    # If there is only one lane, we cannot say anything about lane changes.
    if len(candidate_centerlines) <= 1:
        return []
    ls_centerlines = [LineString(centerlines) for centerlines in candidate_centerlines]
    pts = [Point(agent_traj[t][0], agent_traj[t][1]) for t in range(traj_length)]

    def get_closest_lane(pos: Point, threshold: float = lane_threshold):
        """Get the closest lanes from a position.
        Parameters
        ----------
        pos: Point
            The point position.
        threshold: float
            The threshold to consider the closest lane.

        Returns
        -------
        set
            The set of closest lanes.
        dist
            The distance to the closest lanes.
        """
        lane_idx = []
        d = 100000
        for i, lane in enumerate(ls_centerlines):
            dist = pos.distance(lane)
            # Only consider the lane as closest if it is the first lane encountered
            # or it  is within the distance threshold.
            if len(lane_idx) == 0 or dist < threshold:
                lane_idx.append(i)
                if dist < d:
                    d = dist
            elif dist < d:
                lane_idx = [i]
                d = dist
        return set(lane_idx), d

    def num_intersect_lanes(lanes: LineString):
        """Compute number of intersections given a list of lanes."""
        n_intersects = 0
        for lane_i in lanes:
            for lane_j in lanes:
                if lane_i == lane_j:
                    continue
                n_intersects += int(ls_centerlines[lane_i].intersects(ls_centerlines[lane_j]))
        return n_intersects / 2

    tokens = []
    current_token = None
    timestamps = []
    closest_lanes_d = [get_closest_lane(pts[t]) for t in range(traj_length)]
    closest_lanes = [l[0] for l in closest_lanes_d]
    ds = [l[1] for l in closest_lanes_d]
    for t in range(1, traj_length):
        # Skip the time points if there are many intersecting points, likely an intersection or lane merging.
        pt = Point(agent_traj[t, :2]).buffer(params["intersection_radius"])
        lanes_around = [lane_idx for lane_idx, lane in enumerate(ls_centerlines) if pt.intersects(lane)]
        if num_intersect_lanes(lanes_around) > params["max_intersection"]:
            continue
        t_start = t - window_size if t - window_size > 0 else 0
        t_end = t + window_size if t + window_size < traj_length else traj_length
        lanes_before = set.union(*(closest_lanes[t_start:t]))
        lanes_after = set.union(*(closest_lanes[t:t_end]))
        # Skip the time points if the agent passing through many centerlines.
        if (
            len(lanes_before) > params["max_lanes_in_window"]
            or len(lanes_after) > params["max_lanes_in_window"]
            or len(lanes_before) == 0
            or len(lanes_after) == 0
        ):
            continue
        if ds[t] > skip_threshold:
            # Cannot compute lane change if the closest lane is too far.
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
                current_token = None
        elif len(lanes_after.intersection(lanes_before)) > 0:
            # Lane keep if closest lane does not change.
            if current_token != "LaneKeep" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "LaneKeep"
            timestamps.append(agent_traj[t][2].item())
        else:
            # Compute signed normal distance of current position to the lane
            # closest to the previous positions, and decide lane change
            # direction based on the difference in distances.
            _, dist_curr_to_prev = get_normal_and_tangential_distance_point(pts[t], ls_centerlines[lanes_before.pop()])
            _, dist_curr = get_normal_and_tangential_distance_point(pts[t], ls_centerlines[lanes_after.pop()])
            if dist_curr_to_prev > dist_curr:
                if current_token != "LaneChangeLeft" and current_token is not None:
                    tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                    timestamps = []
                current_token = "LaneChangeLeft"
            else:
                if current_token != "LaneChangeRight" and current_token is not None:
                    tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                    timestamps = []
                current_token = "LaneChangeRight"
            timestamps.append(agent_traj[t][2].item())
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def follow_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict):
    """Filter if the agent follows other agents within a specific distance.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t).
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys:
        - min_overlaps: int
            The minimum overlapping time steps for computing 'follow'.
        - max_following_distance: float
            The maximum distance within which an agent is considered to be "following" another.

    Returns
    -------
    token: list
        A list of dictionaries indicating the duration when "follow" is true and which agent is followed.
    """
    min_overlaps = params["min_overlaps"]
    max_following_time = params["max_following_time"]
    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    target_traj_length = target_agent_traj.shape[0]
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    target_ls = LineString(target_agent_traj[:, :2]).buffer(0.5)
    tokens = []

    for other_agent_id in other_agent_ids:
        # Compute the overlapping area of two trajectories.
        other_agent_traj = agent_trajectories[other_agent_id]
        other_ls = LineString(other_agent_traj[:, :2]).buffer(0.5)
        overlaps = target_ls.intersection(other_ls)
        if overlaps.area == 0:
            continue

        # Get the points in the overlapping area.
        target_overlap = [
            target_agent_traj[t]
            for t in range(target_traj_length)
            if overlaps.contains(Point(target_agent_traj[t, :2]))
        ]
        other_overlap = [
            other_agent_traj[t]
            for t in range(other_agent_traj.shape[0])
            if overlaps.contains(Point(other_agent_traj[t, :2]))
        ]

        # Check for following conditions: overlap length, arrival time, and distance constraint
        if (
            len(target_overlap) > min_overlaps
            and len(other_overlap) > min_overlaps
            and other_overlap[0][2] < target_overlap[0][2]
        ):
            # Verify if the following distance constraint is met throughout the overlap period
            following = True
            for t in range(min(len(target_overlap), len(other_overlap))): 
                if target_overlap[t][0] < 0 or target_overlap[t][1] < 0 or other_overlap[t][0] < 0 or other_overlap[t][1] < 0:
                    break
                if target_overlap[t][2] - other_overlap[t][2] > max_following_time:
                    following = False
                    break
            
            if following and t > 1:
                tokens.append(
                    create_duration_label("Follow", target_overlap[0][2], target_overlap[t][2], subject=other_agent_id)
                )

    return tokens


def find_points_in_region(positions: np.ndarray, region: LineString):
    """Find the points in a region.

    Parameters
    ----------
    positions: np.ndarray
        The positions of a trajectory.
    region: LineString
        The line region.
    """
    if region.is_empty:
        return []
    out = []
    for i, pt in enumerate(positions):
        if region.contains(Point(pt)):
            out.append(i)
    return out


def yield_filter(agent_trajectories: OrderedDict, scene_information: OrderedDict, params: dict) -> list:
    """If agent1 and agent2 start on different paths, end up in the same path, and agent2 gets first to the joint part of the path.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        Map from agent id to the trajectory of the agents. The first agent is the agent that may/may not yield.
        The later agents are all the agent possibly yielded to.
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        Parameters:
        yielding_prefix - minimal threshold to verify that the trajectories are long enough
        yielding_dt - delta t (timepoints) used to estimate speed change
        yielding_time_gap - the time gap between agent 1 and agent 2 reaching the intersection point

    Returns
    -------
    token_list: list
        A list of tokens detected ("yield").
    """
    if len(list(agent_trajectories.values())) < 2:
        return []

    def compute_trajectory_stats(traj, dilation):
        """Compute trajectory stats needed for yielding"""
        positions = traj[:, :2]
        v = positions[1:, :] - positions[:-1, :]
        v = np.concatenate((v, v[-2:-1, :]), axis=0)
        spd1 = (v**2).sum(1)
        t = traj[:, 2]
        sl = LineString(positions.tolist()).buffer(dilation)
        return positions, spd1, t, sl

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    mask = (target_agent_traj[:, 0] >= 0) & (target_agent_traj[:, 1] >= 0)
    target_agent_traj = target_agent_traj[mask]

    yielding_dilation_radius = params["yielding_dilation_radius"]
    yielding_max_time = params["yielding_max_time"]

    positions1, spd1, t1, sl1 = compute_trajectory_stats(target_agent_traj, yielding_dilation_radius)

    tokens = []
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        mask = (other_agent_traj[:, 0] >= 0) & (other_agent_traj[:, 1] >= 0)
        other_agent_traj = other_agent_traj[mask]
        if len(other_agent_traj) < 2:
            continue
        positions2, spd2, t2, sl2 = compute_trajectory_stats(other_agent_traj, yielding_dilation_radius)
        intersection = sl1.intersection(sl2)

        target_overlap = [
            target_agent_traj[t]
            for t in range(target_agent_traj.shape[0])
            if intersection.contains(Point(target_agent_traj[t, :2]))
        ]
        other_overlap = [
            other_agent_traj[t]
            for t in range(other_agent_traj.shape[0])
            if intersection.contains(Point(other_agent_traj[t, :2]))
        ]
        near = True
        for t in range(min(len(target_overlap), len(other_overlap))): 
            if target_overlap[t][2] - other_overlap[t][2] > yielding_max_time:
                near = False
                break
        if not near:
            continue

        idxs1 = find_points_in_region(positions1, intersection)
        idxs2 = find_points_in_region(positions2, intersection)
        if len(idxs1) == 0 or len(idxs2) == 0:
            continue

        # The trajectories upto the intersection.
        positions1b = positions1[: idxs1[0], :]
        positions2b = positions2[: idxs2[0], :]
        if positions1b.shape[0] < 2 or positions2b.shape[0] < 2:
            continue

        sl1b = LineString(positions1b[:, :2]).buffer(yielding_dilation_radius)
        sl2b = LineString(positions2b[:, :2]).buffer(yielding_dilation_radius)

        # The regions that cross each other
        sl1_only = (sl1b.difference(sl2)).buffer(-yielding_dilation_radius / 2.0)
        sl2_only = (sl2b.difference(sl1)).buffer(-yielding_dilation_radius / 2.0)
        # Skip if no overlaps in the trajectory.
        if len(idxs1) == 0 or len(idxs2) == 0:
            continue
        idx1_m1 = max(0, idxs1[0] - params["yielding_dt"])
        if (
            sl1_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 1
            and sl2_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 2
            and t1[idxs1[0]] - t2[idxs2[0]] > params["yielding_time_gap"]  # agent 1 is before agent 2
            and spd1[idx1_m1] - spd1[idxs1[0]] > -0.5  # non-increasing speed.
            and Point(positions2b[0, :]).distance(sl1_only)
            > params["yielding_initial_distance"]  # initial point for 2 is far from trajectory 1
        ):
            tokens.append(create_duration_label("Yield", t2[idxs2[0]], t2[idxs2[-1]], subject=other_agent_id))
    return tokens


def proximity_filter(agent_trajectories: OrderedDict, scene_information: OrderedDict, params: dict) -> list:
    """If agent1 is too close to another agent, return a token.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        Map from agent id to the trajectory of the agents. The first agent is the agent that may/may not yield.
        The later agents are all the agent possibly yielded to.
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        Parameters:
        proximity_threshold - maximal distance between agent1,agent2

    Returns
    -------
    token_list: list
        A list of tokens detected ("proximity").
    """

    if len(list(agent_trajectories.values())) < 2:
        return []

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    proximity_threshold = params["proximity_threshold"]

    # positions1, spd1, t1, sl1 = compute_trajectory_stats(target_agent_traj, yielding_dilation_radius)

    tokens = []
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        # Interpolate the other agent's positions to the same timepoints as the target agent
        x_interp = interp1d(other_agent_traj[:, 2], other_agent_traj[:, 0], fill_value="extrapolate")
        y_interp = interp1d(other_agent_traj[:, 2], other_agent_traj[:, 1], fill_value="extrapolate")
        other_agent_traj_registered = np.transpose(
            np.stack([x_interp(target_agent_traj[:, 2]), y_interp(target_agent_traj[:, 2])])
        )
        # Compute distance between the 2 agents.
        agents_dist = np.sum((other_agent_traj_registered - target_agent_traj[:, :2]) ** 2, 1)
        min_idx = np.argmin(agents_dist)
        min_val = np.min(agents_dist)
        if min_val < proximity_threshold**2:
            tokens.append(
                create_duration_label(
                    "proximity",
                    target_agent_traj[min_idx, 2],
                    target_agent_traj[min_idx, 2],
                    subject=other_agent_id,
                )
            )
    return tokens


def ttc_filter(agent_trajectories: OrderedDict, scene_information: OrderedDict, params: dict) -> list:
    """If the Time to Collision (TTC) is within the threshold, return a token

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        Parameters:
        ttc_threshold - Maximum number of seconds

    Returns
    -------
    token_list: list
        A list of tokens detected ("ttc").
    """
    if len(list(agent_trajectories.values())) < 2:
        return []

    ttc_threshold_squared = params["ttc_threshold"] ** 2

    def compute_trajectory_stats(traj):
        """Compute trajectory stats needed for yielding"""
        positions = traj[:, :2]
        velocities = positions[1:, :] - positions[:-1, :]
        velocities = np.concatenate((velocities, velocities[-2:-1, :]), axis=0)
        slopes = velocities[:, 1] / velocities[:, 0]
        return positions, velocities, slopes

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    target_positions, target_velocities, target_slopes = compute_trajectory_stats(target_agent_traj)
    target_time_points = target_agent_traj[:, 2]

    tokens = []
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        x_interp = interp1d(other_agent_traj[:, 2], other_agent_traj[:, 0], fill_value="extrapolate")
        y_interp = interp1d(other_agent_traj[:, 2], other_agent_traj[:, 1], fill_value="extrapolate")
        other_agent_traj_registered = np.transpose(
            np.stack([x_interp(target_agent_traj[:, 2]), y_interp(target_agent_traj[:, 2])])
        )

        other_positions, other_velocities, other_slopes = compute_trajectory_stats(other_agent_traj_registered)

        relative_distances_squared = ((target_positions - other_positions) ** 2).sum(axis=-1)
        relative_velocities = target_velocities - other_velocities
        relative_velocities_squared = (relative_velocities**2).sum(axis=-1)
        ttc = relative_distances_squared / relative_velocities_squared
        relevant_time_idxs = np.nonzero(ttc < ttc_threshold_squared)[0]
        if len(relevant_time_idxs):
            start_time = target_time_points[relevant_time_idxs[0]].item()
            end_time = target_time_points[relevant_time_idxs[-1]].item()

            tokens.append(
                create_duration_label("TTC", start_time=start_time, end_time=end_time, subject=other_agent_id)
            )

    return tokens

def road_curvature_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """
    Filter if the road is curving and in which direction (left or right).

    Parameters
    ----------
    agent_trajectories : OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x, y, t).
    scene_information : dict
        Scene information, containing map elements such as lane center lines.
    params : dict
        Parameters for the filter, including:
        curvature_threshold: float
            Minimum angle change to consider as curvature.
        window_size: int
            Number of steps to use in calculating directional changes.
        min_segment_length: int
            Minimum number of points to consider a segment as curved.

    Returns
    -------
    tokens : list
        List of dictionaries with curvature labels and timestamps indicating curving segments.
    """
    curvature_threshold = params["curvature_threshold"]
    window_size = params["window_size"]
    min_segment_length = params["min_segment_length"]
    
    agent_traj = list(agent_trajectories.values())[0]
    traj_length = len(agent_traj)
    map_elements = scene_information["map_elements"]
    lane_centers = map_elements["lane_centers"]
    
    # Convert lane center coordinates to LineString objects
    ls_lane_centers = [LineString(lane) for lane in lane_centers]
    tokens = []
    current_token = None
    timestamps = []
    
    def get_closest_lane_point(pos: Point):
        """Get the closest point on lane centers from a position."""
        min_dist = float("inf")
        closest_point = None
        for lane in ls_lane_centers:
            point_on_lane = lane.interpolate(lane.project(pos))
            dist = pos.distance(point_on_lane)
            if dist < min_dist:
                min_dist = dist
                closest_point = point_on_lane
        return closest_point
    
    def calculate_directional_change(p1: Point, p2: Point, p3: Point) -> float:
        """Calculate signed angle change from (p1->p2) to (p2->p3) in degrees."""
        v1 = np.array([p2.x - p1.x, p2.y - p1.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
        return angle
    
    for t in range(window_size, traj_length - window_size):
        pt_ego = Point(agent_traj[t, :2])
        pt_prev = Point(agent_traj[t - window_size, :2])
        pt_next = Point(agent_traj[t + window_size, :2])
        
        closest_point = get_closest_lane_point(pt_ego)
        angle_change = calculate_directional_change(pt_prev, pt_ego, pt_next)
        
        if abs(angle_change) > curvature_threshold:
            direction = "RoadCurveLeft" if angle_change > 0 else "RoadCurveRight"
            if current_token != direction:
                if current_token is not None and len(timestamps) >= min_segment_length:
                    tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                    timestamps = []
                current_token = direction
            timestamps.append(agent_traj[t, 2].item())
        else:
            if current_token is not None and len(timestamps) >= min_segment_length:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
                current_token = None

    if current_token is not None and len(timestamps) >= min_segment_length:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    
    return tokens

def intersection_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:

    def points_to_polygons(points, eps=5):
        points = np.array(points)  # Ensure input is a NumPy array

        # Step 1: Cluster points using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=1)
        labels = dbscan.fit_predict(points)

        polygons = []

        # Step 2: Process each cluster
        for cluster_label in set(labels):
            cluster_points = points[labels == cluster_label]

            # Skip clusters with fewer than 3 points (cannot form a polygon)
            if len(cluster_points) < 3:
                continue

            # Compute the minimum area rectangle using OpenCV
            rect = cv2.minAreaRect(cluster_points.astype('float32'))
            box = cv2.boxPoints(rect)  # Extract rectangle vertices
            box = np.int0(box)         # Convert to integer coordinates

            # Create a shapely polygon from the rectangle vertices
            polygon = Polygon(box)

            # Ensure the polygon is valid (e.g., close the loop)
            if polygon.is_valid:
                polygons.append(polygon)

        return polygons

    min_segment_length = params["min_segment_length"]
    proximity_threshold = params["proximity_threshold"]
    
    # Extract ego trajectory
    agent_traj = list(agent_trajectories.values())[0]
    traj_length = len(agent_traj)
    
    map_elements = scene_information["map_elements"]
    crosswalks = map_elements.get("crosswalks", [])
    stop_signs = map_elements.get("stop_signs", [])

    if not crosswalks:
        # No crosswalks, no intersection
        return []

    flattened_crosswalk = [item for sublist in crosswalks for item in sublist]
    intersection_polygon = points_to_polygons(flattened_crosswalk)

    # Now detect when the ego is inside or outside this intersection_polygon
    tokens = []
    inside_intersection = False
    timestamps = []
    current_label = None

    if intersection_polygon:
        for t in range(traj_length):
            pt_ego = Point(agent_traj[t, :2])
            current_time = agent_traj[t, 2].item()

            # Check if ego is inside the intersection polygon
            for polygon in intersection_polygon:
                is_inside = polygon.contains(pt_ego)
                if is_inside:
                    break

            if is_inside and not inside_intersection:
                # Entering intersection
                inside_intersection = True
                current_label = "Intersection"
                timestamps = [current_time]
            elif not is_inside and inside_intersection:
                # Leaving intersection
                inside_intersection = False
                timestamps.append(current_time)
                if len(timestamps) >= min_segment_length:
                    tokens.append(create_duration_label(current_label, timestamps[0], timestamps[-1]))
                current_label = None
                timestamps = []
            elif is_inside and inside_intersection:
                # Still inside intersection
                timestamps.append(current_time)
            else:
                # Outside and staying outside
                pass

        # If ended inside intersection
        if inside_intersection and len(timestamps) >= min_segment_length:
            tokens.append(create_duration_label("Intersection", timestamps[0], timestamps[-1]))

    def is_near_stopsign(ego_pos: Point, stop_sign_positions: list, threshold: float) -> bool:
        """Check if ego_pos is within threshold distance of any stop sign."""
        for sign in stop_sign_positions:
            for s in sign:
                # Each 's' is a point (x, y) for the stop sign
                if ego_pos.distance(Point(s)) <= threshold:
                    return True
        return False

    # Detect when the ego car enters and leaves the stop sign circle
    inside_stopsign_area = False
    timestamps_stopsign = []

    for t in range(traj_length):
        pt_ego = Point(agent_traj[t, :2])
        current_time = agent_traj[t, 2].item()
        near_stopsign = is_near_stopsign(pt_ego, stop_signs, proximity_threshold)
        
        if near_stopsign and not inside_stopsign_area:
            # Just entered stop sign area
            inside_stopsign_area = True
            timestamps_stopsign = [current_time]
        elif not near_stopsign and inside_stopsign_area:
            # Just exited stop sign area
            inside_stopsign_area = False
            timestamps_stopsign.append(current_time)
            if len(timestamps_stopsign) >= min_segment_length:
                tokens.append(create_duration_label("StopSign", timestamps_stopsign[0], timestamps_stopsign[-1]))
            timestamps_stopsign = []
        elif near_stopsign and inside_stopsign_area:
            # Still inside stop sign area
            timestamps_stopsign.append(current_time)
        else:
            # Outside and remains outside
            pass

    # If ended while still inside stop sign area
    if inside_stopsign_area and len(timestamps_stopsign) >= min_segment_length:
        tokens.append(create_duration_label("StopSign", timestamps_stopsign[0], timestamps_stopsign[-1]))
    
    return tokens
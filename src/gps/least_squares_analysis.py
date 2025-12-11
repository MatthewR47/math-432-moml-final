# This file was created with reference to the gnss_lib_py documentation here:
# https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_android_notebook.html

import numpy as np
import gnss_lib_py as glp
import os
from itertools import combinations
import compute_conditional_value as ccv


FILEPATH = os.path.join("..", "data", "gnss_log.txt")
COMPARISON_EPOCH = 1383435830000.0
RAW_FIXES = glp.AndroidRawFixes(input_path=FILEPATH)


# Function to compute the distance that the calculated position is from
# the reference position
def compute_distance_error(geodetic):
    # Load the reference data for this log file

    # Get the fixed data points for the comparison epoch
    fixed_epoch = RAW_FIXES.where("gps_millis", COMPARISON_EPOCH)
    fixed_latitude = fixed_epoch["lat_rx_deg"].item()
    fixed_longitude = fixed_epoch["lon_rx_deg"].item()
    fixed_altitude = fixed_epoch["alt_rx_m"].item()

    # Reshape to (3, 1) for geodetic_to_ecef
    fixed_geodetic = np.array([[fixed_latitude], [fixed_longitude], [fixed_altitude]])
    geodetic_array = np.array([[geodetic[0]], [geodetic[1]], [geodetic[2]]])

    fixed_ecef = glp.utils.coordinates.geodetic_to_ecef(fixed_geodetic)
    ecef = glp.utils.coordinates.geodetic_to_ecef(geodetic_array)

    difference_vector = ecef - fixed_ecef

    error_distance = np.linalg.norm(difference_vector)
    return error_distance


def pull_geodetic_from_wls(wls_result):
    latitude = wls_result["lat_rx_wls_deg"].item()
    longitude = wls_result["lon_rx_wls_deg"].item()
    altitude = wls_result["alt_rx_wls_m"].item()
    return [latitude, longitude, altitude]


##########################################################################################################################
# Preprocessing Steps
##########################################################################################################################
# Load raw data into the existing glp.AndroidRawGNSS class.
# We are removing measurements that have a known high uncertainty in their time measurement.
raw_data = glp.AndroidRawGnss(
    input_path=FILEPATH,
    filter_measurements=True,
    measurement_filters={"sv_time_uncertainty": 500.0},
    verbose=True,
)

# Add satellite positions to data (this will download data describing where the satellites
# are and computes their positions in terms of x, y, and z coordinates)
full_states = glp.add_sv_states(raw_data, source="precise", verbose=False)

# This is where the sattelite clock bias is added
# We know this from ephemeris data calculations so we might as well add it.
# We will still solve for receiver clock bias later.
full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states["b_sv_m"]

# This line filters to only GPS and Galileo satellites
full_states = full_states.where("gnss_id", ("gps", "galileo"))


########################################################################################################################
# In the following section, we will compute the condition number for all combinations of 4 satellites
# in this epoch (that don't have high time uncertainty).
########################################################################################################################

# Pick Single Epoch to Analyze
single_epoch = full_states.where("gps_millis", COMPARISON_EPOCH)

# Get list of satellite IDs for this epoch and number of combinations possible
sat_ids = np.unique(single_epoch["sv_id"])
num_combinations = len(list(combinations(sat_ids, 4)))
print(f"Total combinations of 4 satellites: {num_combinations}")

# Calculate Weighted Least Squares position estimate
wls_estimate = glp.solve_wls(single_epoch)

# Prepare a receiver state for elevation/azimuth calculations
receiver_state = glp.NavData()
receiver_state["gps_millis"] = wls_estimate["gps_millis"]
receiver_state["x_rx_wls_m"] = wls_estimate["x_rx_wls_m"]
receiver_state["y_rx_wls_m"] = wls_estimate["y_rx_wls_m"]
receiver_state["z_rx_wls_m"] = wls_estimate["z_rx_wls_m"]

results = []

# Generate all possible combinations of 4 satellites
for group_num, selected_sats in enumerate(combinations(sat_ids, 4), 1):
    selected_sats = np.array(selected_sats)

    # Filter to selected satellites
    subset = single_epoch.where("sv_id", selected_sats)

    # Add elevation and azimuth data
    subset_with_el_az = glp.add_el_az(subset, receiver_state, inplace=False)

    # Add satellites to the ComputeConditionNumber object
    ccn = ccv.ComputeConditionNumber()
    for i in range(len(selected_sats)):
        az = subset_with_el_az["az_sv_deg"][i]
        el = subset_with_el_az["el_sv_deg"][i]
        ccn.add_satellite(el, az)

    # Compute condition number and DOPs
    # We need the try catch in case any of the combinations
    # aren't linearly dependent
    try:
        condition_number = ccn.compute_condition_number()
        HDOP, VDOP, PDOP, GDOP = ccn.compute_dops()

        # Calculate WLS for this group
        wls_result = glp.solve_wls(subset)
        geodetic_result = pull_geodetic_from_wls(wls_result)
        error = compute_distance_error(geodetic_result)

        # Store results
        results.append(
            {
                "group": group_num,
                "satellites": selected_sats,
                "condition_number": condition_number,
                "HDOP": HDOP,
                "VDOP": VDOP,
                "PDOP": PDOP,
                "GDOP": GDOP,
                "error": error,
            }
        )

    except Exception:
        pass

########################################################################################################################
# Analysis
########################################################################################################################

# TODO Calculate medians for condition_number, GDOP, and error


# TODO Calculate what percentage of groups that had their condition variable above the median value also had their error above the median value


# TODO Calculate what percentage of groups that had their GDOP above the median value also had their error above the median value


# TODO Print out findings

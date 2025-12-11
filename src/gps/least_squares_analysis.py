# This file was created with reference to the gnss_lib_py documentation here:
# https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_android_notebook.html

import numpy as np
import gnss_lib_py as glp
import os
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
# Analysis Steps
########################################################################################################################

# Pick Single Epoch to Analyze
single_epoch = full_states.where("gps_millis", COMPARISON_EPOCH)

# Calculate Weighted Least Squares position estimate
wls_estimate = glp.solve_wls(single_epoch)

print("WLS Position Estimate:")
print(wls_estimate)

geodetic_estimate = pull_geodetic_from_wls(wls_estimate)
error_distance = compute_distance_error(geodetic_estimate)
print("Distance Error from Reference Position (m):", error_distance)


# Use this section to pick out which satellites to compare (maybe ones that are worse according to the SVD comparison)
########################################################################################################################
# Get list of satellite IDs for this epoch
sat_ids = np.unique(single_epoch["sv_id"])

# Example 1: Use only first 4 satellites (minimum needed)
subset_4sats = single_epoch.where("sv_id", sat_ids[:4])
wls_4sats = glp.solve_wls(subset_4sats)
print("\nWLS with 4 satellites:")
print(wls_4sats)

# Example 2: Use first 6 satellites
subset_6sats = single_epoch.where("sv_id", sat_ids[:6])
wls_6sats = glp.solve_wls(subset_6sats)
print("\nWLS with 6 satellites:")
print(wls_6sats)

# Example 3: Manually pick specific satellites
chosen_sats = [sat_ids[0], sat_ids[2], sat_ids[5], sat_ids[8]]  # pick any 4+
subset_custom = single_epoch.where("sv_id", chosen_sats)
wls_custom = glp.solve_wls(subset_custom)
print("\nWLS with custom satellite selection:")
print(wls_custom)

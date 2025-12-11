# This file was created with reference to the gnss_lib_py documentation here:
# https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_android_notebook.html

import numpy as np
import gnss_lib_py as glp
import os

FILEPATH = os.path.join("..", "data", "gnss_log.txt")

# Load raw data into the existing glp.AndroidRawGNSS class.
# We are removing measurements that have a known high uncertainty in their time measurement.
raw_data = glp.AndroidRawGnss(
    input_path=FILEPATH,
    filter_measurements=True,
    measurement_filters={"sv_time_uncertainty": 500.0},
    verbose=True,
)

# Load the reference data for this log file
raw_fixes = glp.AndroidRawFixes(input_path=FILEPATH)

# Add satellite positions to data (this will download data describing where the satellites
# are and computes their positions in terms of x, y, and z coordinates)
full_states = glp.add_sv_states(raw_data, source="precise", verbose=False)

# This is where the sattelite clock bias is added
# We know this from ephemeris data calculations so we might as well add it.
# We will still solve for receiver clock bias later.
full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states["b_sv_m"]

# This line filters to only GPS and Galileo satellites
full_states = full_states.where("gnss_id", ("gps", "galileo"))

# Pick Second Epoch for more analysis
single_epoch = full_states.where("gps_millis", 1383435830000.0)

print(f"Satellite IDs: {np.unique(single_epoch['sv_id'])}")

# Calculate Weighted Least Squares position estimate
wls_estimate = glp.solve_wls(single_epoch)
print("WLS Position Estimate (ECEF):", wls_estimate)

print("\n")


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
########################################################################################################################

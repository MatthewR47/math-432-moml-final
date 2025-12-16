# This file was created with reference to the gnss_lib_py documentation here:
# https://gnss-lib-py.readthedocs.io/en/stable/tutorials/parsers/tutorials_android_notebook.html
import numpy as np
import gnss_lib_py as glp
import os
from itertools import combinations
import compute_conditional_value as ccv
import matplotlib.pyplot as plt

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


# Makes a plot showing where the satellites are when looking from above
def plot_skyplot(az_el_list, title="Satellite Skyplot", sat_ids=None):
    az = np.radians([az for az, el in az_el_list])
    r = [90 - el for az, el in az_el_list]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(az, r, s=80)
    ax.set_ylim(90, 0)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


######################################################################################################################
# Preprocessing Steps
######################################################################################################################

# Load raw data into the existing glp.AndroidRawGNSS class.
# We are removing measurements that have a known high uncertainty in their time measurement.
raw_data = glp.AndroidRawGnss(
    input_path=FILEPATH,
    filter_measurements=True,
    measurement_filters={"sv_time_uncertainty": 500.0},
    verbose=False,
)

# Add satellite positions to data (this will download data describing where the satellites
# are and computes their positions in terms of x, y, and z coordinates)
full_states = glp.add_sv_states(raw_data, source="precise", verbose=False)

# This is where the satellite clock bias is added
# We know this from ephemeris data calculations so we might as well add it.
# We will still solve for receiver clock bias later.
full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states["b_sv_m"]

# This line filters to only GPS and Galileo satellites
full_states = full_states.where("gnss_id", ("gps"))

######################################################################################################################
# In the following section, we will compute the condition number for all combinations of 4 satellites
# in this epoch (that don't have high time uncertainty).
######################################################################################################################

# Pick Single Epoch to Analyze
single_epoch = full_states.where("gps_millis", COMPARISON_EPOCH)


# Get list of satellite IDs for this epoch and number of combinations possible
sat_ids = np.unique(single_epoch["sv_id"])
num_satellites = len(sat_ids)
print(f"Total Satellites: {num_satellites}")
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
    elevation_list = []
    azimuth_list = []
    az_el = []

    for i in range(len(selected_sats)):
        az = float(subset_with_el_az["az_sv_deg"][i])
        el = float(subset_with_el_az["el_sv_deg"][i])
        azimuth_list.append(az)
        elevation_list.append(el)
        az_el.append((az, el))
        ccn.add_satellite(el, az)

    # Compute condition number and DOPs
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
            "elevation_list": elevation_list,
            "azimuth_list": azimuth_list,
            "az_el": az_el,
            "error": error,
        }
    )

######################################################################################################################
# In the following section we will compute the rank 4 approximation
######################################################################################################################

all_satellites_with_el_az = glp.add_el_az(single_epoch, receiver_state, inplace=False)

receiver_position = np.array(
    [
        receiver_state["x_rx_wls_m"].item(),
        receiver_state["y_rx_wls_m"].item(),
        receiver_state["z_rx_wls_m"].item(),
    ]
)

# Create a matrix that has dimensions number of satellites x 4 unknowns
# This will be our overdetermined system matrix
A = np.zeros((num_satellites, 4))
for i in range(num_satellites):
    satellite_position = np.array(
        [
            single_epoch["x_sv_m"][i],
            single_epoch["y_sv_m"][i],
            single_epoch["z_sv_m"][i],
        ]
    )

    line_of_sight_vector = receiver_position - satellite_position
    unit_los_vector = line_of_sight_vector / np.linalg.norm(line_of_sight_vector)

    A[i, 0] = unit_los_vector[0]
    A[i, 1] = unit_los_vector[1]
    A[i, 2] = unit_los_vector[2]
    A[i, 3] = 1

U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Slice off the the top 4 for rank 4 approximation
# Because our design matrix is already rank 4, this
# doesn't actually take anything off.
Rank_4_A = (U[:, :4] @ np.diag(S[:4])) @ Vt[:4, :]
rank_4_A_cn = S[0] / S[3]

# Error will be the same as we don't cut any data out
all_sat_geodetic = pull_geodetic_from_wls(wls_estimate)
all_sat_error = compute_distance_error(all_sat_geodetic)


print(f"Rank 4 contition number {rank_4_A_cn:.2f}")
print(f"Rank 4 error: {all_sat_error:.2f}")


for pc in range(4):
    print(f"\nPrincipal Component {pc + 1}:")
    print(f"Singular value: {S[pc]:.3f}")

    # Get the weights for this principal component
    weights = U[:, pc]

    # Find satellites contributing the most
    abs_weights = np.abs(weights)
    highest_contributors = np.argsort(abs_weights)[-4:][::-1]

    print("Top contributing satellites:")
    for idx in highest_contributors:
        print(f"Satellite {sat_ids[idx]}: weight = {weights[idx]:+.3f}")

######################################################################################################################
# Analysis
######################################################################################################################

condition_numbers = np.array([r["condition_number"] for r in results])
gdops = np.array([r["GDOP"] for r in results])
errors = np.array([r["error"] for r in results])

# Calculate medians and percentiles
median_condition = np.median(condition_numbers)
median_gdop = np.median(gdops)
median_error = np.median(errors)
p75_error = np.percentile(errors, 75)
p90_error = np.percentile(errors, 90)

high_condition = condition_numbers > median_condition
high_error = errors > median_error
high_gdop = gdops > median_gdop
percentile_75_error = errors > p75_error
percentile_90_error = errors > p90_error

# Calculate Trends
percentage_high_condition_high_error = (
    np.sum(high_condition & high_error) / np.sum(high_condition) * 100
)
percentage_high_gdop_high_error = (
    np.sum(high_gdop & high_error) / np.sum(high_gdop) * 100
)
percentage_high_condition_given_75_error = (
    np.sum(high_condition & percentile_75_error) / np.sum(percentile_75_error) * 100
)
percentage_high_condition_given_90_error = (
    np.sum(high_condition & percentile_90_error) / np.sum(percentile_90_error) * 100
)

print("\n")
print(f"Median Condition Number: {median_condition:.2f}")
print(f"Median GDOP: {median_gdop:.2f}")
print(f"Median Error (m): {median_error:.2f}")
print(f"75th Percentile Error (m): {p75_error:.2f}")
print(f"90th Percentile Error (m): {p90_error:.2f}")
print("\n")
print(
    f"Percentage of groups with condition number > median that also had error > median: {percentage_high_condition_high_error:.2f}%"
)
print(
    f"Percentage of groups with GDOP > median that also had error > median: {percentage_high_gdop_high_error:.2f}%"
)
print(
    f"Percentage of groups with error in the 75th percentile that also had condition number > median: {percentage_high_condition_given_75_error:.2f}%"
)
print(
    f"Percentage of groups with error in the 90th percentile that also had condition number > median: {percentage_high_condition_given_90_error:.2f}%"
)
print("\n")

# Show plots of best and worst geometries
best_result = min(results, key=lambda r: r["condition_number"])
worst_result = max(results, key=lambda r: r["condition_number"])

# plot_skyplot(
#     best_result["az_el"],
#     title=f"Best 4-Satellite Geometry k-value={best_result['condition_number']:.2f} Error (m)={best_result['error']:.2f}",
#     sat_ids=best_result["satellites"],
# )

# plot_skyplot(
#     worst_result["az_el"],
#     title=f"Worst 4-Satellite Geometry k-value={worst_result['condition_number']:.2f} Error (m)={worst_result['error']:.2f}",
#     sat_ids=worst_result["satellites"],
# )

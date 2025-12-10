import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Convert Az/El (degrees) → Unit LOS vector (ux, uy, uz)
# ---------------------------------------------------------
def los_vector(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    ux = math.cos(el) * math.sin(az)   # East
    uy = math.cos(el) * math.cos(az)   # North
    uz = math.sin(el)                 # Up

    return np.array([ux, uy, uz])


# ---------------------------------------------------------
# Build geometry matrix G for a list of (az, el) pairs
# ---------------------------------------------------------
def build_G(sat_list):
    rows = []
    for (az, el) in sat_list:
        u = los_vector(az, el)
        row = np.append(u, 1.0)  # add clock term
        rows.append(row)
    return np.vstack(rows)


# ---------------------------------------------------------
# Compute DOP metrics
# ---------------------------------------------------------
def compute_dops(G):
    GTG = G.T @ G
    Q = np.linalg.inv(GTG)

    q_xx = Q[0,0]
    q_yy = Q[1,1]
    q_zz = Q[2,2]
    q_tt = Q[3,3]

    HDOP = math.sqrt(q_xx + q_yy)
    VDOP = math.sqrt(q_zz)
    PDOP = math.sqrt(q_xx + q_yy + q_zz)
    GDOP = math.sqrt(q_xx + q_yy + q_zz + q_tt)

    return HDOP, VDOP, PDOP, GDOP


# ---------------------------------------------------------
# Sky plot using az/el in degrees
# ---------------------------------------------------------
def plot_sky(sat_list):
    """
    Create a GNSS-style sky plot for given (az, el) degree pairs.
    Azimuth: 0° = North, positive clockwise.
    Elevation: 0° = horizon, 90° = zenith.
    """
    if not sat_list:
        return

    az_deg = np.array([az for (az, _) in sat_list])
    el_deg = np.array([el for (_, el) in sat_list])

    # Polar coordinates: 0° at zenith center, 90° at horizon rim
    theta = np.deg2rad(az_deg)       # angle around (azimuth)
    r = 90.0 - el_deg                # radius: 0 at zenith, 90 at horizon

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Configure sky-plot convention
    ax.set_theta_zero_location("N")  # 0° at North (top)
    ax.set_theta_direction(-1)       # azimuth increases clockwise
    ax.set_rlim(90, 0)               # 90° at outer ring, 0° at center

    # Radial grid labels for elevation
    ax.set_rticks([0, 30, 60, 90])   # corresponds to El = 90, 60, 30, 0
    ax.set_rlabel_position(135)      # move labels away from North axis

    ax.grid(True)
    ax.set_title("Satellite Sky Plot (Az / El)", va="bottom")

    # Plot satellite positions
    ax.scatter(theta, r)

    # Label each satellite by index
    for idx, (t, rr) in enumerate(zip(theta, r), start=1):
        ax.text(t, rr, str(idx), fontsize=9, ha="center", va="center")

    # Save and show figure
    plt.tight_layout()
    plt.savefig("skyplot.png", dpi=200)
    print("Sky plot saved as skyplot.png in the current working directory.")
    plt.show()


# ---------------------------------------------------------
# 3D Sky plot using LOS vectors on a unit sphere
# ---------------------------------------------------------
def plot_sky_3d(sat_list):
    """
    Render a 3D sky plot where each satellite is placed on a unit sphere
    according to its line-of-sight vector.
    """
    if not sat_list:
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Compute unit LOS vectors
    vecs = np.array([los_vector(az, el) for (az, el) in sat_list])
    xs, ys, zs = vecs[:,0], vecs[:,1], vecs[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot satellites
    ax.scatter(xs, ys, zs, s=50)

    # Label satellites
    for idx, (x, y, z) in enumerate(zip(xs, ys, zs), start=1):
        ax.text(x, y, z, str(idx))

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, linewidth=0.3, color="gray")

    ax.set_xlabel("X (East)")
    ax.set_ylabel("Y (North)")
    ax.set_zlabel("Z (Up)")
    ax.set_title("3D Satellite Sky Plot (Unit LOS Vectors)")

    plt.tight_layout()
    plt.savefig("skyplot_3d.png", dpi=200)
    print("3D sky plot saved as skyplot_3d.png.")
    plt.show()


# ---------------------------------------------------------
# Run experiment
# ---------------------------------------------------------
if __name__ == "__main__":

    print("Enter azimuth/elevation pairs (degrees).")
    print("Example: 0 60 means Az=0°, El=60°.")
    print("Enter blank line when finished.\n")

    sats = []
    while True:
        line = input("Enter azimuth elevation: ")
        if not line.strip():
            break
        az, el = map(float, line.split())
        sats.append((az, el))

    if len(sats) < 4:
        print("\nYou need at least 4 satellites for GNSS geometry (G matrix).")
        exit()

    # Build G matrix
    G = build_G(sats)

    # SVD and condition number
    U, S, Vt = np.linalg.svd(G)
    cond_num = S[0] / S[-1]

    # DOP metrics
    HDOP, VDOP, PDOP, GDOP = compute_dops(G)

    # Outputs
    print("\n---------------------------------------------------")
    print("Geometry Matrix G:")
    print(G)

    print("\nSingular Values (largest → smallest):")
    print(S)

    print(f"\nCondition Number: {cond_num:.3f}")

    print("\nDOP Metrics:")
    print(f"HDOP = {HDOP:.3f}")
    print(f"VDOP = {VDOP:.3f}")
    print(f"PDOP = {PDOP:.3f}")
    print(f"GDOP = {GDOP:.3f}")

    print("\nInterpretation:")
    print("- Smallest singular value tells you the weakest direction.")
    print("- Large condition number = poor geometry.")
    print("- High DOP values = poor expected accuracy.")

    # Sky plot visualization
    print("\nGenerating sky plot window...")
    plot_sky(sats)

    print("\nGenerating 3D sky plot window...")
    plot_sky_3d(sats)

    print("---------------------------------------------------")
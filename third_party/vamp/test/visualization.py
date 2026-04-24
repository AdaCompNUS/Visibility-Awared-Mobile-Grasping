import matplotlib

# If running in a headless environment:
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import base64
import io

# Data from your table
points = np.array(
    [10068, 20199, 23570, 27481, 42011, 70025, 86517, 95327, 101194, 155521]
)

# Vamp data (motion planning in microseconds, point cloud in seconds)
vamp_planning = np.array([1607, 1881, 1616, 1493, 2898, 2114, 3356, 2162, 2192, 3448])
vamp_pc = np.array(
    [0.018, 0.036, 0.040, 0.040, 0.065, 0.104, 0.130, 0.145, 0.150, 0.230]
)

# MoveIt data (motion planning in microseconds, point cloud in seconds)
moveit_planning = np.array(
    [272108, 168415, 108050, 250081, 294316, 186843, 369033, 427583, 322321, 504214]
)
moveit_pc = np.array([0.2, 0.462, 0.78, 0.96, 2.19, 8.36, 14.26, 19.1, 20.13, 53.25])

# For a bar chart, we need an x-axis index
indices = np.arange(len(points))

# Figure and subplots
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

################################################################
# 1) Left subplot: Planning Times (with a log scale on the y-axis)
################################################################
bar_width = 0.4

ax1.bar(
    indices - bar_width / 2,
    vamp_planning,
    width=bar_width,
    label="Vamp (µs)",
    color="cornflowerblue",
)
ax1.bar(
    indices + bar_width / 2,
    moveit_planning,
    width=bar_width,
    label="MoveIt (µs)",
    color="salmon",
)

ax1.set_xlabel("Index of Test (Number of Points)")
ax1.set_ylabel("Planning Time (µs)")
ax1.set_title("Motion Planning Time (Log Scale)")
ax1.set_xticks(indices)
ax1.set_xticklabels(points, rotation=45, ha="right")

# Apply log scale to highlight large differences
ax1.set_yscale("log")

ax1.legend()
ax1.grid(True, which="both", ls="--")

# Annotate bars with their values
for i, v in enumerate(vamp_planning):
    ax1.text(
        i - bar_width / 2, v, f"{v}", ha="center", va="bottom", fontsize=8, rotation=90
    )
for i, v in enumerate(moveit_planning):
    ax1.text(
        i + bar_width / 2, v, f"{v}", ha="center", va="bottom", fontsize=8, rotation=90
    )

################################################################
# 2) Right subplot: Point Cloud Construction Times
#    (Currently in linear scale; you can uncomment set_yscale('log')
#    if you also want a log scale here.)
################################################################
ax2.bar(
    indices - bar_width / 2,
    vamp_pc,
    width=bar_width,
    label="Vamp (s)",
    color="cornflowerblue",
)
ax2.bar(
    indices + bar_width / 2,
    moveit_pc,
    width=bar_width,
    label="MoveIt (s)",
    color="salmon",
)

ax2.set_xlabel("Index of Test (Number of Points)")
ax2.set_ylabel("Point Cloud Construction Time (s)")
ax2.set_title("Point Cloud Construction Time (Linear Scale)")
ax2.set_xticks(indices)
ax2.set_xticklabels(points, rotation=45, ha="right")

# Uncomment this if you want log scale for the second subplot as well:
# ax2.set_yscale('log')

ax2.legend()
ax2.grid(True, which="both", ls="--")

# Annotate bars with their values
for i, v in enumerate(vamp_pc):
    ax2.text(
        i - bar_width / 2,
        v,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90,
    )
for i, v in enumerate(moveit_pc):
    ax2.text(
        i + bar_width / 2,
        v,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90,
    )

################################################################
# Final layout and saving
################################################################
plt.suptitle("Vamp vs. MoveIt: Motion Planning & Point Cloud Times", fontsize=14)
plt.tight_layout()

# Save the plot to a bytes buffer in PNG format
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)

# Encode the image in base64 if you want to display it inline or use it in HTML
img_base64 = base64.b64encode(buf.read()).decode("utf-8")

# If you just want to save the figure as a file, uncomment:
plt.savefig("vamp_vs_moveit_barchart.png")

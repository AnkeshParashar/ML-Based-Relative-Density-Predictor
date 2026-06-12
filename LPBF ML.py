import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

# DATA SET

data_AlSi10Mg = {
    "Laser Power(W)": [100, 100, 130, 100, 100, 130, 160, 340, 370, 400, 100,  
                        130, 160, 340, 370, 400, 100, 130, 130, 190, 250, 310,
                        400, 100, 100, 130, 160, 190, 220, 280, 370, 400, 100,
                        100, 160, 160, 190, 190, 250, 280, 310, 340, 340, 370,
                        400, 130, 130, 190, 220, 220, 220, 250, 310, 370, 400,
                        400, 130, 160, 160, 190, 190, 190, 220, 220, 250, 250,
                        280, 310, 340, 370, 370, 160, 160, 160, 190, 220, 220,
                        250, 250, 250, 280, 280, 280, 280, 280, 280, 310, 310,
                        310, 310, 340, 340, 400, 400, 190, 220, 340, 370, 370],

    "Scan Speed(m/s)": [2.25, 2.5, 2.5, 1.75, 2, 2.25, 2.5, 0.5, 0.5, 0.5, 1.5,
                        2, 2.25, 0.75, 0.75, 0.75, 1.25, 1.5, 1.75, 2.5, 0.5, 0.75,
                        1, 0.75, 1, 1.25, 2, 2.25, 0.5, 0.75, 1, 1.25, 0.5,
                        0.75, 1.5, 1.75, 2, 0.5, 0.75, 1, 1, 1, 1.25, 1.25,
                        1.5, 0.5, 0.75, 0.5, 2.25, 2.5, 0.75, 1, 1.25, 1.5, 1.75,
                        2, 1, 0.75, 0.5, 1.5, 0.75, 1.75, 2, 2.25, 2.25, 2.5,
                        1.25, 1.5, 1.5, 1.75, 2, 1, 1.25, 0.75, 1, 1.75, 1,
                        1.75, 1.25, 2, 2.5, 1.25, 1.5, 2, 1.75, 2.25, 1.75, 2,
                        2.25, 2.5, 2, 2.5, 2.25, 2.5, 1.25, 1.25, 2.25, 2.25, 2.5],

    "Relative Density(%)": [92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 94,
                            94, 94, 94, 94, 94, 95, 95, 95, 95, 95, 95,
                            95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 97,
                            97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97,
                            97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                            98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3,
                            99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.3,
                            99.3, 99.3, 99.3, 99.3, 99.3, 99.3, 99.6, 99.6, 99.6, 99.6, 99.6]
}

# ------------------------------------------
# Material 1: AlSi10Mg
# ------------------------------------------

df_AlSi10Mg = pd.DataFrame(data_AlSi10Mg)

# VISULIZATION

# INTERPOLATION
x = df_AlSi10Mg["Laser Power(W)"]
y = df_AlSi10Mg["Scan Speed(m/s)"]
z = df_AlSi10Mg["Relative Density(%)"]

laser_i = np.linspace(x.min(), x.max(), 100)
scan_i = np.linspace(y.min(), y.max(), 100)
new_laser, new_scan = np.meshgrid(laser_i, scan_i)

new_rel_density = griddata((x, y), z, (new_laser, new_scan), method="cubic")

# 3D SURFACE
fig_3d = plt.figure(figsize=(12, 6))
visual = fig_3d.add_subplot(projection="3d")

surface = visual.plot_surface(new_laser, new_scan, new_rel_density, cmap = "jet", edgecolor = "none")

visual.scatter(x, y, z, color = "black", s = 10)
visual.contour(new_laser, new_scan, new_rel_density, levels = np.arange(92, 100, 0.4), zdir = "z", offset = 90, cmap = "jet")

visual.set_xlim(0, 500)
visual.set_ylim(0, 3.00)
visual.set_title("Relationship between Laser_Power, Scan_Speed, Relative_Density", fontsize = 15)
visual.set_xlabel("Laser Power(W)")
visual.set_ylabel("Scan Speed(m/s)")
visual.set_zlabel("Relative Density(%)")
plt.colorbar(surface, label="Relative Density (%)")

# CONTOURS
fig_countour = plt.figure(figsize=(7.5, 6))
contours = plt.contour(new_laser, new_scan, new_rel_density, levels = np.arange(92, 100, 0.4), cmap = "jet")

plt.clabel(contours, inline=True, fontsize=10)
plt.xlim(100, 400)
plt.ylim(0.5, 2.50)
plt.xlabel("Laser Power (W)")
plt.ylabel("Scan Speed (m/s)")
plt.title("Relative Density Contour Plot")
plt.colorbar(contours, label="Relative Density (%)")

plt.grid(True)
plt.show()

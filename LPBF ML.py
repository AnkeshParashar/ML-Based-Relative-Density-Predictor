import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.metrics import root_mean_squared_error, r2_score

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

data_Ti6Al4V = {
    "Laser Power(W)": [150, 150, 150, 150, 150, 150,
                        200, 200, 200, 200, 200, 200,
                        250, 250, 250, 250, 250, 250,
                        300, 300, 300, 300, 300, 300,
                        350, 350, 350, 350, 350, 350,
                        400, 400, 400, 400, 400, 400],
    "Scan Speed(m/s)": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                         0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    "Relative Density(%)": [99.00, 98.54, 97.65, 95.83, 94.17, 92.51,
                             99.28, 99.13, 98.99, 98.57, 97.92, 96.05,
                             99.43, 99.33, 99.31, 99.18, 98.58, 97.16,
                             99.45, 99.39, 99.18, 98.78, 98.28, 97.60,
                             99.29, 99.00, 98.54, 98.33, 98.05, 97.67,
                             99.20, 99.12, 98.90, 98.81, 98.58, 98.34]
}

data_IN718 = {
    "Laser Power(W)": [125, 125, 125, 125,
                        200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                        275, 275, 275, 275, 275, 275, 275, 275, 275,
                        350, 350, 350, 350, 350, 350, 350, 350, 350],
    "Scan Speed(m/s)": [0.2, 0.4, 0.6, 0.8,
                         0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
                         0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                         0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
    "Relative Density(%)": [91.11, 95.33, 96.95, 97.00,
                             93.81, 94.79, 96.67, 97.83, 99.59, 98.69, 98.28, 98.09, 97.48, 95.85,
                             97.45, 98.26, 99.71, 99.75, 99.80, 99.86, 99.25, 99.28, 99.30,
                             96.98, 98.33, 98.26, 98.78, 98.67, 98.85, 98.79, 98.14, 97.62]
}

print("1: AlSi10Mg")
print("2: Ti6Al4V")
print("3: IN718")

material = int(input("Choose Material: "))

# ------------------------------------------
# Material 1: AlSi10Mg
# ------------------------------------------

if material == 1:

    print("AlSi10Mg")
    df_AlSi10Mg = pd.DataFrame(data_AlSi10Mg)
    print(df_AlSi10Mg)
    print("-----------------------------")

    # INPUT(X) and OUTPUT(y)

    X = df_AlSi10Mg[["Laser Power(W)", "Scan Speed(m/s)"]].values
    y = df_AlSi10Mg["Relative Density(%)"].values

    # PIPELINE FORMATION --> Scaling of Input(X) and Defining Model

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale = 1.0)

    pipe = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, normalize_y = True, alpha = 0.1)
    )

    # BEST SPLITTING of X 

    mean_scores = []

    for random_state in range(5):

        kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)

        scores = [cross_val_score(pipe, X, y, cv = kf, scoring = "r2")]
        current_mean = np.mean(scores)
        print("Random State: ", random_state, " Current Mean: ", current_mean)
        mean_scores.append(current_mean)

    max_mean = np.max(mean_scores)
    max_mean_random_state = np.argmax(mean_scores)
    print("-----------------------------")
    print("Max Mean: ", max_mean)
    print("Max Mean Random State: ", max_mean_random_state)
    print("-----------------------------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = max_mean_random_state)

    # TRAINING of MODEL

    pipe.fit(X_train, y_train)

    gpr = pipe.named_steps["gaussianprocessregressor"]
    print("Initial Kernel:", gpr.kernel)
    print("Optimized Kernel:", gpr.kernel_)
    print("-----------------------------")

    # MODEL PREDICTION

    y_pred, y_std = pipe.predict(X_test, return_std = True)

    for i in range(len(y_test)):
        print("Test Case:", i + 1)
        print("Actual Density(%):", y_test[i])
        print("Predicted Density(%):", y_pred[i])
        print("Uncertainity(%): ±", y_std[i])
    print("-----------------------------")

    # EVALUATION of MODEL (SCORES)

    rootmean_squared_error = root_mean_squared_error(y_test, y_pred)
    print("Root Mean Squared Error:", rootmean_squared_error)
    r2score = r2_score(y_test, y_pred)
    print("R2 Score:", r2score)
    print("-----------------------------")


    # PREDICT NEW PARAMETERS

    # FIXED PARAMETERS
    print("Hatch Spacing(µm): 100")
    print("Layer Thickness(µm): 30")

    # PREDICTION
    while True:

        print("Chooese Laser Power between 10W and 500W")
        new_laser_power = float(input("New Laser Power(W): "))

        if 10 <= new_laser_power <=500:
            break

        print("Laser Power must be between 10W and 500W")

    print("Laser Power(W):", new_laser_power)

    while True:

        print("Choose Scan Speed between 0.1m/s and 3m/s")
        new_scan_speed = float(input("New Scan Speed(m/s): "))

        if 0.1 <= new_scan_speed <= 3:
            break

        print("Scan Speed must be between 0.1m/s and 3m/s")

    print("Scan Speed(m/s):", new_scan_speed)
    print("-----------------------------")

    new_parameters = [[new_laser_power, new_scan_speed]]

    new_y_pred, new_y_std = pipe.predict(new_parameters, return_std = True)
    print("Predicted Density(%):", new_y_pred)
    print("Uncertainity(%): ±", new_y_std)
    print("-----------------------------")

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

    visual.set_xlim(100, 500)
    visual.set_ylim(0, 2.5)
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

    # COMPARISION BETWEEN ACTUAL v/s PREDICTED DENSITY
    experiment_no = list(range(1, len(y_pred) + 1))

    data_for_visualisation = {
        "Experiment Number": experiment_no,
        "Actual Density(%)": y_test,
        "Predicted Density(%)": y_pred
    }

    df2 = pd.DataFrame(data_for_visualisation)

    plt.figure(figsize = (7.5, 6))
    plt.plot(df2["Experiment Number"], df2["Actual Density(%)"], color = "black", marker = "o", label = "Actual Density(%)")
    plt.plot(df2["Experiment Number"], df2["Predicted Density(%)"], color = "red", marker = "o", linestyle = "--", label = "Predicted Density(%)")
    plt.title("Predicted Density v/s Actual Density(%)")
    plt.xlabel("Experiment Number")
    plt.ylabel("Density(%)")
    plt.legend()
    plt.grid(True)

    # REPRESENTATION of NEW PREDICTED DENSITY
    new_exp_no = len(experiment_no) + 1
    plt.scatter(new_exp_no, new_y_pred, color="blue", marker="*", label="New Prediction")

    plt.show()

# ------------------------------------------
# Material 2: Ti6Al4V
# ------------------------------------------

if material == 2:

    print("Ti6Al4V")
    df_Ti6Al4V = pd.DataFrame(data_Ti6Al4V)
    print(df_Ti6Al4V)
    print("-----------------------------")

    # INPUT(X) and OUTPUT(y)

    X = df_Ti6Al4V[["Laser Power(W)", "Scan Speed(m/s)"]].values
    y = df_Ti6Al4V["Relative Density(%)"].values

    # PIPELINE FORMATION --> Scaling of Input(X) and Defining Model

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale = 1.0)

    pipe = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, normalize_y = True, alpha = 0.1)
    )

    # BEST SPLITTING of X 

    mean_scores = []

    for random_state in range(5):

        kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)

        scores = [cross_val_score(pipe, X, y, cv = kf, scoring = "r2")]
        current_mean = np.mean(scores)
        print("Random State: ", random_state, " Current Mean: ", current_mean)
        mean_scores.append(current_mean)

    max_mean = np.max(mean_scores)
    max_mean_random_state = np.argmax(mean_scores)
    print("-----------------------------")
    print("Max Mean: ", max_mean)
    print("Max Mean Random State: ", max_mean_random_state)
    print("-----------------------------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = max_mean_random_state)

    # TRAINING of MODEL

    pipe.fit(X_train, y_train)

    gpr = pipe.named_steps["gaussianprocessregressor"]
    print("Initial Kernel:", gpr.kernel)
    print("Optimized Kernel:", gpr.kernel_)
    print("-----------------------------")

    # MODEL PREDICTION

    y_pred, y_std = pipe.predict(X_test, return_std = True)

    for i in range(len(y_test)):
        print("Test Case:", i + 1)
        print("Actual Density(%):", y_test[i])
        print("Predicted Density(%):", y_pred[i])
        print("Uncertainity(%): ±", y_std[i])
        print("-----------------------------")

    # EVALUATION of MODEL (SCORES)

    rootmean_squared_error = root_mean_squared_error(y_test, y_pred)
    print("Root Mean Squared Error:", rootmean_squared_error)
    r2score = r2_score(y_test, y_pred)
    print("R2 Score:", r2score)
    print("-----------------------------")

    # PREDICT NEW PARAMETERS

    # FIXED PARAMETERS
    print("Hatch Spacing(µm): 110")
    print("Layer Thickness(µm): 50")

    # PREDICTION
    while True:
        print("Chooese Laser Power between 100W and 500W")
        new_laser_power = float(input("New Laser Power(W): "))

        if 100 <= new_laser_power <=500:
            break

        print("Laser Power must be between 100W and 500W")

    print("Laser Power(W):", new_laser_power)

    while True:
        print("Choose Scan Speed between 0.5m/s and 2m/s")
        new_scan_speed = float(input("New Scan Speed(m/s): "))

        if 0.5 <= new_scan_speed <= 2:
            break

        print("Scan Speed must be between 0.5m/s and 2m/s")

    print("Scan Speed(m/s):", new_scan_speed)
    print("-----------------------------")

    new_parameters = [[new_laser_power, new_scan_speed]]

    new_y_pred, new_y_std = pipe.predict(new_parameters, return_std = True)
    print("Predicted Density(%):", new_y_pred)
    print("Uncertainity(%): ±", new_y_std)
    print("-----------------------------")

    # VISULIZATION

    # INTERPOLATION
    x = df_Ti6Al4V["Laser Power(W)"]
    y = df_Ti6Al4V["Scan Speed(m/s)"]
    z = df_Ti6Al4V["Relative Density(%)"]

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

    visual.set_xlim(150, 450)
    visual.set_ylim(0.5, 1.5)
    visual.set_title("Relationship between Laser_Power, Scan_Speed, Relative_Density", fontsize = 15)
    visual.set_xlabel("Laser Power(W)")
    visual.set_ylabel("Scan Speed(m/s)")
    visual.set_zlabel("Relative Density(%)")
    plt.colorbar(surface, label="Relative Density (%)")

    # CONTOURS
    fig_countour = plt.figure(figsize=(7.5, 6))
    contours = plt.contour(new_laser, new_scan, new_rel_density, levels = np.arange(92, 100, 0.4), cmap = "jet")

    plt.clabel(contours, fontsize=10)
    plt.xlim(150, 400)
    plt.ylim(0.9, 1.40)
    plt.xlabel("Laser Power (W)")
    plt.ylabel("Scan Speed (m/s)")
    plt.title("Relative Density Contour Plot")
    plt.colorbar(contours, label="Relative Density (%)")
    plt.grid(True)

    # COMPARISION BETWEEN ACTUAL v/s PREDICTED DENSITY
    experiment_no = list(range(1, len(y_pred) + 1))

    data_for_visualisation = {
        "Experiment Number": experiment_no,
        "Actual Density(%)": y_test,
        "Predicted Density(%)": y_pred
    }

    df2 = pd.DataFrame(data_for_visualisation)

    plt.figure(figsize = (7.5, 6))
    plt.plot(df2["Experiment Number"], df2["Actual Density(%)"], color = "black", marker = "o", label = "Actual Density(%)")
    plt.plot(df2["Experiment Number"], df2["Predicted Density(%)"], color = "red", marker = "o", linestyle = "--", label = "Predicted Density(%)")
    plt.title("Predicted Density v/s Actual Density(%)")
    plt.xlabel("Experiment Number")
    plt.ylabel("Density(%)")
    plt.legend()
    plt.grid(True)

    # REPRESENTATION of NEW PREDICTED DENSITY
    new_exp_no = len(experiment_no) + 1
    plt.scatter(new_exp_no, new_y_pred, color="blue", marker="*", label="New Prediction")

    plt.show()

# ------------------------------------------
# Material 3: IN718
# ------------------------------------------

if material == 3:

    print("IN718")
    df_IN718 = pd.DataFrame(data_IN718)
    print(df_IN718)
    print("-----------------------------")

    # INPUT(X) and OUTPUT(y)

    X = df_IN718[["Laser Power(W)", "Scan Speed(m/s)"]].values
    y = df_IN718["Relative Density(%)"].values

    # PIPELINE FORMATION --> Scaling of Input(X) and Defining Model

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale = 1.0)

    pipe = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, normalize_y = True, alpha = 0.1)
    )

    # BEST SPLITTING of X 

    mean_scores = []

    for random_state in range(5):

        kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)

        scores = [cross_val_score(pipe, X, y, cv = kf, scoring = "r2")]
        current_mean = np.mean(scores)
        print("Random State: ", random_state, " Current Mean: ", current_mean)
        mean_scores.append(current_mean)

    max_mean = np.max(mean_scores)
    max_mean_random_state = np.argmax(mean_scores)
    print("-----------------------------")
    print("Max Mean: ", max_mean)
    print("Max Mean Random State: ", max_mean_random_state)
    print("-----------------------------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = max_mean_random_state)

    # TRAINING of MODEL

    pipe.fit(X_train, y_train)

    gpr = pipe.named_steps["gaussianprocessregressor"]
    print("Initial Kernel:", gpr.kernel)
    print("Optimized Kernel:", gpr.kernel_)
    print("-----------------------------")

    # MODEL PREDICTION

    y_pred, y_std = pipe.predict(X_test, return_std = True)

    for i in range(len(y_test)):
        print("Test Case:", i + 1)
        print("Actual Density(%):", y_test[i])
        print("Predicted Density(%):", y_pred[i])
        print("Uncertainity(%): ±", y_std[i])
        print("-----------------------------")

    # EVALUATION of MODEL (SCORES)

    rootmean_squared_error = root_mean_squared_error(y_test, y_pred)
    print("Root Mean Squared Error:", rootmean_squared_error)
    r2score = r2_score(y_test, y_pred)
    print("R2 Score:", r2score)
    print("-----------------------------")

    # PREDICT NEW PARAMETERS

    # FIXED PARAMETERS
    print("Hatch Spacing(µm): 120")
    print("Layer Thickness(µm): 30")

    # PREDICTION
    while True:

        print("Chooese Laser Power between 100W and 400W")
        new_laser_power = float(input("New Laser Power(W): "))

        if 100 <= new_laser_power <=400:
            break

        print("Laser Power must be between 100W and 400W")

    print("Laser Power(W):", new_laser_power)

    while True:

        print("Choose Scan Speed between 0.1m/s and 2.5m/s")
        new_scan_speed = float(input("New Scan Speed(m/s): "))

        if 0.1 <= new_scan_speed <= 2.5:
            break

        print("Scan Speed must be between 0.1m/s and 2.5m/s")

    print("Scan Speed(m/s):", new_scan_speed)
    print("-----------------------------")

    new_parameters = [[new_laser_power, new_scan_speed]]

    new_y_pred, new_y_std = pipe.predict(new_parameters, return_std = True)
    print("Predicted Density(%):", new_y_pred)
    print("Uncertainity(%): ±", new_y_std)
    print("-----------------------------")

    # VISULIZATION

    # INTERPOLATION
    x = df_IN718["Laser Power(W)"]
    y = df_IN718["Scan Speed(m/s)"]
    z = df_IN718["Relative Density(%)"]

    laser_i = np.linspace(x.min(), x.max(), 100)
    scan_i = np.linspace(y.min(), y.max(), 100)
    new_laser, new_scan = np.meshgrid(laser_i, scan_i)

    new_rel_density = griddata((x, y), z, (new_laser, new_scan), method="cubic")

    # 3D SURFACE
    fig_3d = plt.figure(figsize=(12, 6))
    visual = fig_3d.add_subplot(projection="3d")

    surface = visual.plot_surface(new_laser, new_scan, new_rel_density, cmap = "jet", edgecolor = "none")

    visual.scatter(x, y, z, color = "black", s = 10)
    visual.contour(new_laser, new_scan, new_rel_density, levels = np.arange(90, 100, 0.4), zdir = "z", offset = 90, cmap = "jet")

    visual.set_xlim(100, 400)
    visual.set_ylim(0, 2.5)
    visual.set_title("Relationship between Laser_Power, Scan_Speed, Relative_Density", fontsize = 15)
    visual.set_xlabel("Laser Power(W)")
    visual.set_ylabel("Scan Speed(m/s)")
    visual.set_zlabel("Relative Density(%)")
    plt.colorbar(surface, label="Relative Density (%)")

    # CONTOURS
    fig_countour = plt.figure(figsize=(7.5, 6))
    contours = plt.contour(new_laser, new_scan, new_rel_density, levels = np.arange(90, 100, 0.4), cmap = "jet")

    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlim(125, 350)
    plt.ylim(0.2, 2.15)
    plt.xlabel("Laser Power (W)")
    plt.ylabel("Scan Speed (m/s)")
    plt.title("Relative Density Contour Plot")
    plt.colorbar(contours, label="Relative Density (%)")
    plt.grid(True)

    # COMPARISION BETWEEN ACTUAL v/s PREDICTED DENSITY
    experiment_no = list(range(1, len(y_pred) + 1))

    data_for_visualisation = {
        "Experiment Number": experiment_no,
        "Actual Density(%)": y_test,
        "Predicted Density(%)": y_pred
    }

    df2 = pd.DataFrame(data_for_visualisation)

    plt.figure(figsize = (7.5, 6))
    plt.plot(df2["Experiment Number"], df2["Actual Density(%)"], color = "black", marker = "o", label = "Actual Density(%)")
    plt.plot(df2["Experiment Number"], df2["Predicted Density(%)"], color = "red", marker = "o", linestyle = "--", label = "Predicted Density(%)")
    plt.title("Predicted Density v/s Actual Density(%)")
    plt.xlabel("Experiment Number")
    plt.ylabel("Density(%)")
    plt.legend()
    plt.grid(True)

    # REPRESENTATION of NEW PREDICTED DENSITY
    new_exp_no = len(experiment_no) + 1
    plt.scatter(new_exp_no, new_y_pred, color="blue", marker="*", label="New Prediction")

    plt.show()

import os
import numpy as np
import csv
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pickle


#######################################
##
##    FIRST EXTENDED KALMAN FILTER
##
#######################################

def first_ekf(mu, Sigma, Tc, E, w, F, G, Q_w, H, Q_o, o_r, o_w):
    """

    First Extended Kalman Filter for IMU Localization. Algorithms
    obtained from Colombo et al.

    Returns new Sigma (attitude covariance matrix) and new mu (attitude update). 

    mu: previous mean of the Gaussian distribution
    Tc: sampling period
    E: system matrix mapping angular velocities of the platform into 
        time derivatives of the angles
    w: vector of angular velocities measured by the gyroscope
    F: Jacobian of the system matrix w.r.t. the state vector theta
        computed at [mu, w]
    G: Jacobian of the system matrix w.r.t. the input velocity 
        vector w computed at [mu, u]
    Q_w: covariance matrix associated with w including both intrinsic
        and measurement uncertainty contributions
    H: Jacobian of the output matrix U * o_w
    Q_o: covariance matrix of the measurement vector o_r
    o_r: current measurement
    o_w: contains the gravitational acceleration vector, the north
        directed unit vector and the yaw

    NOTE from the paper:
        P: covariance matrix
        theta: attitude update - 3D vector containing estimated roll,
            pitch and yaw

    """
'''
    ########## PREDICTION STEP ##########
    # Use dynamics to predict what will happen

    mu_bar = mu + Tc*E*w
    Signma_bar = F*Sigma*F.T + G*Q_w*G.T

    ########## CORRECTION STEP ##########
    # Use sensor measurement to correct prediction

    o_r_bar = U(mu_bar)*o_w
    new_o_r = U(mu_bar)*o_w_new
    K = Sigma_bar*H.T*np.linalg.inv(H*Sigma_bar*H.T + Q_o) # Kalman gain
    mu_new = mu_bar + K*(o_r - o_r_bar)
    Sigma_new = (np.identity(K.shape[0] - K*H)*Sigma_bar

    return mu_new, Sigma_new'''
    
#######################################
##                                   ##
##    PARSE DATA INTO NUMPY ARRAYS   ##
##                                   ##
#######################################

def get_data_directory():
    """
    Get repository data directory.
    """
    
    # change directory
    os.chdir('../../data/initial/')
    # get working directory
    data_directory = os.getcwd()
    os.chdir('../../dev/first_ekf/')
    return data_directory

def gen_sensor_array(time_col, z_col, device_csv_array):
    """
    Create NumPy array given start and end rows and columns.
    """
    num_rows = len(device_csv_array)
    return device_csv_array[0: num_rows-1, time_col: z_col]

def make_matrices():
    """
    Format IMU CSV data into Numpy matrices.
    """
    data_dir = get_data_directory()

    print("Generating iPhone Array...")
    iphone_csv_array = genfromtxt(os.path.join(data_dir, 'iphoneIMU.csv'), delimiter=",", skip_header=1)
    print("Generating iWatch Array...")
    iwatch_csv_array = genfromtxt(os.path.join(data_dir, 'iWatchIMU.csv'), delimiter=",", skip_header=1)

    ##################### create accel, gyro, magnetometer numpy arrays: iPhone #####################
    # accel
    iphone_accel_time_col = 18
    iphone_accel_z_col = 22
    print("Generating iPhone Accel Array...")
    iphone_accel = gen_sensor_array(iphone_accel_time_col, iphone_accel_z_col, iphone_csv_array)

    # gyro
    iphone_gyro_time_col = 22
    iphone_gyro_z_col = 26
    print("Generating iPhone Gyro Array...")
    iphone_gyro = gen_sensor_array(iphone_gyro_time_col, iphone_gyro_z_col, iphone_csv_array)

    # mag
    iphone_mag_time_col = 26
    iphone_mag_z_col = 30
    print("Generating iPhone Mag Array...")
    iphone_mag = gen_sensor_array(iphone_mag_time_col, iphone_mag_z_col, iphone_csv_array)

    ##################### create accel and gyro numpy arrays: iWatch #####################
    # accel
    iwatch_accel_time_col = 10
    iwatch_accel_z_col = 14
    print("Generating iWatch Accel Array...")
    iwatch_accel = gen_sensor_array(iwatch_accel_time_col, iwatch_accel_z_col, iwatch_csv_array)

    # gyro
    iwatch_gyro_time_col = 14
    iwatch_gyro_z_col = 18
    print("Generating iWatch Gyro Array...")
    iwatch_gyro = gen_sensor_array(iwatch_gyro_time_col, iwatch_gyro_z_col, iwatch_csv_array)

    # finish
    return iphone_accel, iphone_gyro, iphone_mag, iwatch_accel, iwatch_gyro

def main():
    # make three matrices of gyro, accel, magnetometer data
    iphone_accel, iphone_gyro, iphone_mag, iwatch_accel, iwatch_gyro = make_matrices()

if __name__ == "__main__":
    main()
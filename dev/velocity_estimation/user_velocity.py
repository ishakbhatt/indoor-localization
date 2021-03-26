###########################################
##                                       ##
##  USER VELOCITY ESTIMATION ALGORITHM   ##
##                                       ##
###########################################

# imports
import os
import numpy as np
from numpy import genfromtxt
from scipy.signal import welch


###########################################
##                                       ##
##  USER VELOCITY ESTIMATION ALGORITHM   ##
##                                       ##
###########################################

# calculate power spectral densities

def power_spectral_density(sensor_array):
    freq, 


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
    os.chdir('../dev/user_velocity/')
    return data_directory

def gen_sensor_array(time_col, z_col, device_csv_array):
    """
    Create NumPy array given start and end rows and columns.
    """
    num_rows = len(device_csv_array)
    return device_csv_array[0: num_rows-1][time_col: z_col]  

def make_matrices():
    """
    Format IMU CSV data into Numpy matrices.
    """
    data_dir = get_data_directory()

    iphone_csv_array = genfromtxt(os.path.join(data_dir, 'iphoneIMU.csv'), delimiter=",", skip_header=1)
    iwatch_csv_array = genfromtxt(os.path.join(data_dir, 'iWatchIMU.csv'), delimiter=",", skip_header=1)

    ##################### create accel, gyro, magnetometer numpy arrays: iPhone #####################
    # accel
    accel_time_col = 18
    accel_z_col = 21
    iphone_accel = gen_sensor_array(accel_time_col, accel_z_col, iphone_csv_array)    

    # gyro
    gyro_time_col = 22
    gyro_z_col = 25
    iphone_gyro = gen_sensor_array(accel_time_col, accel_z_col, iphone_csv_array)  

    ##################### create accel, gyro, magnetometer numpy arrays: iWatch #####################e
    # accel
    iwatch_accel = gen_sensor_array(accel_time_col, accel_z_col, iwatch_csv_array)    

    # gyro
    iwatch_gyro = gen_sensor_array(accel_time_col, accel_z_col, iwatch_csv_array)  

    # finish
    return iphone_accel, iphone_gyro, iwatch_accel, iwatch_gyro

# test
def main():


if __name__ == "__main__": 
    main()









###########################################
##                                       ##
##  USER VELOCITY ESTIMATION ALGORITHM   ##
##                                       ##
###########################################

# libraries & functions
import os
import numpy as np
from numpy import genfromtxt
from scipy.signal import welch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

###########################################
##                                       ##
##           FILTER SELECTION            ##
##                                       ##
###########################################

# get results directory
def get_results_directory():
    '''
    Return velocity estimate results directory.
    '''
    os.chdir('results')
    results_directory = os.getcwd()
    os.chdir('..')
    return results_directory

# calculate power spectral densities
def power_spectral_density(sensor_array, sensor):
    """
    Plot the PSD of the sensor data to estimate filter with cutoff freq.
    """
    # Calculate PSD using Welch's PSD Algorithm
    sensor_overlap = (sensor_array.size/2)
    f_sample = 30 # 30 Hz sampling freq TODO: change to 100 Hz for final experiment
    freq, p_density = welch(sensor_array, f_sample, window='hann')#, noverlap=sensor_overlap) 
    print("Calculated PSD for " + sensor + ".")

    # Plot the PSD and save to results
    plt.semilogy(freq, p_density[1])
    plt.ylim([0.5e-3, 1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density [V**2/Hz]')
    plt.title("PSD Estimation for " + sensor)
    plt.show()
    plt.savefig(get_results_directory() + "/" + sensor + "_PSD.png")
    print("Saved PSD to results.")

#######################################
##                                   ##
##    PARSE DATA INTO NUMPY ARRAYS   ##
##                                   ##
#######################################

def get_data_directory():
    """
    Return repository data directory.
    """
    # change directory
    os.chdir('../../data/initial/')
    # get working directory
    data_directory = os.getcwd()
    print(data_directory)
    os.chdir('../../dev/velocity_estimation/')
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

    ##################### create accel, gyro, magnetometer numpy arrays: iWatch #####################
    # accel
    iwatch_accel = gen_sensor_array(accel_time_col, accel_z_col, iwatch_csv_array)    

    # gyro
    iwatch_gyro = gen_sensor_array(accel_time_col, accel_z_col, iwatch_csv_array)  

    # finish
    return iphone_accel, iphone_gyro, iwatch_accel, iwatch_gyro

# test
def main():
    iphone_accel, iphone_gyro, iwatch_accel, iwatch_gyro = make_matrices()
    power_spectral_density(iphone_accel, "iPhoneAccelerometer")
    power_spectral_density(iphone_gyro, "iPhoneGyroscope")
    power_spectral_density(iwatch_accel, "iWatchAccelerometer")
    power_spectral_density(iwatch_gyro, "iWatchGyroscope")
    print("Finished.")


if __name__ == "__main__": 
    main()









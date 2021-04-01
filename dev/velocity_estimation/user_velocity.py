#!/usr/bin/env python
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
from scipy.signal import butter
from scipy.signal import lfilter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

###########################################
##                                       ##
##           FILTER SELECTION            ##
##                                       ##
###########################################

def low_pass_filter(sensor_array, sensor, cutoff_freq):
    '''
    Apply low pass filter on sensor data based on PSD.
    '''
    samp_duration = 0
    xyz = ["Time", "X", "Y", "Z"]
    filtered_signals = []

    for i in range(len(xyz)):
        if i == 0:
            samp_duration = sensor_array[sensor_array.shape[0]-1, 0]
            print("Calculating sample duration...")
        else:
            plt.figure()
            order = 5
            fs = 30 # TODO: change to 100 Hz
            num_samples = sensor_array.shape[0]
            time = np.linspace(0, samp_duration, num_samples, endpoint=False)
            normalized_cutoff_freq = 2 * cutoff_freq/fs
            numerator_coeffs, denominator_coeffs = butter(order, normalized_cutoff_freq)
            filtered_signal = lfilter(numerator_coeffs, denominator_coeffs, sensor_array[0:sensor_array.shape[0], i])
            print("Filtered signal for " + sensor + " " + xyz[i] + " axis" + ".")
            plt.plot(time, sensor_array[0:sensor_array.shape[0], i], 'b-', label=sensor)
            plt.plot(time, filtered_signal, 'g-', linewidth=2, label='filtered signal')
            plt.xlabel('Time (s)')
            plt.legend(loc="upper right")
            plt.title("Signal vs LP Filtered signal: " + sensor + " " + xyz[i])
            plt.savefig(get_results_directory() + "/" + sensor + "_" + xyz[i] + "_filtered.png")
            filtered_signals.append(filtered_signal) # XYZ

    x_filtered = filtered_signals[0]
    y_filtered = filtered_signals[1]
    z_filtered = filtered_signals[2]

    return x_filtered, y_filtered, z_filtered


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
    xyz = ["Time", "X", "Y", "Z"]
    plt.figure()
    # Calculate PSD using Welch's PSD Algorithm
    for i in range(len(xyz)):
        if (i!=0):
            sensor_overlap = (sensor_array.shape[1]/2)
            f_sample = 30 # 30 Hz sampling freq TODO: change to 100 Hz for final experiment
            freq, p_density = welch(sensor_array[0:sensor_array.shape[0], i], f_sample, window='hann', noverlap=sensor_overlap)
            print("Calculated PSD for " + sensor + " " + xyz[i] + " axis" + ".")

            # Plot the PSD and save to results
            plt.semilogy(freq, p_density, label=xyz[i])
            plt.ylim([0.5e-3, 1])
            plt.xlabel('Frequency (Hz)')
            plt.legend(loc="upper right")
            plt.ylabel('Power Spectral Density [dB]')
            plt.title("PSD Estimation for " + sensor)
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
    accel_time_col = 18
    accel_z_col = 22
    print("Generating iPhone Accel Array...")
    iphone_accel = gen_sensor_array(accel_time_col, accel_z_col, iphone_csv_array)    

    # gyro
    gyro_time_col = 22
    gyro_z_col = 26
    print("Generating iPhone Gyro Array...")
    iphone_gyro = gen_sensor_array(gyro_time_col, gyro_z_col, iphone_csv_array)  

    ##################### create accel, gyro, magnetometer numpy arrays: iWatch #####################
    # accel
    print("Generating iWatch Accel Array...")
    iwatch_accel = gen_sensor_array(accel_time_col, accel_z_col, iwatch_csv_array)

    # gyro
    print("Generating iWatch Gyro Array...")
    iwatch_gyro = gen_sensor_array(gyro_time_col, gyro_z_col, iwatch_csv_array)  

    # finish
    return iphone_accel, iphone_gyro, iwatch_accel, iwatch_gyro

# test
def main():
    # Turn CSV into numpy matrices
    iphone_accel, iphone_gyro, iwatch_accel, iwatch_gyro = make_matrices()

    # Plot PSD using Welch's method for cutoff freqs
    power_spectral_density(iphone_accel, "iPhoneAccelerometer")
    power_spectral_density(iphone_gyro, "iPhoneGyroscope")
    power_spectral_density(iwatch_accel, "iWatchAccelerometer")
    power_spectral_density(iwatch_gyro, "iWatchGyroscope")

    iphone_accel_x_filtered, iphone_accel_y_filtered, iphone_accel_z_filtered = low_pass_filter(iphone_accel, "iPhoneAccelerometer", 10)
    iphone_gyro_x_filtered, iphone_gyro_y_filtered, iphone_gyro_z_filtered = low_pass_filter(iphone_gyro, "iPhoneGyroscope", 13)
    iwatch_accel_x_filtered, iwatch_accel_y_filtered, iwatch_accel_z_filtered = low_pass_filter(iwatch_accel, "iWatchAccelerometer", 10)
    iwatch_gyro_x_filtered, iwatch_gyro_y_filtered, iwatch_gyro_z_filtered = low_pass_filter(iwatch_gyro, "iWatchGyroscope", 6)
    print("Finished.")

if __name__ == "__main__": 
    main()

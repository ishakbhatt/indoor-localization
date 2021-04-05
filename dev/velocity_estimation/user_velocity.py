#!/usr/bin/env python
###########################################
##                                       ##
##  USER VELOCITY ESTIMATION ALGORITHM   ##
##                                       ##
###########################################

# libraries & functions
import os
import csv
import numpy as np
from numpy import genfromtxt
from scipy.signal import welch
from scipy.signal import butter
from scipy.signal import lfilter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime, timezone

###########################################
##                                       ##
##         DEEP NEURAL NETWORK           ##
##                                       ##
###########################################

def deep_neural_network(horizontal_accel, vertical_accel, horizontal_gyro, vertical_gyro):
    '''
    Apply a DCNN to estimate user velocity.
    '''
    # https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
    # um what is a dcnn pls send help


###########################################
##                                       ##
##      COORDINATE SYSTEM ALIGNMENT      ##
##                                       ##
###########################################

def coordinate_sys_alignment(sensor_x, sensor_y, sensor_z):
    '''
    Transform accel & gyro data into orientation.
    '''
    # derive gravity component on each axis
    vector_x = np.average(sensor_x)
    vector_y = np.average(sensor_y)
    vector_z = np.average(sensor_z)
    vec = [vector_x, vector_y, vector_z]

    # construct dynamic component of the sensor
    d = [(sensor_x-vector_x), (sensor_y-vector_y), (sensor_z-vector_z)]

    #for i in range(len(vec)):
        # take norm of vector
    v_sq = [vec[0]**2, vec[1]**2, vec[2]**2]
    v_norm = np.sqrt(sum(v_sq))

    # project dynamic component onto vector
    vertical = ((np.dot(d, vec)/v_norm**2)*vec)

    # calculate horizontal component
    horizontal = (d - vec)

    return vertical, horizontal

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
            samp_duration = sensor_array[sensor_array.shape[0]-1, 0]-sensor_array[0, 0]
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

def new_gen_sensor_array(in_cols, device_csv_array):
    """
    Create NumPy array given columns to include.
    """
    return device_csv_array[:, in_cols]

def gen_sensor_array(time_col, z_col, device_csv_array):
    """
    Create NumPy array given start and end rows and columns.
    """
    num_rows = len(device_csv_array)
    return device_csv_array[:, time_col: z_col]

def genfromtxt_with_unix_convert(data_dir, is_converted):
    """
    Read a csv file into a NumPy array.
    Converts timestamps (in 0th column) to unix time if is_converted is False.
    """
    data = []
    with open(data_dir, 'r') as imu_data:
        csv_reader = csv.reader(imu_data, delimiter=',')
        line_count = 0
        if not is_converted:
            for row in csv_reader:
                if not line_count:
                    line_count+= 1
                    continue
                date = row[0][0:12]
                time = row[0][11:23]
                y = int(date[0:4])
                mo = int(date[5:7])
                day = int(date[8:10])
                h = int(time[0:2])
                m = int(time[3:5])
                s = int(time[6:8])
                ms = int(time[9:12])
                d = datetime(y, mo, day,h,m,s,ms,tzinfo=timezone.utc)
                ts = datetime(y, mo, day,h,m,s,ms,tzinfo=timezone.utc).timestamp()
                print(y, mo, day, h, m, s, d, ts)
                row[0] = ts
                data.append(row)
                line_count+= 1
        else:
            for row in csv_reader:
                if not line_count:
                    line_count+= 1
                    continue
                data.append(row)
                line_count+= 1
    return np.asarray(data)

def make_matrices():
    """
    Format IMU CSV data into Numpy matrices.
    """
    data_dir = get_data_directory()

    print("Generating iPhone Array...")
    iphone_csv_array = genfromtxt_with_unix_convert(os.path.join(data_dir, 'iphoneIMU.csv'), True)
    print("Generating iWatch Array...")
    iwatch_csv_array = genfromtxt_with_unix_convert(os.path.join(data_dir, 'iWatchIMU.csv'), False)

    ##################### create accel, gyro, magnetometer numpy arrays: iPhone #####################
    # accel
    iphone_accel_cols = [0, 19, 20, 21]
    print("Generating iPhone Accel Array...")
    iphone_accel = new_gen_sensor_array(iphone_accel_cols, iphone_csv_array) # extract cols
    iphone_accel = iphone_accel.astype('float64') # convert all values to floats

    # gyro
    iphone_gyro_cols = [0, 23, 24, 25]
    print("Generating iPhone Gyro Array...")
    iphone_gyro = new_gen_sensor_array(iphone_gyro_cols, iphone_csv_array)
    iphone_gyro = iphone_gyro.astype('float64')

    ##################### create accel, and gyro numpy arrays: iWatch #####################
    # accel
    iwatch_accel_cols = [0, 11, 12, 13]
    print("Generating iWatch Accel Array...")
    iwatch_accel = new_gen_sensor_array(iwatch_accel_cols, iwatch_csv_array)
    iwatch_accel = iwatch_accel.astype('float64')

    # gyro
    iwatch_gyro_cols = [0, 18, 19, 20]
    print("Generating iWatch Gyro Array...")
    iwatch_gyro = new_gen_sensor_array(iwatch_gyro_cols, iwatch_csv_array)
    iwatch_gyro = iwatch_gyro.astype('float64')

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

    # Low Pass filter accel and gyro data
    iphone_accel_x_filtered, iphone_accel_y_filtered, iphone_accel_z_filtered = low_pass_filter(iphone_accel, "iPhoneAccelerometer", 8)
    iphone_gyro_x_filtered, iphone_gyro_y_filtered, iphone_gyro_z_filtered = low_pass_filter(iphone_gyro, "iPhoneGyroscope", 10)
    iwatch_accel_x_filtered, iwatch_accel_y_filtered, iwatch_accel_z_filtered = low_pass_filter(iwatch_accel, "iWatchAccelerometer", 6)
    iwatch_gyro_x_filtered, iwatch_gyro_y_filtered, iwatch_gyro_z_filtered = low_pass_filter(iwatch_gyro, "iWatchGyroscope", 14)

    iphone_accel_horizonal, iphone_accel_vertical = coordinate_sys_alignment(iphone_accel_x_filtered, iphone_accel_y_filtered, iphone_accel_z_filtered)
    iphone_gyro_horizonal, iphone_gyro_vertical = coordinate_sys_alignment(iphone_gyro_x_filtered, iphone_gyro_y_filtered, iphone_gyro_z_filtered)
    iwatch_accel_horizonal, iwatch_accel_vertical = coordinate_sys_alignment(iwatch_accel_x_filtered, iwatch_accel_y_filtered, iwatch_accel_z_filtered)
    iwatch_gyro_horizonal, iwatch_gyro_vertical = coordinate_sys_alignment(iwatch_gyro_x_filtered, iwatch_gyro_y_filtered, iwatch_gyro_z_filtered)

    user_velocity_iphone = deep_neural_network(iphone_accel_horizonal, iphone_accel_vertical, iphone_gyro_horizonal, iphone_gyro_vertical)
    user_velocity_iwatch = deep_neural_network(iwatch_accel_horizonal, iwatch_accel_vertical, iwatch_gyro_horizonal, iwatch_gyro_vertical)
    final_velocity = (user_velocity_iphone + user_velocity_iwatch) / 2
    print("User Velocity Estimation: " + final_velocity)

    print("Finished.")

if __name__ == "__main__": 
    main()

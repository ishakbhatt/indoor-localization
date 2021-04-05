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
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

###########################################
##                                       ##
##         DEEP NEURAL NETWORK           ##
##                                       ##
###########################################

def magnitude(horizontal, vertical):
    '''
    Take magnitude of each row of horizontal and vertical components.
    '''
    horizontal_magnitude = np.empty([horizontal.shape[0], 1])
    vertical_magnitude = np.empty([vertical.shape[1], 1])
    for i in range(horizontal.shape[0]):
        horizontal_magnitude[i] = np.linalg.norm(horizontal[i])
        vertical_magnitude[i] = np.linalg.norm(vertical[i])
    return horizontal_magnitude, vertical_magnitude

def construct_images(accel_horizontal_mag, accel_vertical_mag, gyro_horizontal_mag, gyro_vertical_mag):
    '''
    Construct 45x4x1 images of the horizontal and vertical magnitudes of accel & gyro sensors.
    '''
    images = []
    image_num_rows = 45
    # construct combined image
    combined = np.hstack(accel_horizontal_mag, accel_vertical_mag, gyro_horizontal_mag, gyro_vertical_mag)

    num_equal_images = (combined.shape[0]-(combined.shape[0] % image_num_rows))/image_num_rows
    start = 0
    for i in range(num_equal_images):
        image = combined[start:start+image_num_rows, :]
        images.append(image)
        start = start + image_num_rows
    # Combined image not guaranteed to be multiple of image_num_rows, so append last image (which will likely be smaller)
    images.append(combined[start:start+(combined.shape[0] % image_num_rows)])
    return images


def dcnn_convolutional_layer(images):
    '''
    Apply Convolutional layer to images.
    '''
    conv_16 = []
    conv_32 = []
    conv_48 = []
    conv_64 = []

    # apply 4 convolutional layers to each image
    for image in range(len(images)):

        # convert image to tensor object
        tf_image = tf.convert_to_tensor(image)
        # employ (16, 32, 48, 64) conv filters with kernel size of 2x2 and stride of 1 (default)
        # Convolutional layer 1: 16 filters
        convolved_16 = keras.layers.Conv2D(16, (2, 2))(tf_image)
        # Convolutional layer 2: 32 filters
        convolved_32 = keras.layers.Conv2D(32, (2, 2))(tf_image)
        # Convolutional layer 3: 48 filters
        convolved_48 = keras.layers.Conv2D(48, (2, 2))(tf_image)
        # Convolutional layer 4: 64 filters
        convolved_64 = keras.layers.Conv2D(64, (2, 2))(tf_image)

        conv_16.append(convolved_16)
        conv_32.append(convolved_32)
        conv_48.append(convolved_48)
        conv_64.append(convolved_64)

    # TODO: "calculate the dot product of the weights of the filter and the input to optimize..."
    return conv_16, conv_32, conv_48, conv_64

def batch_norm(convolved_16, convolved_32, convolved_48, convolved_64):
    '''
    Apply batch normalization layer between convolutional layer and ReLU layer.
    Batch normalization automatically standardizes inputs to a layer in the neural network.
    https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
    - accelerates training process of a neural network and (sometimes) performance of model via regularization
    - inputs will have mean 0 and std dev of 1
    '''

    bn = keras.layers.BatchNormalization()



def deep_neural_network(horizontal_accel, vertical_accel, horizontal_gyro, vertical_gyro):
    '''
    Apply a DCNN to estimate user velocity.
    https://keras.io/getting_started/intro_to_keras_for_engineers/
    '''

    ##################### CONSTRUCT THE IMAGES #####################
    # magnitude
    accel_horizontal_mag, accel_vertical_mag = magnitude(horizontal_accel, vertical_accel)
    gyro_horizontal_mag, gyro_vertical_mag = magnitude(horizontal_gyro, vertical_gyro)
    images = construct_images(accel_horizontal_mag, accel_vertical_mag, gyro_horizontal_mag, gyro_vertical_mag)

    ##################### BUILD THE MODEL #####################

    # Create Keras model
    model = keras.Sequential()

    # CASCADED LAYER 1
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))
    # Max Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    # CASCADED LAYER 2
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))
    # Max Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    # CASCADED LAYER 3
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))
    # Max Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    # CASCADED LAYER 4 (Maxpooling excluded in last group)
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))

    # Dropout layer: alleviate overfitting
    model.add(keras.layers.Dropout(.2))

    # Fully connected layer: compute class scores fed into regression layer
    model.add(keras.layers.Dense(1))

    # Regression layer
    model.add(keras.wrappers.scikit_learn.KerasRegressor())
    # Summary
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(loss='mean_squared_error')

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
    v_sq = [vec[0]**2, vec[1]**2, vec[2]**2]
    v_norm = np.sqrt(sum(v_sq))

    # horizontal and vertical components
    vertical = []
    horizontal = []
    for i in range(d[0].shape[0]):
        d_vec = [d[0][i], d[1][i], d[2][i]]
        # project dynamic component onto vector
        print("Projecting dynamic component from sample " + str(i) + " onto vertical...")
        vertical_current = [(np.dot(d_vec, vec)/v_norm**2)*vec[0], (np.dot(d_vec, vec)/v_norm**2)*vec[1], (np.dot(d_vec, vec)/v_norm**2)*vec[2]]
        vertical.append(vertical_current)

        # calculate horizontal component
        print("Computing horizontal component...")
        horizontal_current = [d_vec[0]-vec[0], d_vec[1]-vec[1], d_vec[2]-vec[2]]
        horizontal.append(horizontal_current)

    print("Converting horizontal and vertical components into numpy arrays...")
    vertical = np.array(vertical)
    horizontal = np.array(horizontal)
    print("Finished Coordinate System Alignment.")
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
            samp_duration = sensor_array[:, 0]-sensor_array[0, 0]
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
            freq, p_density = welch(sensor_array[:, i], f_sample, window='hann', noverlap=sensor_overlap)
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
    return device_csv_array[:, time_col: z_col]

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

    ##################### create accel, and gyro numpy arrays: iWatch #####################
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
    '''
    user_velocity_iphone = deep_neural_network(iphone_accel_horizonal, iphone_accel_vertical, iphone_gyro_horizonal, iphone_gyro_vertical)
    user_velocity_iwatch = deep_neural_network(iwatch_accel_horizonal, iwatch_accel_vertical, iwatch_gyro_horizonal, iwatch_gyro_vertical)
    final_velocity = (user_velocity_iphone + user_velocity_iwatch) / 2
    print("User Velocity Estimation: " + final_velocity)'''

    print("Finished.")

if __name__ == "__main__": 
    main()

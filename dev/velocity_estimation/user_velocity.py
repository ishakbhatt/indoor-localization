###########################################
##                                       ##
##  USER VELOCITY ESTIMATION ALGORITHM   ##
##                                       ##
###########################################

# libraries & functions
import os
import csv
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal import butter
from scipy.signal import lfilter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime, timezone
import tensorflow as tf
import parse
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
    vertical_magnitude = np.empty([vertical.shape[0], 1])
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
    combined2 = np.concatenate((accel_horizontal_mag, accel_vertical_mag), axis=1)
    combined1 = np.concatenate((gyro_horizontal_mag, gyro_vertical_mag), axis=1)
    combined = np.concatenate((combined2, combined1), axis=1)

    num_equal_images = (combined.shape[0]-(combined.shape[0] % image_num_rows))/image_num_rows
    start = 0
    for i in range(int(num_equal_images)):
        image = combined[start:start+image_num_rows, :]
        images.append(image)
        start = start + image_num_rows
    # Combined image not guaranteed to be multiple of image_num_rows, so append last image (which will likely be smaller)
    images.append(combined[start:start+(combined.shape[0] % image_num_rows)])
    return images

def concat_train_imu(array_list):
    '''concatenates training arrays'''
    sub_array1 = np.concatenate((array_list[0], array_list[1]), axis=0)
    sub_array2 = np.concatenate((sub_array1, array_list[2]), axis=0)
    sub_array3 = np.concatenate((sub_array2, array_list[3]), axis=0)
    sub_array4 = np.concatenate((sub_array3, array_list[4]), axis=0)
    sub_array5 = np.concatenate((sub_array4, array_list[5]), axis=0)
    final = np.concatenate((sub_array5, array_list[6]), axis=0)
    return final

def concat_test_imu(array_list):
    '''concatenates testing IMU arrays'''
    sub_array1 = np.concatenate((array_list[0], array_list[1]), axis=0)
    final = np.concatenate((sub_array1, array_list[2]), axis=0)
    return final

def concat_train_vel(array_list):
    '''concatenates training velocity arrays'''
    sub_array = np.concatenate((array_list[0][:, 4], array_list[1][:, 4]), axis=0)
    sub_array2 = np.concatenate((sub_array, array_list[2][:, 4]), axis=0)
    sub_array3 = np.concatenate((sub_array2, array_list[3][:, 4]), axis=0)
    sub_array4 = np.concatenate((sub_array3, array_list[4][:, 4]), axis=0)
    sub_array5 = np.concatenate((sub_array4, array_list[5][:, 4]), axis=0)
    final = np.concatenate((sub_array5, array_list[6][:, 4]), axis=0)
    return final

def concat_test_vel(array_list):
    '''concatenates testing IMU arrays'''
    sub_array1 = np.concatenate((array_list[0][:, 4], array_list[1][:, 4]), axis=0)
    final = np.concatenate((sub_array1, array_list[2][:, 4]), axis=0)
    return final

def deep_neural_network(horizontal_accel_train, vertical_accel_train, horizontal_gyro_train, vertical_gyro_train, velocity_train,
                        horizontal_accel_test, vertical_accel_test, horizontal_gyro_test, vertical_gyro_test, velocity_test): # TODO: update with train and test
    '''
    Apply a DCNN to estimate user velocity.
    https://keras.io/getting_started/intro_to_keras_for_engineers/
    '''

    ##################### CONSTRUCT THE IMAGES #####################
    # magnitude
    accel_horizontal_mag_train, accel_vertical_mag_train = magnitude(horizontal_accel_train, vertical_accel_train)
    gyro_horizontal_mag_train, gyro_vertical_mag_train = magnitude(horizontal_gyro_train, vertical_gyro_train)
    images_train = construct_images(accel_horizontal_mag_train, accel_vertical_mag_train, gyro_horizontal_mag_train, gyro_vertical_mag_train)

    accel_horizontal_mag_test, accel_vertical_mag_test = magnitude(horizontal_accel_test, vertical_accel_test)
    gyro_horizontal_mag_test, gyro_vertical_mag_test = magnitude(horizontal_gyro_test, vertical_gyro_test)
    images_test = construct_images(accel_horizontal_mag_test, accel_vertical_mag_test, gyro_horizontal_mag_test, gyro_vertical_mag_test)

    # combined TODO: decide combined or constructed images
    sub_array1_train = np.concatenate((accel_horizontal_mag_train, accel_vertical_mag_train), axis=1)
    sub_array2_train = np.concatenate((gyro_horizontal_mag_train, accel_vertical_mag_train), axis=1)
    combined_train = np.concatenate((sub_array1_train, sub_array2_train), axis=1)

    sub_array1_test = np.concatenate((accel_horizontal_mag_test, accel_vertical_mag_test), axis=1)
    sub_array2_test = np.concatenate((gyro_horizontal_mag_test, accel_vertical_mag_test), axis=1)
    combined_test = np.concatenate((sub_array1_test, sub_array2_test), axis=1)


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
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # CASCADED LAYER 2
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))
    # Max Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # CASCADED LAYER 3
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(2, 2)))
    # batch normalization layer to lower sensitivity to network initialization
    model.add(keras.layers.BatchNormalization())
    # ReLU activation layer
    model.add(keras.layers.Activation('relu'))
    # Max Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

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
    #model.add(keras.wrappers.scikit_learn.KerasRegressor())

    # compile model using mse as a measure of model performance
    model.compile(loss='mean_squared_error')

    # fit the model (Train) # TODO: tune epoch and batch_size
    model.fit(combined_train, velocity_train, validation_data=(combined_train, velocity_train), epochs=1, batch_size=combined_train.shape[0])

    # Summary
    model.summary()

    # evaluate the model
    print("Evaluate on test data")
    results = model.evaluate(combined_test, velocity_test, batch_size=128) # todo: fix
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(combined_test[:3])
    print("predictions shape:", predictions.shape)

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
    xyz = ["Time", "X", "Y", "Z", "velocity"]
    filtered_signals = []

    for i in range(len(xyz)):
        if i == 0:
            samp_duration = (sensor_array[:, 0]-sensor_array[0, 0])[sensor_array.shape[0]-1]
            print("Calculating sample duration...")
        elif i!=0 and i!=4:
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
        if (i!=0 and i!=4):
            sensor_overlap = (sensor_array.shape[1]/2)
            f_sample = 30 # 30 Hz sampling freq
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

def get_data_directory(data):
    """
    Return specified repository in data directory.
    """
    # change directory
    os.chdir('../../data/' + data + '/')
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
                if line_count!=1:
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
    data_dir = get_data_directory('initial')

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

    # Parse time, accel xyz, gyro xyz, velocity into numpy arrays
    iphone_accel_train = []
    iphone_gyro_train = []
    iwatch_accel_train = []
    iwatch_gyro_train = []

    iphone_accel_test = []
    iphone_gyro_test = []
    iwatch_accel_test = []
    iwatch_gyro_test = []

    for i in range(8):
        if i != 0:
            phone_a_train, phone_g_train = parse.parse_dcnn_data('train', 'iphone', i)
            watch_a_train, watch_g_train = parse.parse_dcnn_data('train', 'watch', i)
            iphone_accel_train.append(phone_a_train)
            iphone_gyro_train.append(phone_g_train)
            iwatch_accel_train.append(watch_a_train)
            iwatch_gyro_train.append(watch_g_train)

    for i in range(4):
        if i != 0:
            phone_a_test, phone_g_test = parse.parse_dcnn_data('test', 'iphone', i)
            watch_a_test, watch_g_test = parse.parse_dcnn_data('test', 'watch', i)
            iphone_accel_test.append(phone_a_test)
            iphone_gyro_test.append(phone_g_test)
            iwatch_accel_test.append(watch_a_test)
            iwatch_gyro_test.append(watch_g_test)

    # Plot PSD using Welch's method for cutoff freqs
    for i in range(7):
        iphone_train_accel_sensor = "iPhoneAccelTrain" + str(i+1)
        iphone_train_gyro_sensor = "iPhoneGyroTrain" + str(i+1)
        iwatch_train_accel_sensor = "iWatchAccelTrain" + str(i+1)
        iwatch_train_gyro_sensor = "iWatchGyroTrain" + str(i+1)
        power_spectral_density(iphone_accel_train[i], iphone_train_accel_sensor)
        power_spectral_density(iphone_gyro_train[i], iphone_train_gyro_sensor)
        power_spectral_density(iwatch_accel_train[i], iwatch_train_accel_sensor)
        power_spectral_density(iwatch_gyro_train[i], iwatch_train_gyro_sensor)

    for i in range(3):
        iphone_test_accel_sensor = "iPhoneAccelTest" + str(i+1)
        iphone_test_gyro_sensor = "iPhoneGyroTest" + str(i+1)
        iwatch_test_accel_sensor = "iWatchAccelTest" + str(i+1)
        iwatch_test_gyro_sensor = "iWatchGyroTest" + str(i+1)
        power_spectral_density(iphone_accel_test[i], iphone_test_accel_sensor)
        power_spectral_density(iphone_gyro_test[i], iphone_test_gyro_sensor)
        power_spectral_density(iwatch_accel_test[i], iwatch_test_accel_sensor)
        power_spectral_density(iwatch_gyro_test[i], iwatch_test_gyro_sensor)

    # parse in LPF cutoff frequencies
    train_data = get_data_directory('train')
    test_data = get_data_directory('test')
    iphone_train_freq = (pd.read_csv(os.path.join(train_data, 'iphone_train_cutoff_freq.csv'), sep=',',header=0)).to_numpy()
    watch_train_freq = (pd.read_csv(os.path.join(train_data, 'watch_train_cutoff_freq.csv'), sep=',', header=0)).to_numpy()
    iphone_test_freq = (pd.read_csv(os.path.join(test_data, 'iphone_test_cutoff_freq.csv'), sep=',', header=0)).to_numpy()
    watch_test_freq = (pd.read_csv(os.path.join(test_data, 'watch_test_cutoff_freq.csv'), sep=',', header=0)).to_numpy()

    # remove extra column with accel/gyro names
    iphone_train_freq = iphone_train_freq[:, 0:7]
    watch_train_freq = watch_train_freq[:, 0:7]
    iphone_test_freq = iphone_test_freq[:, 0:3]
    watch_test_freq = watch_test_freq[:, 0:3]

    # send LPF'd data into list of numpy arrays
    iphone_accel_filtered_train = []  # inner list: x, y, z outer list: train #
    iphone_gyro_filtered_train = []
    iwatch_accel_filtered_train = []
    iwatch_gyro_filtered_train = []

    iphone_accel_filtered_test = []  # inner list: x, y, z outer list: test #
    iphone_gyro_filtered_test = []
    iwatch_accel_filtered_test = []
    iwatch_gyro_filtered_test = []

    # Low Pass filter accel and gyro data: Train
    for i in range(7):
        iphone_train_accel_sensor = "iPhoneAccelTrain" + str(i+1)
        iphone_train_gyro_sensor = "iPhoneGyroTrain" + str(i+1)
        iwatch_train_accel_sensor = "iWatchAccelTrain" + str(i+1)
        iwatch_train_gyro_sensor = "iWatchGyroTrain" + str(i+1)

        # LPF
        x, y, z = low_pass_filter(iphone_accel_train[i], iphone_train_accel_sensor, iphone_train_freq[0][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iphone_accel_filtered_train.append(temp_list)

        x, y, z = low_pass_filter(iphone_gyro_train[i], iphone_train_gyro_sensor, iphone_train_freq[1][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iphone_gyro_filtered_train.append(temp_list)

        x, y, z = low_pass_filter(iwatch_accel_train[i], iwatch_train_accel_sensor, watch_train_freq[0][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iwatch_accel_filtered_train.append(temp_list)

        x, y, z = low_pass_filter(iwatch_gyro_train[i], iwatch_train_gyro_sensor, watch_train_freq[1][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iwatch_gyro_filtered_train.append(temp_list)

        #iphone_accel_filtered_train[i][0], iphone_accel_filtered_train[i][1], iphone_accel_filtered_train[i][2] = low_pass_filter(iphone_accel_train[i], iphone_train_accel_sensor, iphone_train_freq[0][i])
        #iphone_gyro_filtered_train[i][0], iphone_gyro_filtered_train[i][1], iphone_gyro_filtered_train[i][2] = low_pass_filter(iphone_gyro_train[i], iphone_train_gyro_sensor, iphone_train_freq[1][i])
        #iwatch_accel_filtered_train[i][0], iwatch_accel_filtered_train[i][1], iwatch_accel_filtered_train[i][2] = low_pass_filter(iwatch_accel_train[i], iwatch_train_accel_sensor, watch_train_freq[0][i])
        #iwatch_gyro_filtered_train[i][0], iwatch_gyro_filtered_train[i][1], iwatch_gyro_filtered_train[i][2] = low_pass_filter(iwatch_gyro_train[i], iwatch_train_gyro_sensor, watch_train_freq[1][i])


    # Low Pass filter accel and gyro data: Test
    for i in range(3):
        iphone_test_accel_sensor = "iPhoneAccelTest" + str(i+1)
        iphone_test_gyro_sensor = "iPhoneGyroTest" + str(i+1)
        iwatch_test_accel_sensor = "iWatchAccelTest" + str(i+1)
        iwatch_test_gyro_sensor = "iWatchGyroTest" + str(i+1)

        # LPF
        x, y, z = low_pass_filter(iphone_accel_test[i], iphone_test_accel_sensor, iphone_train_freq[0][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iphone_accel_filtered_test.append(temp_list)

        x, y, z = low_pass_filter(iphone_gyro_test[i], iphone_test_gyro_sensor, iphone_test_freq[1][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iphone_gyro_filtered_test.append(temp_list)

        x, y, z = low_pass_filter(iwatch_accel_test[i], iwatch_test_accel_sensor, watch_test_freq[0][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iwatch_accel_filtered_test.append(temp_list)

        x, y, z = low_pass_filter(iwatch_gyro_test[i], iwatch_test_gyro_sensor, watch_test_freq[1][i])
        temp_list = []
        temp_list.extend([x, y, z])
        iwatch_gyro_filtered_test.append(temp_list)

        #iphone_accel_filtered_test[i][0], iphone_accel_filtered_test[i][1], iphone_accel_filtered_test[i][2] = low_pass_filter(iphone_accel_test[i], iphone_test_accel_sensor, iphone_test_freq[0][i])
        ##iphone_gyro_filtered_test[i][0], iphone_gyro_filtered_test[i][1], iphone_gyro_filtered_test[i][2] = low_pass_filter(iphone_gyro_test[i], iphone_test_gyro_sensor, iphone_test_freq[1][i])
        #iwatch_accel_filtered_test[i][0], iwatch_accel_filtered_test[i][1], iwatch_accel_filtered_test[i][2] = low_pass_filter(iphone_accel_test[i], iwatch_test_accel_sensor, watch_test_freq[0][i])
        #iwatch_gyro_filtered_test[i][0], iwatch_gyro_filtered_test[i][1], iwatch_gyro_filtered_test[i][2] = low_pass_filter(iphone_gyro_test[i], iwatch_test_gyro_sensor, watch_test_freq[1][i])

    # Coordinate System Alignment
    iphone_accel_horizontal_train = []
    iphone_accel_vertical_train = []
    iphone_gyro_vertical_train = []
    iphone_gyro_horizontal_train =[]
    iwatch_accel_horizontal_train = []
    iwatch_accel_vertical_train = []
    iwatch_gyro_vertical_train = []
    iwatch_gyro_horizontal_train = []

    iphone_accel_horizontal_test =[]
    iphone_accel_vertical_test = []
    iphone_gyro_vertical_test = []
    iphone_gyro_horizontal_test = []
    iwatch_accel_horizontal_test =[]
    iwatch_accel_vertical_test = []
    iwatch_gyro_vertical_test = []
    iwatch_gyro_horizontal_test = []

    for i in range(7):  # Coordinate System Alignment: Training Data
        h, v = coordinate_sys_alignment(
            iphone_accel_filtered_train[i][0], iphone_accel_filtered_train[i][1], iphone_accel_filtered_train[i][2])
        iphone_accel_horizontal_train.append(h)
        iphone_accel_vertical_train.append(v)

        h, v = coordinate_sys_alignment(
            iphone_gyro_filtered_train[i][0], iphone_gyro_filtered_train[i][1], iphone_gyro_filtered_train[i][2])
        iphone_gyro_horizontal_train.append(h)
        iphone_gyro_vertical_train.append(v)

        h, v = coordinate_sys_alignment(
            iwatch_accel_filtered_train[i][0], iwatch_accel_filtered_train[i][1], iwatch_accel_filtered_train[i][2])
        iwatch_accel_horizontal_train.append(h)
        iwatch_accel_vertical_train.append(v)

        h, v = coordinate_sys_alignment(
            iwatch_gyro_filtered_train[i][0], iwatch_gyro_filtered_train[i][1], iwatch_gyro_filtered_train[i][2])
        iwatch_gyro_horizontal_train.append(h)
        iwatch_gyro_vertical_train.append(v)

        #iphone_accel_horizontal_train[i], iphone_accel_vertical_train[i] = coordinate_sys_alignment(
        #    iphone_accel_filtered_train[i][0], iphone_accel_filtered_train[i][1], iphone_accel_filtered_train[i][2])
        #iphone_gyro_horizontal_train[i], iphone_gyro_vertical_train[i] = coordinate_sys_alignment(
        #    iphone_gyro_filtered_train[i][0], iphone_gyro_filtered_train[i][1], iphone_gyro_filtered_train[i][2])
        #iwatch_accel_horizontal_train[i], iwatch_accel_vertical_train[i] = coordinate_sys_alignment(
        #    iwatch_accel_filtered_train[i][0], iwatch_accel_filtered_train[i][1], iwatch_accel_filtered_train[i][2])
        #iwatch_gyro_horizontal_train[i], iwatch_gyro_vertical_train[i] = coordinate_sys_alignment(
        #    iwatch_gyro_filtered_train[i][0], iwatch_gyro_filtered_train[i][1], iwatch_gyro_filtered_train[i][2])

    for i in range(3):  # Coordinate System Alignment: Test Data
        h, v = coordinate_sys_alignment(
            iphone_accel_filtered_test[i][0], iphone_accel_filtered_test[i][1], iphone_accel_filtered_test[i][2])
        iphone_accel_horizontal_test.append(h)
        iphone_accel_vertical_test.append(v)

        h, v = coordinate_sys_alignment(
            iphone_gyro_filtered_test[i][0], iphone_gyro_filtered_test[i][1], iphone_gyro_filtered_test[i][2])
        iphone_gyro_horizontal_test.append(h)
        iphone_gyro_vertical_test.append(v)

        h, v = coordinate_sys_alignment(
            iwatch_accel_filtered_test[i][0], iwatch_accel_filtered_test[i][1], iwatch_accel_filtered_test[i][2])
        iwatch_accel_horizontal_test.append(h)
        iwatch_accel_vertical_test.append(v)

        h, v = coordinate_sys_alignment(
            iwatch_gyro_filtered_test[i][0], iwatch_gyro_filtered_test[i][1], iwatch_gyro_filtered_test[i][2])
        iwatch_gyro_horizontal_test.append(h)
        iwatch_gyro_vertical_test.append(v)

    # Combine training data into one numpy array
    iphone_accel_h_train = np.empty([iphone_accel_horizontal_train[0].shape[0], 3])
    iphone_accel_v_train = np.empty([iphone_accel_vertical_train[0].shape[1], 3])
    iphone_gyro_h_train = np.empty([iphone_accel_horizontal_train[0].shape[0], 3])
    iphone_gyro_v_train = np.empty([iphone_accel_vertical_train[0].shape[1], 3])

    iwatch_accel_h_train = np.empty([iwatch_accel_horizontal_train[0].shape[0], 3])
    iwatch_accel_v_train = np.empty([iwatch_accel_vertical_train[0].shape[1], 3])
    iwatch_gyro_h_train = np.empty([iwatch_accel_horizontal_train[0].shape[0], 3])
    iwatch_gyro_v_train = np.empty([iwatch_accel_vertical_train[0].shape[1], 3])

    iphone_accel_h_test = np.empty([iphone_accel_horizontal_test[0].shape[0], 3])
    iphone_accel_v_test = np.empty([iphone_accel_vertical_test[0].shape[1], 3])
    iphone_gyro_h_test = np.empty([iphone_accel_horizontal_test[0].shape[0], 3])
    iphone_gyro_v_test = np.empty([iphone_accel_vertical_test[0].shape[1], 3])

    iwatch_accel_h_test = np.empty([iwatch_accel_horizontal_test[0].shape[0], 3])
    iwatch_accel_v_test = np.empty([iwatch_accel_vertical_test[0].shape[1], 3])
    iwatch_gyro_h_test = np.empty([iwatch_accel_horizontal_test[0].shape[0], 3])
    iwatch_gyro_v_test = np.empty([iwatch_accel_vertical_test[0].shape[1], 3])

    iphone_accel_h_train = concat_train_imu(iphone_accel_horizontal_train)
    iphone_accel_v_train = concat_train_imu(iphone_accel_vertical_train)
    iphone_gyro_h_train = concat_train_imu(iphone_gyro_horizontal_train)
    iphone_gyro_v_train = concat_train_imu(iphone_gyro_vertical_train)
    iwatch_accel_h_train = concat_train_imu(iwatch_accel_horizontal_train)
    iwatch_accel_v_train = concat_train_imu(iwatch_accel_vertical_train)
    iwatch_gyro_h_train = concat_train_imu(iwatch_gyro_horizontal_train)
    iwatch_gyro_v_train = concat_train_imu(iwatch_gyro_vertical_train)

    iphone_accel_h_test = concat_test_imu(iphone_accel_horizontal_test)
    iphone_accel_v_test = concat_test_imu(iphone_accel_vertical_test)
    iphone_gyro_h_test = concat_test_imu(iphone_gyro_horizontal_test)
    iphone_gyro_v_test = concat_test_imu(iphone_gyro_vertical_test)
    iwatch_accel_h_test = concat_test_imu(iwatch_accel_horizontal_test)
    iwatch_accel_v_test = concat_test_imu(iwatch_accel_vertical_test)
    iwatch_gyro_h_test = concat_test_imu(iwatch_gyro_horizontal_test)
    iwatch_gyro_v_test = concat_test_imu(iwatch_gyro_vertical_test)

    iphone_velocity_train = concat_train_vel(iphone_accel_train)
    iphone_velocity_test = concat_test_vel(iphone_accel_test)
    iwatch_velocity_train = concat_train_vel(iwatch_accel_train)
    iwatch_velocity_test = concat_test_vel(iwatch_accel_train)

    # velocities


    deep_neural_network(iphone_accel_h_train, iphone_accel_v_train, iphone_gyro_h_train, iphone_gyro_v_train,
                        iphone_velocity_train, iphone_accel_h_test, iphone_accel_v_test, iphone_gyro_h_test,
                        iphone_gyro_v_test, iphone_velocity_test)

    '''deep_neural_network(iwatch_accel_h_train, iwatch_accel_v_train, iwatch_gyro_h_train, iwatch_gyro_v_train,
                        iwatch_velocity_train, iwatch_accel_h_test, iwatch_accel_v_test, iwatch_gyro_h_test,
                        iwatch_gyro_v_test, iwatch_velocity_test)'''
    print("Finished.")

if __name__ == "__main__": 
    main()

import os
import sys
import csv
import numpy as np
import itertools
import user_velocity as uv
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matricies import R
import scipy.integrate as integrate
import pandas as pd
import math

def parse_dcnn_data(set_type, device, set_num):
    """

    Returns 2 n x 5 numpy arrays containing data for the 
    specified data set. 
    
    accel array columns:
    - unix timestamp
    - accel x
    - accel y
    - accel z
    - velocity

    gyro array columns:
    - unix timestamp
    - gyro x
    - gyro y
    - gyro z
    - velocity

    """
    if(not ((set_type == 'train') or (set_type == 'test'))):
        print("Incorrect set type parameter. Please input 'train' or 'test' set type.")
        sys.exit()
    
    if(not ((device == 'iphone') or (device == 'watch'))):
        print("Incorrect device parameter. Please input 'iphone' or 'watch' device.")
        sys.exit()

    # init output arrays
    gyro_data = [0,0,0,0,0]
    accel_data = [0,0,0,0,0]
    
    # get correct directory
    data_dir = uv.get_data_directory(set_type)

    # get timestamp and vel
    if(set_type == 'train'):
        vel_cols = [2, 3]
    else:
        vel_cols = [4, 5]
    vel_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_vel' + str(set_num) + '.csv'), True)
    if(set_type == 'train' and set_num == 2):
        vel_arr = vel_arr[0:15, :]
    vel_arr = uv.new_gen_sensor_array(vel_cols, vel_arr)
    vel_arr = vel_arr.astype('float64')

    # get timestamps
    if(device == 'iphone'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), True)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64') - 14400

        # accel
        accel_cols = [19, 20, 21]
        accel = uv.new_gen_sensor_array(accel_cols, imu_arr) # extract cols
        accel = accel.astype('float64') # convert all values to floats
        accel = np.concatenate((timestamps, accel), axis=1)

        # gyro
        gyro_cols = [23, 24, 25]
        gyro = uv.new_gen_sensor_array(gyro_cols, imu_arr)
        gyro = gyro.astype('float64')
        gyro = np.concatenate((timestamps, gyro), axis=1)

    elif(device == 'watch'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), False)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64')

        # accel
        accel_cols = [11, 12, 13]
        accel = uv.new_gen_sensor_array(accel_cols, imu_arr)
        accel = accel.astype('float64')
        accel = np.concatenate((timestamps, accel), axis=1)

        # gyro
        gyro_cols = [18, 19, 20]
        gyro = uv.new_gen_sensor_array(gyro_cols, imu_arr)
        gyro = gyro.astype('float64')
        gyro = np.concatenate((timestamps, gyro), axis=1)

    # init loop variables
    curr_time = vel_arr[0][0]
    next_time = vel_arr[1][0]
    curr_vel = vel_arr[1][1]
    vel_it = 1
    
    # match up velocities with imu data
    for (a_row, g_row) in zip(accel, gyro):
        if(a_row[0] < curr_time): # skip until we find the start
            continue
        elif(a_row[0] > curr_time): # set velocities = curr_vel
            a_row = np.append(a_row, curr_vel)
            g_row = np.append(g_row, curr_vel)
            accel_data = np.vstack((accel_data, a_row))
            gyro_data = np.vstack((gyro_data, g_row))
            if(a_row[0] >= next_time): # set a new curr_vel
                vel_it += 1
                if(vel_it == len(vel_arr)):
                    break
                curr_time = next_time
                next_time = vel_arr[vel_it][0]
                curr_vel = vel_arr[vel_it][1]

    # remove initial row of zeros
    accel_data = np.delete(accel_data,0,0)
    gyro_data = np.delete(gyro_data,0,0)

    return accel_data, gyro_data

def parse_dcnn_data_for_pos_est(set_type, device, set_num):
    """

    Returns 2 n x 5 numpy arrays containing data for the 
    specified data set. 
    
    accel array columns:
    - unix timestamp
    - accel x
    - accel y
    - accel z
    - velocity

    gyro array columns:
    - unix timestamp
    - gyro x
    - gyro y
    - gyro z
    - velocity

    """
    if(not ((set_type == 'train') or (set_type == 'test'))):
        print("Incorrect set type parameter. Please input 'train' or 'test' set type.")
        sys.exit()
    
    if(not ((device == 'iphone') or (device == 'watch'))):
        print("Incorrect device parameter. Please input 'iphone' or 'watch' device.")
        sys.exit()

    # init output arrays
    gyro_data = [0,0,0,0,0,0,0]
    accel_data = [0,0,0,0,0,0,0]
    
    # get correct directory
    # data_dir = uv.get_data_directory(set_type)
    data_dir = "/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/data/test/"

    # get timestamp and vel
    if(set_type == 'train'):
        vel_cols = [2, 3]
    else:
        vel_cols = [4, 5, 0, 1]
    vel_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_vel' + str(set_num) + '.csv'), True)
    if(set_type == 'train' and set_num == 2):
        vel_arr = vel_arr[0:15, :]
    vel_arr = uv.new_gen_sensor_array(vel_cols, vel_arr)
    vel_arr = vel_arr.astype('float64')

    # get timestamps
    if(device == 'iphone'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), True)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64') - 14400

        # accel
        accel_cols = [19, 20, 21]
        accel = uv.new_gen_sensor_array(accel_cols, imu_arr) # extract cols
        accel = accel.astype('float64') # convert all values to floats
        accel = np.concatenate((timestamps, accel), axis=1)

        # gyro
        gyro_cols = [23, 24, 25]
        gyro = uv.new_gen_sensor_array(gyro_cols, imu_arr)
        gyro = gyro.astype('float64')
        gyro = np.concatenate((timestamps, gyro), axis=1)

    elif(device == 'watch'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), False)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64')

        # accel
        accel_cols = [11, 12, 13]
        accel = uv.new_gen_sensor_array(accel_cols, imu_arr)
        accel = accel.astype('float64')
        accel = np.concatenate((timestamps, accel), axis=1)

        # gyro
        gyro_cols = [18, 19, 20]
        gyro = uv.new_gen_sensor_array(gyro_cols, imu_arr)
        gyro = gyro.astype('float64')
        gyro = np.concatenate((timestamps, gyro), axis=1)

    # init loop variables
    curr_time = vel_arr[0][0]
    next_time = vel_arr[1][0]
    curr_vel = vel_arr[1][1]
    curr_x = vel_arr[1][2]
    curr_y = vel_arr[1][3]
    vel_it = 1
    
    # match up velocities with imu data
    for (a_row, g_row) in zip(accel, gyro):
        if(a_row[0] < curr_time): # skip until we find the start
            continue
        elif(a_row[0] > curr_time):
            if(a_row[0] >= next_time): # set a new curr_vel
                a_row = np.append(a_row, curr_vel)
                a_row = np.append(a_row, curr_x)
                a_row = np.append(a_row, curr_y)
                g_row = np.append(g_row, curr_vel)
                g_row = np.append(g_row, curr_x)
                g_row = np.append(g_row, curr_y)
                accel_data = np.vstack((accel_data, a_row))
                gyro_data = np.vstack((gyro_data, g_row))
                vel_it += 1
                if(vel_it == len(vel_arr)):
                    break
                curr_time = next_time
                next_time = vel_arr[vel_it][0]
                curr_vel = vel_arr[vel_it][1]
                curr_x = vel_arr[vel_it][2]
                curr_y = vel_arr[vel_it][3]

    # remove initial row of zeros
    accel_data = np.delete(accel_data,0,0)
    gyro_data = np.delete(gyro_data,0,0)

    return accel_data, gyro_data

def parse_rssi_data(m, d, y, set_num, threshold):
    """

    Returns an n x 7 numpy array with RSSI data for the specified set_num.

    Columns:
    - unix timestamp
    - ground truth x
    - ground truth y
    - Node 1 RSSI values
    - Node 2 RSSI values
    - Node 3 RSSI values
    - NOde 4 RSSI values

    Takes in the date the data was recorded (needed for unix
    timestamp conversion):

        m - month
        d - day
        y - year

    Takes in a threshold in seconds for assigning RSSI values at a given timestamp.

    """
    rssi_data = []
    # get correct directory
    data_dir = uv.get_data_directory(set_type)

    # read in velocity vec and extract timestamp and coordinate
    vel_cols = [4, 0, 1]
    vel_arr = uv.genfromtxt_with_unix_convert(data_dir + 'test_vel' + str(set_num) + '.csv', True)
    vel_arr = uv.new_gen_sensor_array(vel_cols, vel_arr)
    vel_arr = vel_arr.astype('float64')

    rssi_raw_data = []

    with open(data_dir + 'test_rssi' + str(set_num) + '.csv', 'r') as rssi: # read in RSSI file for the given set_num
        csv_reader = csv.reader(rssi, delimiter=',')
        next(csv_reader) # skip header
        for row in csv_reader:
            ssid = row[0]
            i = 0 # node_reading count
            while(ssid[0:4] == 'Node'):
                if(len(rssi_raw_data) < i + 1): # first reading of node values
                    # get and append timestamp
                    h = int(row[4][2:3]) + 12
                    min = int(row[4][4:6]) # extract hour + 12, minute, second
                    s = int(row[4][7:9])
                    ts = datetime(y, m, d,h, min,s,0,tzinfo=timezone.utc).timestamp()
                    empty = [0, 0, 0, 0, 0]
                    empty[0] = ts
                    rssi_raw_data.append(empty) # add a new row

                # get and append rssi_val
                node_num = int(ssid[5]) # get Node
                rssi_val = row[2][3:5] # get RSSI value (remove negative sign)
                rssi_raw_data[i][node_num] = rssi_val
                
                i += 1 # go to next timestamp
                row = next(csv_reader) # read next row
                ssid = row[0] # update ssid
    
    # get closest ground truth coordinate at given timestamp
    for v_row in vel_arr:
        for r_row in rssi_raw_data:
            if((r_row[0] >= v_row[0] - threshold) and (r_row[0] <= v_row[0] + threshold)):
                final_data = np.concatenate((v_row, r_row[1:5]), axis=0)
                final_data = final_data.tolist()
                rssi_data.append(final_data)
                break
    print(len(rssi_data), "positions could be estimated from path", set_num)
    return rssi_data

def results_no_rssi(device):
    orig_device_name = device
    results_dir = "/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/dev/velocity_estimation/imu_integration_results/"
    for path_num in range(1,4):
        accel, gyro = parse_dcnn_data_for_pos_est('test', orig_device_name, path_num)
        thetas = []
        velocities = []
        positions = []
        gt_positions = []
        base_ts = accel[0][0]
        # perform integration on gyro data to get theta vector
        for i in range(0,len(accel) - 1):
            print("-------- New Position Estimation --------")
            b = accel[i + 1][0] - base_ts
            theta_x = integrate.quad(lambda x: gyro[i][1], 0, b)[0]
            theta_y = integrate.quad(lambda x: gyro[i][2], 0, b)[0]
            theta_z = integrate.quad(lambda x: gyro[i][3], 0, b)[0]
            thetas.append([theta_x, theta_y, theta_z])
            
            # find rotation matrix
            Rot = R([theta_x, theta_y, theta_z])

            # apply rotation matrix to acceleration vector
            a_vec = np.transpose(np.asarray([[accel[i][1], accel[i][2], accel[i][3]]]))
            rot_accel = np.dot(Rot,a_vec)

            # perform integration to get velocity
            x_vel = integrate.quad(lambda x: rot_accel[0], 0, b)[0]
            y_vel = integrate.quad(lambda x: rot_accel[1], 0, b)[0]
            z_vel = integrate.quad(lambda x: rot_accel[2], 0, b)[0]
            velocities.append([x_vel, y_vel, z_vel])

            # perform integration to get position
            x = integrate.quad(lambda x: x_vel, 0, b)[0] + 8 # translate to initial position
            y = integrate.quad(lambda x: y_vel, 0, b)[0] + 1 # translate to initial position 
            z = integrate.quad(lambda x: z_vel, 0, b)[0]
            positions.append([x, y, z])
            gt_x = accel[i][5]
            gt_y = accel[i][6]
            gt_positions.append([gt_x, gt_y])

        time = np.linspace(0,len(positions),num=len(positions))
        positions = np.asarray(positions)
        xs, = plt.plot(time, positions[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        ys, = plt.plot(time, positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        plt.title("Positions vs. time, " + device + ", no RSSI")
        plt.legend([xs, ys], ["x", "y"])
        plt.savefig(results_dir + device + "_pos_v_t_" + str(path_num) + "_no_rssi" + ".png")
        # # plt.legend("x", "y")

        plt.figure()
        velocities = np.asarray(velocities)
        xvels, = plt.plot(time, velocities[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        yvels, = plt.plot(time, velocities[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        plt.title("Velocities vs. time, " + device + ", no RSSI")
        plt.legend([xvels, yvels], ["x vel", "y vel"])
        plt.savefig(results_dir + device + "_vel_v_t_" + str(path_num) + "_no_rssi" + ".png")
        # plt.show()

        plt.figure()
        gt_positions = np.asarray(gt_positions)
        x_m, = plt.plot(time, positions[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        x_gt, = plt.plot(time, gt_positions[:,0], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        plt.title("X values vs. time, " + device + ", no RSSI")
        plt.legend([x_m, x_gt], ["measured", "gt"])
        plt.savefig(results_dir + device + "_x_gt_v_t_" + str(path_num) + "_no_rssi" + ".png")
        # plt.show()

        plt.figure()
        meas_pos, = plt.plot(positions[:,0], positions[:,1], color='green', marker='o', linestyle='dashed', linewidth=2)
        gt_pos, = plt.plot(gt_positions[:,0], gt_positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2)
        plt.title("Path Estimation, " + device + ", no RSSI")
        plt.legend([meas_pos, gt_pos], ["estimation", "ground truth"])
        plt.savefig(results_dir + device + "_path_plot_outliers" + str(path_num) + "_no_rssi" + ".png")

        rsme_dist = []
        rmse_x_vals = []
        rmse_y_vals = []
        positions_no_outliers = []
        gt_positions_no_outliers = []
        for (m_row, gt_row) in zip(positions, gt_positions):
            x_err = rmse_x_vals.append(abs(m_row[0] - gt_row[0]))
            y_err = rmse_y_vals.append(abs(m_row[1] - gt_row[1]))
            dist_err = math.dist([m_row[0], m_row[1]], [gt_row[0], gt_row[1]])
            rsme_dist.append(dist_err)
            if(dist_err < 10):
                positions_no_outliers.append(m_row)
                gt_positions_no_outliers.append(gt_row)
        
        plt.figure()
        positions_no_outliers = np.asarray(positions_no_outliers)
        gt_positions_no_outliers = np.asarray(gt_positions_no_outliers)
        meas_pos, = plt.plot(positions_no_outliers[:,0], positions_no_outliers[:,1], color='green', marker='o', linestyle='dashed', linewidth=2)
        gt_pos, = plt.plot(gt_positions[:,0], gt_positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2)
        plt.title("Path Estimation (No outliers), " + device + ", no RSSI")
        plt.legend([meas_pos, gt_pos], ["estimation", "ground truth"])
        plt.savefig(results_dir + device + "_path_plot_no_outliers" + str(path_num) + "_no_rssi" + ".png")

        plt.figure()
        x_err, = plt.plot(time, rmse_x_vals, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        y_err, = plt.plot(time, rmse_y_vals, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        plt.title("Error vs. time, " + "Path " + str(path_num) + ", " + device + ", no RSSI")
        plt.legend([x_err, y_err], ["x", "y"])
        plt.savefig(results_dir + device + "_x_y_error_" + str(path_num) + "_no_rssi" + ".png")
        
        plt.figure()
        x_err, = plt.plot(time, rsme_dist, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        plt.title("Path " + str(path_num) + ", " + device + ", no RSSI")
        plt.xlabel("time (s)")
        plt.ylabel("Euclidean distance error (m)")
        plt.savefig(results_dir + device + "_dist_error_" + str(path_num) + "_no_rssi" + ".png")
        
        #plt.show()
        input("Press enter for next set of graphs")

def results_with_rssi(device, is_using_rssi, rssi_threshold, path_num, gyro_rssi_match_up_threshold, outlier_threshold, is_plotting, num_plot_points=-1):
    rssi_vals = []
    orig_device_name = device
    results_dir = "/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/dev/velocity_estimation/imu_integration_results/"
    data_dir = "/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/"
    if(is_using_rssi):
        # read in rssi values for given path
        with open(data_dir + 'out_path' + str(path_num) + rssi_threshold + ".csv", 'r') as rssi:
            csv_reader = csv.reader(rssi, delimiter=',')
            for row in csv_reader:
                timestamp = row[0]
                x_est = row[1]
                y_est = row[2]
                new_r = [timestamp, x_est, y_est]
                rssi_vals.append(new_r)

            # convert to proper format for analysis
            rssi_vals = np.asarray(rssi_vals)
            rssi_vals = rssi_vals.astype('float64')
    
    # get accel and gyro data
    accel, gyro = parse_dcnn_data_for_pos_est('test', orig_device_name, path_num)
    
    thetas = []
    velocities = []
    positions = []
    gt_positions = []
    positions_no_outliers = []
    gt_positions_no_outliers = []
    time = []

    base_ts = accel[0][0] # starting timestamp
    orig_base_ts = accel[0][0] # first timestamp (doesn't change)
    init_x = 8 # starting x coordinate
    init_y = 1 # starting y coordinate

    if(num_plot_points == -1):
        num_plot_points = len(accel) - 1 # set number of points to maximum

    # perform integration on gyro data to get position
    for i in range(0,num_plot_points): # TODO: check that it isn't num_plot_points - 1
        print("-------- New Position Estimation --------")
        curr_time = accel[i][0]
        b = accel[i + 1][0] - base_ts # upper limit of integration
        
        # look for matching rssi measurement
        if(is_using_rssi):
            for j in range(0, len(rssi_vals) -1):
                print("Time diff: ", curr_time - rssi_vals[j][0])
                if(abs(curr_time - rssi_vals[j][0]) < gyro_rssi_match_up_threshold): # if there is a closeby rssi measurement
                    print("RSSI match!")
                    base_ts = accel[i][0] # new base_ts
                    init_x = rssi_vals[j][1] # new initial x
                    init_y = rssi_vals[j][2] # new initial y

        # integrate gyro data to get roll, pitch, and yaw
        theta_x = integrate.quad(lambda x: gyro[i][1], 0, b)[0]
        theta_y = integrate.quad(lambda x: gyro[i][2], 0, b)[0]
        theta_z = integrate.quad(lambda x: gyro[i][3], 0, b)[0]
        thetas.append([theta_x, theta_y, theta_z])
        
        # find rotation matrix
        Rot = R([theta_x, theta_y, theta_z])

        # apply rotation matrix to acceleration vector
        a_vec = np.transpose(np.asarray([[accel[i][1], accel[i][2], accel[i][3]]]))
        rot_accel = np.dot(Rot,a_vec)

        # integrate acceleration to get velocity
        x_vel = integrate.quad(lambda x: rot_accel[0], 0, b)[0]
        y_vel = integrate.quad(lambda x: rot_accel[1], 0, b)[0]
        z_vel = integrate.quad(lambda x: rot_accel[2], 0, b)[0]
        velocities.append([x_vel, y_vel, z_vel])

        # integration velocity to get position
        x = integrate.quad(lambda x: x_vel, 0, b)[0] + init_x
        y = integrate.quad(lambda x: y_vel, 0, b)[0] + init_y
        z = integrate.quad(lambda x: z_vel, 0, b)[0]
        positions.append([x, y, z])

        # get ground truth values
        gt_x = accel[i][5]
        gt_y = accel[i][6]
        gt_positions.append([gt_x, gt_y])

        # time vector for plotting
        time.append(curr_time - orig_base_ts)

    if(len(time) != len(positions)):
        print("ERROR: time vector is off by ", len(time) - len(positions))
        exit()
    
    # TODO: change these to be numpy arrays upon declaration
    time = np.asarray(time)
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    gt_positions = np.asarray(gt_positions)
    positions_no_outliers = np.asarray(positions_no_outliers)
    gt_positions_no_outliers = np.asarray(gt_positions_no_outliers)

    # find error metrics
    errors = []
    x_errors = []
    y_errors = []
    for (m_row, gt_row) in zip(positions, gt_positions):
        x_err = x_errors.append(abs(m_row[0] - gt_row[0]))
        y_err = y_errors.append(abs(m_row[1] - gt_row[1]))
        dist_err = math.dist([m_row[0], m_row[1]], [gt_row[0], gt_row[1]])
        errors.append(dist_err)
        if(dist_err < outlier_threshold):
            positions_no_outliers.append(m_row)
            gt_positions_no_outliers.append(gt_row)
   
    if(is_plotting):
        # set plotting names
        rssi_title = ''
        rssi_file_name = ''
        if(is_using_rssi):
            rssi_title = 'with RSSI'
            rssi_file_name = '_with_rssi'
        else:
            rssi_title = 'no RSSI'
            rssi_file_name = '_no_rssi'

        # create plots
        xs, = plt.plot(time, positions[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        ys, = plt.plot(time, positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("Positions vs. time, " + device + ", " + rssi_title)
        plt.legend([xs, ys], ["x", "y"])
        plt.savefig(results_dir + device + "_x_y_positions_" + str(path_num) + rssi_file_name + ".png")

        plt.figure()
        xvels, = plt.plot(time, velocities[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        yvels, = plt.plot(time, velocities[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("Velocities vs. time, " + device + ", " + rssi_title)
        plt.legend([xvels, yvels], ["x vel", "y vel"])
        plt.savefig(results_dir + device + "_x_y_velocities_" + str(path_num) + rssi_file_name + ".png")

        plt.figure()
        x_m, = plt.plot(time, positions[:,0], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        x_gt, = plt.plot(time, gt_positions[:,0], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("X values vs. time, " + device + ", " + rssi_title)
        plt.legend([x_m, x_gt], ["measured", "ground truth"])
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.savefig(results_dir + device + "_x_positions_" + str(path_num) + rssi_file_name + ".png")

        plt.figure()
        x_m, = plt.plot(time, positions[:,1], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        x_gt, = plt.plot(time, gt_positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("Y values vs. time, " + device + ", " + rssi_title)
        plt.legend([x_m, x_gt], ["measured", "ground truth"])
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.savefig(results_dir + device + "_y_positions_" + str(path_num) + rssi_file_name + ".png")

        plt.figure()
        meas_pos, = plt.plot(positions[:,0], positions[:,1], color='green', marker='o', linestyle='dashed', linewidth=2)
        gt_pos, = plt.plot(gt_positions[:,0], gt_positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2)
        plt.title("Path Estimation, " + device + ", " + rssi_title)
        plt.legend([meas_pos, gt_pos], ["estimation", "ground truth"])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(results_dir + device + "_path_plot_outliers" + str(path_num) + rssi_file_name + ".png")
        
        plt.figure()
        meas_pos, = plt.plot(positions_no_outliers[:,0], positions_no_outliers[:,1], color='green', marker='o', linestyle='dashed', linewidth=2)
        gt_pos, = plt.plot(gt_positions[:,0], gt_positions[:,1], color='red', marker='o', linestyle='dashed', linewidth=2)
        plt.title("Path Estimation (No outliers), " + device + ", " + rssi_title)
        plt.legend([meas_pos, gt_pos], ["estimation", "ground truth"])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(results_dir + device + "_path_plot_no_outliers" + str(path_num) + rssi_file_name + ".png")

        plt.figure()
        x_err, = plt.plot(time, x_errors, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        y_err, = plt.plot(time, y_errors, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("Error vs. time, " + "Path " + str(path_num) + ", " + device + ", " + rssi_title)
        plt.legend([x_err, y_err], ["x", "y"])
        plt.xlabel("time (s)")
        plt.ylabel("Euclidean distance error (m)")
        plt.savefig(results_dir + device + "_x_y_errors_" + str(path_num) + rssi_file_name + ".png")
        
        plt.figure()
        x_err, = plt.plot(time, errors, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.title("Path " + str(path_num) + ", " + device + ", ", rssi_title)
        plt.xlabel("time (s)")
        plt.ylabel("Euclidean distance error (m)")
        plt.savefig(results_dir + device + "_dist_error_" + str(path_num) + rssi_file_name + ".png")
    
    print("Average error for path " + str(path_num) + ": ", sum(errors)/len(errors))

    return time, positions, gt_positions

def results_with_rssi_avg_devices(device, path_num, is_using_rssi):
    # read in rssi values
    rssi_vals = []
    data_dir = "/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/"
    with open(data_dir + 'out_path' + str(path_num) + '_0_5.csv', 'r') as rssi: # read in RSSI file for the given set_num
        csv_reader = csv.reader(rssi, delimiter=',')
        for row in csv_reader:
            timestamp = row[0]
            x_est = row[1]
            y_est = row[2]
            new_r = [timestamp, x_est, y_est]
            rssi_vals.append(new_r)

    rssi_vals = np.asarray(rssi_vals)
    rssi_vals = rssi_vals.astype('float64')
    accel, gyro = parse_dcnn_data_for_pos_est('test', device, path_num)
    thetas = []
    velocities = []
    ts_positions = []
    positions = []
    gt_positions = []
    base_ts = accel[0][0]
    init_x = 8
    init_y = 1

    # perform integration on gyro data to get theta vector
    for i in range(0,len(accel) - 1):
        print("-------- New Position Estimation --------")
        curr_time = accel[i][0]
        if(is_using_rssi):
            for j in range(0, len(rssi_vals) -1):
                print("Time diff: ", curr_time - rssi_vals[j][0])
                if(abs(curr_time - rssi_vals[j][0]) < 0.5): # check if there is a new rssi measurement
                    print("RSSI match!")
                    base_ts = accel[i][0] # set new base_ts
                    init_x = rssi_vals[j][1] # set new init point
                    init_y = rssi_vals[j][2]

        b = accel[i + 1][0] - base_ts
        theta_x = integrate.quad(lambda x: gyro[i][1], 0, b)[0]
        theta_y = integrate.quad(lambda x: gyro[i][2], 0, b)[0]
        theta_z = integrate.quad(lambda x: gyro[i][3], 0, b)[0]
        thetas.append([theta_x, theta_y, theta_z])
        #print("Theta: ", theta_x, theta_y, theta_z)
        
        # find rotation matrix
        Rot = R([theta_x, theta_y, theta_z])

        # apply rotation matrix to acceleration vector
        a_vec = np.transpose(np.asarray([[accel[i][1], accel[i][2], accel[i][3]]]))
        rot_accel = np.dot(Rot,a_vec)
        #print("Rotated accel: ", rot_accel[0], rot_accel[1], rot_accel[2])

        # perform integration to get velocity
        x_vel = integrate.quad(lambda x: rot_accel[0], 0, b)[0]
        y_vel = integrate.quad(lambda x: rot_accel[1], 0, b)[0]
        z_vel = integrate.quad(lambda x: rot_accel[2], 0, b)[0]
        velocities.append([x_vel, y_vel, z_vel])
        #print ("Velocity: ", x_vel, y_vel, z_vel)

        # perform integration to get position
        x = integrate.quad(lambda x: x_vel, 0, b)[0] + init_x
        y = integrate.quad(lambda x: y_vel, 0, b)[0] + init_y
        z = integrate.quad(lambda x: z_vel, 0, b)[0]
        positions.append([x, y, z])
        ts_positions.append([curr_time, x, y, z])
        #print("Postion: ", x, y, z)
        gt_x = accel[i][5]
        gt_y = accel[i][6]
        gt_positions.append([gt_x, gt_y])
    
    return ts_positions, gt_positions

def evaluate_averaging():
    path_num = 2
    avg_positions = []
    gt = []
    iphone, iphone_gt = results_with_rssi_avg_devices('iphone', path_num, True)
    watch, watch_gt = results_with_rssi_avg_devices('watch', path_num, False)
    print("No RSSI: averaging " + str(len(iphone)) + " iphone points, " + str(len(watch)) + " watch points")
    for p, w, p_row, w_row in zip(iphone_gt, watch_gt, iphone, watch):
        if(p[0] == w[0] and p[1] == w[1]): # if ground truths are equal
            x = (p_row[1] + w_row[1])/2.0
            y = (p_row[2] + w_row[2])/2.0
            avg_positions.append([x, y])
    
    if(len(avg_positions) == len(watch)):
        gt = watch_gt
    else:
        gt = iphone_gt
    
    rsme_dist = []
    for (m_row, gt_row) in zip(avg_positions, gt):
        dist_err = math.dist([m_row[0], m_row[1]], [gt_row[0], gt_row[1]])
        rsme_dist.append(dist_err)

    plt.figure()
    time = np.linspace(0,len(avg_positions),num=len(avg_positions))
    x_err, = plt.plot(time, rsme_dist, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    plt.title("Path " + str(path_num) + " Averaged iPhone with RSSI and Watch")
    plt.xlabel("time (s)")
    plt.ylabel("Euclidean distance error (m)")
    plt.savefig("muli_device_dist_err_iphone_with_rssi" + str(path_num) + ".png")

def main():
    # train
    # a1, g1 = parse_dcnn_data('train', 'iphone', 1)
    # a2, g2 = parse_dcnn_data('train', 'watch', 1) 
    # a1, g1 = parse_dcnn_data('train', 'iphone', 2)
    # a2, g2 = parse_dcnn_data('train', 'watch', 2) 
    # a1, g1 = parse_dcnn_data('train', 'iphone', 3)
    # a2, g2 = parse_dcnn_data('train', 'watch', 3) 

    # test
    # a1, g1 = parse_dcnn_data('test', 'iphone', 1)
    # a2, g2 = parse_dcnn_data('test', 'watch', 1) 
    # a1, g1 = parse_dcnn_data('test', 'iphone', 2)
    # a2, g2 = parse_dcnn_data('test', 'watch', 2) 
    # a1, g1 = parse_dcnn_data('test', 'iphone', 3)
    # a2, g2 = parse_dcnn_data('test', 'watch', 3) 

    # Uncomment to get individual device results
    # results_no_rssi('iphone')
    # input("Press enter for next large batch of graphs")
    # results_no_rssi('watch')
    # input("Press enter for next large batch of graphs")
    results_with_rssi('iphone')
    # input("Press enter for next large batch of graphs")
    # results_with_rssi('watch')

    # Uncomment to get multi-device results
    # evaluate_averaging()

    # Uncomment to parse rssi_data into a csv file
    threshold = 2 # seconds
    for i in range(1,4):
        r = parse_rssi_data(4,5,2021,i,threshold) # 
        df = pd.DataFrame(r)
        df.to_csv('test_rssi_parsed' + str(i) + '_thresh' + str(threshold) + '.csv', index=False)

    input("Press enter to end")
    print("Success!")

if __name__ == "__main__": 
    main()
import os
import sys
import csv
import numpy as np
import itertools
import user_velocity as uv
from datetime import datetime, timezone
import pandas as pd

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

def main():
    # train
    a1, g1 = parse_dcnn_data('train', 'iphone', 1)
    a2, g2 = parse_dcnn_data('train', 'watch', 1) 
    a1, g1 = parse_dcnn_data('train', 'iphone', 2)
    a2, g2 = parse_dcnn_data('train', 'watch', 2) 
    a1, g1 = parse_dcnn_data('train', 'iphone', 3)
    a2, g2 = parse_dcnn_data('train', 'watch', 3) 

    # test
    a1, g1 = parse_dcnn_data('test', 'iphone', 1)
    a2, g2 = parse_dcnn_data('test', 'watch', 1) 
    a1, g1 = parse_dcnn_data('test', 'iphone', 2)
    a2, g2 = parse_dcnn_data('test', 'watch', 2) 
    a1, g1 = parse_dcnn_data('test', 'iphone', 3)
    a2, g2 = parse_dcnn_data('test', 'watch', 3) 
    
    # Uncomment to parse rssi_data into a csv file
    # threshold = 2 # seconds
    # for i in range(1,4):
    #     r = parse_rssi_data(4,5,2021,i,threshold) # 
    #     df = pd.DataFrame(r)
    #     df.to_csv('test_rssi_parsed' + str(i) + '_thresh' + str(threshold) + '.csv', index=False)
    
    print("Success!")

if __name__ == "__main__": 
    main()
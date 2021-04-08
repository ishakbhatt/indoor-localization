import os
import sys
import numpy as np
import itertools
import user_velocity as uv

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

    if(set_num > 3 or set_num < 1):
        print("Incorrect set number parameter. Please input set number 1, 2, or 3.")
        sys.exit()

    # init output arrays
    gyro_data = [0,0,0,0,0]
    accel_data = [0,0,0,0,0]
    
    # get correct directory
    # data_dir = uv.get_data_directory('train')
    data_dir = '/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/data/' + set_type + '/'

    # get timestamp and vel
    vel_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_vel' + str(set_num) + '.csv'), True)
    vel_cols = [2, 3]
    vel_arr = uv.new_gen_sensor_array(vel_cols, vel_arr)
    vel_arr = vel_arr.astype('float64')

    # get timestamps
    if(device == 'iphone'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), True)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64') - 14400
    elif(device == 'watch'):
        imu_arr = uv.genfromtxt_with_unix_convert(os.path.join(data_dir, set_type + '_' + device + str(set_num) + '.csv'), False)
        timestamps = uv.new_gen_sensor_array([0], imu_arr)
        timestamps = timestamps.astype('float64')

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

def main():
    a, g = parse_dcnn_data('train', 'iphone', 1)
    aa, gg = parse_dcnn_data('train', 'watch', 1)
    print("Success!")

if __name__ == "__main__": 
    main()
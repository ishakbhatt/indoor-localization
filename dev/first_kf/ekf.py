import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

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

    return mu_new, Sigma_new
    
def get_data_directory():
    """
    Get repository data directory.
    """
    
    # change directory
    os.chdir('../../data/') # TODO: no file or directory
    # get working directory
    data_directory = os.getcwd()
    os.chdir('../dev/first_kf/')
    return data_directory

def make_matrices():
    """
    Format IMU CSV data into Numpy matrices.
    """
    data_dir = get_data_directory()
    # open csv file
    with open(os.path.join(data_dir, 'iphoneIMU.csv'), 'r') as csv_file
        csv_reader = csv.reader(csv_file, delimiter=',')
        for 
    


def main():
    # make three matrices of gyro, accel, magnetometer data

if __name__ == "__main__":
    main()
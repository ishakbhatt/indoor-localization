# EECS 507 Final Project: Indoor Localization of Apple Devices

This repository contains source code for the EECS 507 (Embedded Systems Research) final project for the Winter 2021 semester. 

The goal of this project is to estimate an object's position in a two-dimensional space based on inertial (IMU) and WiFi signal strength (RSSI) data from a combination of Apple devices, specifically from the iPhone XR and the Apple Watch Series 3. Our algorithm uses RSSI to correct double integration of accelerometer data for position estimation. For future research, we investigated a velocity estimation method that leverages a deep convolutional neural network. Due to the time constraint, we were unable to accurately get velocity estimations for singly-integrated position values, so primarily focused on developing a dataset for this application.

Our final paper can be found [here](https://github.com/ishakbhatt/indoor-localization/blob/main/Multi_Device_Sensor_Fusion_with_Inertial_Data_and_WLAN_Signals_for_Improved_Indoor_Localization.pdf).

import numpy as np
import matplotlib.pyplot as plt
import json

from GPSjson.transfrom_coords import *

ACTUAL_GRAVITY = 9.80665

class KalmanFilterFusedPositionAccelerometer:
    def __init__(self, sensor_data, latLonStandardDeviation, altitudeStandardDeviation, accelerometerStandardDeviation):
        self.timestamp = sensor_data['timestamp']
        self.gps_lat = sensor_data['gps_lat']
        self.gps_lon = sensor_data['gps_lon']
        self.gps_alt = sensor_data['gps_alt']
        self.pitch = sensor_data['pitch']
        self.yaw = sensor_data['yaw']
        self.roll = sensor_data['roll']
        self.abs_north_acc = sensor_data['abs_north_acc']
        self.abs_east_acc = sensor_data['abs_east_acc']
        self.abs_up_acc = sensor_data['abs_up_acc']
        self.vel_north = sensor_data['vel_north']
        self.vel_east = sensor_data['vel_east']
        self.vel_down = sensor_data['vel_down']
        self.vel_error = sensor_data['vel_error']
        self.altitude_error = sensor_data['altitude_error']
        self.latlon_std = latLonStandardDeviation
        self.alt_std = altitudeStandardDeviation
        self.acc_std = accelerometerStandardDeviation

        self.current_state = np.empty(shape=(6, 1))
        self.u = np.empty(shape=(3, 1))
        self.z = np.empty(shape=(6, 1))
        self.H = np.identity(6)
        self.P = np.identity(6)
        self.Q = np.empty(shape=(6, 6))
        self.R = np.empty(shape=(6, 6))
        self.B = np.zeros(shape=(6, 3))
        self.A = np.identity(6)
        self.I = np.identity(6)

    def init_KF_GPS_Acc(self):
        xy = geographic2plane([self.gps_lat, self.gps_lon])
        # latlon = plane2geographic(xy)

        # self.current_state[:3] = np.array([self.gps_lon, self.gps_lat, self.gps_alt]).reshape(3, 1)
        self.current_state[:3] = np.array([xy[0], xy[1], self.gps_alt]).reshape(3, 1)
        self.current_state[3:] = np.array([self.vel_east, self.vel_north, -self.vel_down]).reshape(3, 1)

        process_variance = self.acc_std * self.acc_std
        self.Q = np.diag([process_variance, process_variance, process_variance,
                          process_variance, process_variance, process_variance])

        latlon_variance = self.latlon_std * self.latlon_std
        alt_variance = self.alt_std * self.alt_std
        self.R = np.diag([latlon_variance, latlon_variance, alt_variance,
                          latlon_variance, latlon_variance, alt_variance])

    def predict(self, acc_east, acc_north, acc_up, timestamp_now):
        delta_sec = timestamp_now - self.timestamp

        self.recreate_state_matrix(delta_sec)    # A
        self.recreate_control_matrix(delta_sec)  # B

        self.u[0] = acc_east
        self.u[1] = acc_north
        self.u[2] = acc_up

        self.current_state = self.A @ self.current_state + self.B @ self.u

        self.P = self.A @ self.P @ self.A.T + self.Q

        self.timestamp = timestamp_now

    def update(self, position, velocity, pos_error, vel_error):
        self.z[:3] = position
        self.z[3:] = velocity

        # if positionError != nil {
        # k.R.Put(0, 0, * positionError ** positionError)
        # } else {
        # }
        self.R[3, 3] = vel_error * vel_error
        self.R[4, 4] = vel_error * vel_error
        self.R[5, 5] = vel_error * vel_error

        y = self.z - self.current_state
        s = self.P + self.R

        K = self.P @ np.linalg.pinv(s)

        self.current_state = self.current_state + K @ y

        self.P = (self.I - K) @ self.P

    def recreate_state_matrix(self, delta_sec):
        self.A[0, 3] = delta_sec
        self.A[1, 4] = delta_sec
        self.A[2, 5] = delta_sec

    def recreate_control_matrix(self, delta_sec):
        dt_squared = 0.5 * delta_sec * delta_sec

        self.B[0, 0] = dt_squared
        self.B[1, 1] = dt_squared
        self.B[2, 2] = dt_squared
        self.B[3, 0] = delta_sec
        self.B[4, 1] = delta_sec
        self.B[5, 2] = delta_sec

    def get_predicted_position(self):
        return self.current_state[:3]

    def get_predicted_velocity(self):
        return self.current_state[3:]


pos_measure_save = np.empty(shape=(1473, 3))
vel_measure_save = np.empty(shape=(1473, 3))
# pos_measure_save = np.empty(shape=(347, 3))     # edit version
# vel_measure_save = np.empty(shape=(347, 3))
pos_esti_save = np.empty(shape=(6557, 3))
vel_esti_save = np.empty(shape=(6557, 3))

time_measure = np.zeros(1473)
# time_measure = np.zeros(347)
time_esti = np.zeros(6557)

if __name__ == "__main__":
    with open('GPSjson/pos_final.json') as json_file:
        sensor_data = json.load(json_file)

    # Initial State Matrix
    X = sensor_data[0]

    latLonStandardDeviation = 2.0 # +/- 1 m, increased for safety
    altitudeStandardDeviation = 3.518522417151836

    # got this value by getting standard deviation from accelerometer, assuming that mean SHOULD be 0
    accelerometerStandardDeviation = ACTUAL_GRAVITY * 0.033436506994600976

    # Initialization
    kf = KalmanFilterFusedPositionAccelerometer(X, latLonStandardDeviation, altitudeStandardDeviation,
                                                accelerometerStandardDeviation)
    kf.init_KF_GPS_Acc()

    cnt = 0
    for i in range(1, len(sensor_data)):
        data = sensor_data[i]
        # Predict
        kf.predict(data['abs_east_acc'], data['abs_north_acc'], data['abs_up_acc'], data['timestamp'])

        if data['gps_lon'] != 0:
            # Update
            xy = geographic2plane([data['gps_lat'], data['gps_lon']])
            position = np.array([xy[0], xy[1], data['gps_alt']]).reshape(3, 1)
            velocity = np.array([data['vel_east'], data['vel_north'], -data['vel_down']]).reshape(3, 1)

            kf.update(position, velocity, [], data['vel_error'])

            pos_measure_save[cnt] = np.array([xy[0], xy[1], data['gps_alt']])
            vel_measure_save[cnt] = np.array([data['vel_east'], data['vel_north'], -data['vel_down']])

            time_measure[cnt] = data['timestamp'] - X['timestamp']

            cnt += 1

        # Return current the position & velocity
        predicted_xy = kf.get_predicted_position()
        predicted_vel = kf.get_predicted_velocity()

        predicted_latlon = plane2geographic(predicted_xy)

        delta_T = data['timestamp'] - X['timestamp']

        resultant_vel = np.sqrt(predicted_vel[0] * predicted_vel[0] + predicted_vel[1] * predicted_vel[1]) # m/s

        pos_esti_save[i-1] = predicted_xy.T
        vel_esti_save[i-1] = predicted_vel.T
        time_esti[i-1] = delta_T

    # Compare the results between measurements(GPS position) and computed ... graph
    plt.figure(1)
    plt.plot(pos_measure_save[:, 0], pos_measure_save[:, 1], 'r*--', label='Measurements', markersize=10)
    plt.plot(pos_esti_save[:, 0], pos_esti_save[:, 1], 'b-', label='Estimation (KF)')
    plt.legend(loc='upper left')
    plt.title('Position: Meas. v.s. Esti. (KF)')
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.figure(2)
    plt.plot(time_measure, pos_measure_save[:, 0], 'r*--', label='Measurements', markersize=10)
    plt.plot(time_esti, pos_esti_save[:, 0], 'b-', label='Estimation (KF)')
    plt.legend(loc='upper left')
    plt.title('Position(X): Meas. v.s. Esti. (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Easting [m]')

    # plt.show()
    plt.figure(3)
    plt.plot(time_measure, pos_measure_save[:, 1], 'r*--', label='Measurements', markersize=10)
    plt.plot(time_esti, pos_esti_save[:, 1], 'b-', label='Estimation (KF)')
    plt.legend(loc='upper left')
    plt.title('Position(Y): Meas. v.s. Esti. (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Northing [m]')

    plt.figure(4)
    plt.plot(time_measure, vel_measure_save[:, 0], 'g*--', label='Measurements', markersize=10)
    plt.plot(time_esti, vel_esti_save[:, 0], 'b-', label='Estimation (KF)')
    plt.legend(loc='lower right')
    plt.title('Velocity(X): Meas. v.s. Esti. (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Velocity X [m/s]')

    plt.figure(5)
    plt.plot(time_measure, vel_measure_save[:, 1], 'g*--', label='Measurements', markersize=10)
    plt.plot(time_esti, vel_esti_save[:, 1], 'b-', label='Estimation (KF)')
    plt.legend(loc='lower right')
    plt.title('Velocity(Y): Meas. v.s. Esti. (KF)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Velocity Y [m/s]')

    plt.show()
    # plt.savefig('result.png')

    print('Hello')
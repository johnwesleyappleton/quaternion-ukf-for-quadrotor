import numpy as np
from scipy import io, linalg
from quaternion import Quaternion

# data files are numbered on the server.
# for example imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an unscented kalman filter


def estimate_rot(data_num=1):
    # load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    accel = imu['vals'][0:3]
    gyro = imu['vals'][3:6]
    T = np.shape(imu['ts'])[1]

    # reorder gyro readings
    gyro = np.asarray([gyro[1], gyro[2], gyro[0]])

    # calibrate accel
    accel_bias = [510.808, 500.994, 505.046]
    accel_sensitivity = 32.64
    accel = (np.asarray([-1, -1, 1]) * (accel.T - accel_bias) * 3300 / (1023 * accel_sensitivity)).T

    # calibrate gyro
    gyro_bias = [372.7589, 374.6554, 369.4072]
    gyro_sensitivity = np.asarray([200, 200, 200])
    gyro = ((gyro.T - gyro_bias) * np.pi * 3300 / (1023 * gyro_sensitivity * 180)).T

    # initialize roll, pitch, and yaw arrays
    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)

    # unscented kalman filter
    x_q = Quaternion()  # state of orientation
    x_w = np.zeros(3)  # state of angular velocities
    P = 0.5 * np.identity(6)  # covariance of the state
    R = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])  # covariance of the dynamics
    Q = np.diag([9, 9, 9, 9, 9, 9])  # covariance of the measurements
    for i in range(T):
        # timestep
        if i == 0:
            dt = imu['ts'][0, i] - 0.0001
        else:
            dt = imu['ts'][0, i] - imu['ts'][0, i-1]

        # get and transform sigma points
        P += R * dt
        X = get_sigma_points(P, x_q, x_w)
        Y = transform_sigma_points(X, dt)

        # propagate dynamics
        x_q, x_w, P, W = propagate_dynamics(x_q, Y)

        # get measurements
        acc = accel[:, i]
        gyr = gyro[:, i]

        # measurement update
        x_q, x_w, P = measurement_update(x_q, x_w, acc, gyr, W, P, Q)

        # convert to euler angles
        euler = x_q.euler_angles()
        roll[i] = euler[0]
        pitch[i] = euler[1]
        yaw[i] = euler[2]

    # show plots
    # plots(roll, pitch, yaw)

    return roll, pitch, yaw


def get_sigma_points(P, x_q, x_w):
    S = linalg.sqrtm(P)
    W = np.hstack((np.sqrt(6) * S, - np.sqrt(6) * S))
    X = np.zeros((7, 12))
    for i in range(12):
        # orientation
        sig_q = Quaternion()
        sig_q.from_axis_angle(W[:3, i])
        sig_q = (x_q * sig_q).q

        # angular velocities
        sig_w = W[3:, i] + x_w

        # sigma point
        X[:, i] = np.hstack((sig_q, sig_w))

    return X


def transform_sigma_points(X, dt):
    Y = np.zeros((7, 12))
    for i in range(12):
        # get orientation and velocity
        sig_q = Quaternion(X[0, i], X[1:4, i])
        sig_w = X[4:, i]

        # transformation
        delta_q = Quaternion()
        delta_q.from_axis_angle(sig_w * dt)
        sig_q = (delta_q * sig_q).q

        # transformed sigma point
        Y[:, i] = np.hstack((sig_q, sig_w))

    return Y


def propagate_dynamics(x_q, Y):
    E = np.ones((3, 12))
    e = np.sum(E, axis=1) / 12
    while np.linalg.norm(e) > 0.001:
        for i in range(12):
            # compute error vectors
            q_i = Quaternion(Y[0, i], Y[1:4, i])
            e_i = q_i * x_q.inv()
            E[:, i] = e_i.axis_angle()

        # get new mean
        e = np.sum(E, axis=1) / 12
        e_q = Quaternion()
        e_q.from_axis_angle(e)
        x_q = e_q * x_q

    # mean of angular velocity
    x_w = np.mean(Y[4:], axis=1)

    # covariance of the state
    W = np.vstack((E, (Y[4:, :].T - x_w).T))
    cov = W @ W.T / 12

    return x_q, x_w, cov, W


def measurement_update(x_q, x_w, acc, gyr, W, P, Q):
    X = get_sigma_points(P, x_q, x_w)
    Z = np.zeros((6, 12))
    g = Quaternion(0, [0, 0, 9.81])
    for i in range(12):
        # propagate using measurement model
        q_j = Quaternion(X[0, i], X[1:4, i])

        # get z
        Z[:3, i] = (q_j.inv() * g * q_j).vec()
        Z[3:, i] = X[4:, i]

    # get mean and covariance of Z
    z_k_bar = np.mean(Z, axis=1)
    Z_err = (Z.T - z_k_bar).T
    Pzz = Z_err @ Z_err.T / 12 + Q

    # get cross covariance
    Pxz = W @ Z_err.T / 12

    # get innovation
    innovation = np.hstack((acc, gyr)) - z_k_bar

    # get kalman gain
    K = Pxz @ np.linalg.inv(Pzz)

    # get kalman update
    update = K @ innovation
    q = Quaternion()
    q.from_axis_angle(update[:3])

    # update mean
    x_q = q * x_q
    x_w += update[3:]

    # update covariance
    P -= K @ Pzz @ K.T

    return x_q, x_w, P


def calibrate(data_num=1):
    # load vicon data
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    vicon_rots = vicon['rots'][:, :]

    # load imu data
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    accel = imu['vals'][0:3]
    gyro = imu['vals'][3:6]
    ts = imu['ts'][0]

    # define length
    T = min(vicon_rots.shape[2], ts.shape[0])
    vicon_rots = vicon_rots[:, :, :T]
    accel = accel[:, :T]
    gyro = gyro[:, :T]
    ts = ts[:T]

    # reorder gyro readings
    gyro = np.asarray([gyro[1], gyro[2], gyro[0]])

    # tune accelerometer
    err = np.zeros(10000)
    stat_accel = accel[:, :500]
    bias = np.zeros(3)
    bias[:2] = np.mean(stat_accel[:2], axis=1)
    bias[2] = np.mean(bias[:2])
    for i in range(10000):
        acc = (np.asarray([-1, -1, 1]) * (stat_accel.T - bias) * 3300 / (1023 * (i+1)/100)).T
        err[i] = np.linalg.norm((np.linalg.norm(acc, axis=0) - 9.81))

    # calculate accel bias and sensitivity
    accel_sensitivity = (np.argmin(err) + 1) / 100
    scale_factor = 3300 / 1023 / accel_sensitivity
    accel_bias = np.mean(stat_accel, axis=1) - np.asarray([0, 0, 1]) / scale_factor

    # report accel bias and sensitivity
    print("Accelerometer bias is ", accel_bias)
    print("Accelerometer sensitivity is ", accel_sensitivity)

    # get vicon velocities
    vel = np.zeros((T-1, 3))
    q_0 = Quaternion()
    q_1 = Quaternion()
    for i in range(2, T):
        q_0.from_rotm(vicon_rots[:, :, -i])
        q_1.from_rotm(vicon_rots[:, :, -i+1])
        vel[-i+1] = (q_1 * q_0.inv()).axis_angle() / (ts[-i+1] - ts[-i])

    # tune gyroscope
    err = np.zeros((100000, 3))
    bias = np.mean(gyro[:, :500], axis=1)
    for i in range(100000):
        gyr = ((gyro.T - bias) * (np.pi/180) * 3300 / (1023 * (i+1)/100)).T[:, :T-1]
        err[i] = np.linalg.norm(gyr - vel.T, axis=1)

    # calculate gyro bias and sensitivity
    gyro_sensitivity = (np.argmin(err, axis=0) + 1) / 100
    gyro_bias = np.mean(gyro[:, :500], axis=1)

    # report gyro bias and sensitivity
    print("Gyroscope bias is ", gyro_bias)
    print("Gyroscope sensitivity is ", gyro_sensitivity)


def plots(roll, pitch, yaw, data_num=1):
    import matplotlib.pyplot as plt

    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    vicon_rots = vicon['rots']

    vicon_roll = np.arctan2(vicon_rots[2, 1], vicon_rots[2, 2])
    vicon_pitch = np.arctan2(-vicon_rots[2, 0], np.sqrt(vicon_rots[2, 1] ** 2 + vicon_rots[2, 2] ** 2))
    vicon_yaw = np.arctan2(vicon_rots[1, 0], vicon_rots[0, 0])

    plt.plot(roll, label='kalman')
    plt.plot(vicon_roll, label='vicon')
    plt.legend()
    plt.show()

    plt.plot(pitch, label='kalman')
    plt.plot(vicon_pitch, label='vicon')
    plt.legend()
    plt.show()

    plt.plot(yaw, label='kalman')
    plt.plot(vicon_yaw, label='vicon')
    plt.legend()
    plt.show()


# calibrate()
estimate_rot()

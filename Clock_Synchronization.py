# Used for importing data from matlab files
import scipy.io
# Used for data manipulation (including linear algebra)
import numpy as np
# Matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker as mlk

# Filepath to matlab file
fp = "../data/ClockSync.mat"


def import_data():
    """
    Imports data from a matlab file, using pre-known structure

    :return: returns a list with all variables
    """
    # Import file
    raw_data = scipy.io.loadmat(fp)

    # Importing scalars
    K = raw_data.get('K')[0][0]
    dist = raw_data.get('dist')[0][0]
    phi_1 = raw_data.get('phi_1')[0][0]
    phi_2 = raw_data.get('phi_2')[0][0]
    speed_medium = raw_data.get('speed_medium')[0][0]
    tau = raw_data.get('tau')[0][0]
    trial = raw_data.get('trial')[0][0]

    # Import arrays
    noise_var = raw_data.get('noise_var')[0]
    rx_t = raw_data.get('rx_timestamps')
    tx_t = raw_data.get('tx_timestamps')

    # return all imported data structures as a list
    return K, dist, phi_1, phi_2, speed_medium, tau, trial, \
           noise_var, rx_t, tx_t


def main():
    """
    Main function, includes the implementation of the described approach

    :return:
    """
    K, dist, phi_1, phi_2, speed_medium, tau, trial, \
    noise_var, rx_t, tx_t = import_data()

    # Define H as [ 1 ]
    H = np.ones(10)

    # Variable (empty) to keep all calculated x values
    x = np.empty((10, 10000, 6), dtype=np.float32)

    # Variable (empty) to keep all estimated A_hat
    A_hat = np.empty(6)

    # Variance
    A_var = np.empty(6)

    # Errors
    A_mse = np.empty(6)

    # CRLB
    CRLB = np.empty(6)

    estimator_var = np.empty(6)

    # Find the value of X
    for idx_z in range(0, len(rx_t[0][0])):
        x_sum = 0
        for idx_y in range(0, len(rx_t[0])):

            # Calculate the average of x within this single trial
            x[:, idx_y, idx_z] = rx_t[:, idx_y, idx_z]\
                                 - tx_t[:, idx_y, idx_z] - tau
            x_sum = x_sum + sum(x[:, idx_y, idx_z]) / 10

        # Calculate the A_hat values
        A_hat[idx_z] = x_sum / 10000

        # Calculate the A_var values
        A_var[idx_z] = noise_var[idx_z] / 10000

    A_mean = sum(A_hat) / 6

    for idx_z in range(0, len(rx_t[0][0])):
        A_mse[idx_z] = (phi_2 - A_hat[idx_z]) ** 2

        CRLB[idx_z] = noise_var[idx_z] / 10000
        estimator_var[idx_z] = CRLB[idx_z]

    print(A_hat)
    print(A_var)
    print(A_mean)
    print(A_mse)

    print(noise_var)

    fig1, ax1 = plt.subplots()
    ax1.plot(noise_var, A_mse, "b", label='Numerical')
    ax1.plot(noise_var, estimator_var, "--r", label='Theoretical')
    ax1.set_xscale('log')
    ax1.set_xlabel("Noise Variance")
    ax1.set_ylabel("MSE")
    ax1.set_title("Mean Square Error vs Noise Variance (numerical and theoretical)")
    ax1.legend(loc='upper center', shadow=True)
    ax1.yaxis.set_major_formatter(mlk.FormatStrFormatter('%.2e'))

    fig2, ax2 = plt.subplots()
    ax2.plot(noise_var, CRLB, "g", label='CRLB')
    ax2.plot(noise_var, estimator_var, "*--r", label='Theoretical')
    ax2.set_xscale('log')
    ax2.set_xlabel("Noise Variance")
    ax2.set_ylabel("CRLB")
    ax2.set_title("CRLB, theoretical MSE vs Noise Variance")
    ax2.legend(loc='upper center', shadow=True)
    ax2.yaxis.set_major_formatter(mlk.FormatStrFormatter('%.2e'))
    plt.show()

if __name__ == '__main__':
    main()





import numpy as np
import csv
import matplotlib.pyplot as plt


def generate_testdata(data_id, noise_type):
    """
    :param data_id: range from 1 to 192
    :param noise_type: supported noise types: ["gaussian", "uniform"]
    """
    print("generating data {:03d}...".format(data_id))
    with open("data_raw/rrest-syn{:03d}_data.csv".format(data_id), 'rt') as csvfile:
        reader = csv.reader(csvfile)
        ppg = np.array([row[0] for row in reader]).astype(np.float)

    noise0 = np.zeros(ppg.shape)
    noise1 = np.zeros(ppg.shape)
    noise2 = np.zeros(ppg.shape)

    if noise_type == "gaussian":
        noise0 = np.random.normal(0, 1, ppg.shape)
        noise1 = np.random.normal(0, 1, ppg.shape)
        noise2 = np.random.normal(0, 1, ppg.shape)
    elif noise_type == "uniform":
        noise0 = np.random.uniform(low=-2.0, high=2.0, size=ppg.shape)
        noise1 = np.random.uniform(low=-2.0, high=2.0, size=ppg.shape)
        noise2 = np.random.uniform(low=-2.0, high=2.0, size=ppg.shape)

    ppg_withnoise0 = ppg + noise0
    ppg_withnoise1 = ppg + noise1
    ppg_withnoise2 = ppg + noise2
    ppg_withnoise = np.array([ppg_withnoise0, ppg_withnoise1, ppg_withnoise2])
    np.savetxt("data_test/rrest-syn{:03d}_data.csv".format(data_id), ppg_withnoise.transpose(), delimiter=",")

    # with open('data_test/rrest-syn{:03d}_data.csv'.format(data_id), 'wt') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow(ppg_withnoise0)
    #     writer.writerow(ppg_withnoise1)
    #     writer.writerow(ppg_withnoise2)


def display_data(data_type, data_id, display_elements=5000, display_interval=10):
    """
    :param data_type: ["raw", "test"]
    :param data_id: 1~192
    :param display_elements: number of points displayed
    :param display_interval: max time
    """
    if data_type == "raw":
        with open('data_raw/rrest-syn{:03d}_data.csv'.format(data_id), 'rt') as csvfile:
            reader = csv.reader(csvfile)
            ppg = np.array([float(row[0]) for row in reader])
    elif data_type == "test":
        with open('data_test/rrest-syn{:03d}_data.csv'.format(data_id), 'rt') as csvfile:
            reader = csv.reader(csvfile)
            ppg = np.array([float(row[0]) for row in reader])

    even_times = np.linspace(0, display_interval, display_elements)
    plt.figure()

    if data_type == "raw":
        plt.title("Clean PPG Signal")
    elif data_type == "test":
        plt.title("Generated PPG Signal")

    plt.xlim((0, display_interval))
    plt.ylim((0.0, 1.5))
    plt.xlabel("Sampling Points")
    plt.plot(even_times, ppg[:display_elements])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    for i in range(1,193):
        generate_testdata(i, "gaussian")

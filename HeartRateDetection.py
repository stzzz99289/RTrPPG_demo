import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.ROI import get_ROI, get_raw_signals
from lib.signal_process import process_raw, process_raw_debug, process_raw_chrom

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn import preprocessing
from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning

# filter some warnings
simplefilter("ignore", category=ConvergenceWarning)
filterwarnings('ignore')


def putCentertext(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = int((image.shape[1] - textsize[0]) / 2)
    textY = int((image.shape[0] + textsize[1]) / 2)
    scale = 1
    thickness = 2
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def putBottomtext(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textX = 0
    textY = image.shape[0]
    scale = 1
    thickness = 2
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def plt2cv(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


class getHRV:

    def __init__(self, input_file=None, output_file='out.avi'):
        self.cam_index = 0
        self.offline = False
        self.input_fps = 24
        self.height = 1080
        self.width = 1920
        if input_file is not None:
            self.cap = cv2.VideoCapture(input_file)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
            # self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            print("input video info: total number of frames: {}, fps: {}".format(total_frames, self.input_fps))
            self.offline = True
        else:
            self.cap = cv2.VideoCapture(self.cam_index)

        # each [update_time] seconds, calculate a new hr value
        # each new hr value will be calculated based on info in previous [interval_time] seconds 
        self.update_time = 2  
        self.interval_scale = 5  
        self.interval_time = self.interval_scale * self.update_time  
        self.start_time = time.time()

        # lists used to stores all info and info in a buffer
        self.ROI_values_all = []
        self.ROI_values_buffer = []
        self.raw_times_all = []
        self.raw_times_buffer = []

        self.update_index = [0]  # index in ROI_values_all
        self.update_count = 1
        self.buffer_start = 0  # buffer start time
        self.buffer_end = self.interval_scale  # buffer end time

        self.bpms = []
        self.adjusted_bpms = []
        self.bpm_times = []

        self.calculating = False
        self.recording = False

        # images displayed for online demo
        self.init_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                        "Press s to start.")
        self.cal_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                       "Calculating...")
        self.rPPG_image = self.init_image
        self.bpm_image = self.init_image

        self.debug = False
        self.od = True

        self.frame_count = 0
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), self.input_fps, (self.width,self.height))
    
    def set_rPPG_image_debug(self, buffer_size, rppgs, bpms, kurt, target_index, current_time):
        rPPG_fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        even_times = np.linspace(self.buffer_start, self.buffer_end, buffer_size)
        line0, = ax1.plot(even_times, rppgs[0])
        ax1.set_title("bpm:{:.0f}, kurt:{:.2f}".format(bpms[0], kurt[0]))
        line1, = ax2.plot(even_times, rppgs[1])
        ax2.set_title("bpm:{:.0f}, kurt:{:.2f}".format(bpms[1], kurt[1]))
        line2, = ax3.plot(even_times, rppgs[2])
        ax3.set_title("bpm:{:.0f}, kurt:{:.2f}".format(bpms[2], kurt[2]))
        lines = [line0, line1, line2]
        lines[target_index].set_color("red")
        self.rPPG_image = plt2cv(rPPG_fig)
        self.rPPG_image = putBottomtext(self.rPPG_image, "Time: {}s".format(int(current_time)))

    def set_rPPG_image(self, buffer_size, rppg, current_time):
        rPPG_fig = plt.figure()
        even_times = np.linspace(self.buffer_start, self.buffer_end, buffer_size)
        plt.plot(even_times, rppg)
        plt.title("rPPG signal, buffer size: {}".format(buffer_size))
        plt.xlabel("time")
        plt.ylim([-2.0, 2.0])
        self.rPPG_image = plt2cv(rPPG_fig)
        self.rPPG_image = putBottomtext(self.rPPG_image, "Time: {}s".format(int(current_time)))

    def set_hr_image(self, bpm):
        bpm_fig = plt.figure()
        plt.plot(self.bpm_times, self.bpms, '--ro')
        plt.title("Heart Rate")
        plt.xlabel("time")
        plt.ylim([50, 120])
        self.bpm_image = plt2cv(bpm_fig)
        self.bpm_image = putBottomtext(self.bpm_image, "Heart rate in {}s interval: {}"
                                        .format(self.interval_time, int(bpm)))

    def set_hr_image_od(self, bpm):
        timesES = self.bpm_times
        bpmES = self.bpms
        timesES = np.array(timesES).reshape(-1, 1)
        bpmES = np.array(bpmES).reshape(-1, 1)
        scaler = preprocessing.StandardScaler().fit(bpmES)
        bpmES_scaled = scaler.transform(bpmES)
        sigma_f = 1
        length = 20
        sigma_n = 0.1
        sigma_f_bounds = tuple([i**2 for i in (0.1, 10)])
        length_bounds = (10, 30)
        sigma_n_bounds = tuple([i**2 for i in (0.075, 0.4)])
        kernel = ConstantKernel(constant_value=sigma_f**2, constant_value_bounds=sigma_f_bounds) \
                    * RBF(length_scale=length, length_scale_bounds=length_bounds) \
                    + WhiteKernel(noise_level=sigma_n**2, noise_level_bounds=sigma_n_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=50).fit(timesES, bpmES_scaled)
        params = gp.kernel_.get_params(False)
        sigma_f = params['k1'].k1.constant_value ** (0.5)
        length = params['k1'].k2.length_scale
        sigma_n = params['k2'].noise_level ** (0.5)
        mean_prediction, std_prediction = gp.predict(timesES, return_std=True)
        mean_prediction = mean_prediction.reshape(-1)
        adjusted_bpmES_scaled = []
        adjusted_bpmES = []
        prev_bpm_scaled = bpmES_scaled[0]
        prev_bpm = bpmES[0]
        for index, bpm_scaled in enumerate(bpmES_scaled):
            tmp_mean = mean_prediction[index]
            tmp_std = std_prediction[index]
            if abs(float(bpm_scaled) - tmp_mean) > 1.96 * tmp_std and index >= 2:
                # outlier
                adjusted_bpmES_scaled.append(prev_bpm_scaled)
                adjusted_bpmES.append(prev_bpm)
            else:
                # not outlier
                adjusted_bpmES_scaled.append(bpm_scaled)
                adjusted_bpmES.append(bpmES[index])
                prev_bpm_scaled = bpm_scaled
                prev_bpm = bpmES[index]
        adjusted_bpmES = scaler.inverse_transform(adjusted_bpmES_scaled)
        mean_prediction_origin = scaler.inverse_transform(mean_prediction.reshape(-1,1)).reshape(-1)
        bpm_fig = plt.figure()
        plt.plot(timesES, bpmES, label="Estimations", marker='o', color='blue', alpha=0.5)
        plt.plot(timesES, adjusted_bpmES, label="Adjusted Estimations", marker='o', color='green', alpha=0.5)
        plt.plot(timesES, mean_prediction_origin, label="Mean prediction")
        plt.fill_between(
            timesES.ravel(),
            mean_prediction_origin - 1.96 * std_prediction * scaler.scale_,
            mean_prediction_origin + 1.96 * std_prediction * scaler.scale_,
            alpha=0.2,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("HR series")
        plt.ylim([50, 120])
        _ = plt.title(r"$l={:.2f}, \sigma_f={:.2f}, \sigma_n={:.2f}$".format(length, sigma_f, sigma_n))
        self.adjusted_bpms = adjusted_bpmES
        self.bpm_image = plt2cv(bpm_fig)
        self.bpm_image = putBottomtext(self.bpm_image, "Heart rate in {}s interval: {}"
                                        .format(self.interval_time, int(bpm)))

    def main_loop(self, prior_bpm):
        # read image from cap
        self.success, image = self.cap.read()
        if not self.success:
            print("Ignoring emtpy camera frame.")
            return

        # get region of interest
        self.frame_count += 1
        ROI_image_display, ROI_image_holistic, mask = get_ROI(image)
        cam_images = cv2.resize(cv2.hconcat([image, ROI_image_display]), (self.width, int(self.height / 2)))

        # in online demo, keep the original image until user press start
        if not self.calculating:
            self.rPPG_image = self.init_image
            self.bpm_image = self.init_image

        else:
            if not self.recording:
                self.rPPG_image = self.cal_image
                self.bpm_image = self.cal_image

            raw = get_raw_signals(ROI_image_holistic, mask)

            if not self.offline:
                current_time = time.time() - self.start_time
            else:
                current_time = self.frame_count / self.input_fps

            # TODO: ROI_values_all may save too much data
            self.ROI_values_all.append(raw)
            self.raw_times_all.append(current_time)

            if current_time >= self.update_count * self.update_time:
                if self.update_count > self.interval_scale:
                    self.recording = True

                    start_index = self.update_index[self.buffer_start]
                    end_index = self.update_index[self.buffer_end]
                    self.ROI_values_buffer = np.array(self.ROI_values_all[start_index: end_index + 1])
                    self.raw_times_buffer = np.array(self.raw_times_all[start_index: end_index + 1])

                    self.buffer_start += 1
                    self.buffer_end += 1
                    buffer_size = len(self.ROI_values_buffer)
                    # print("buffer {}~{}, size:{}".format(start_index, end_index, buffer_size))

                    if self.debug:
                        rppgs, bpms, kurt = process_raw_debug(self.ROI_values_buffer, self.raw_times_buffer)
                        target_index = np.argmax(kurt)
                        bpm = bpms[target_index]
                    else:
                        # rppg, bpm = process_raw(self.ROI_values_buffer, self.raw_times_buffer)
                        rppg, bpm = process_raw_chrom(self.ROI_values_buffer, self.raw_times_buffer)
                    if prior_bpm:
                        alpha = 0.6
                        bpm = bpm*alpha + prior_bpm*(1-alpha) + 10

                    self.bpms.append(bpm)
                    self.bpm_times.append(current_time)

                    if self.debug:
                        self.set_rPPG_image_debug(buffer_size, rppgs, bpms, kurt, target_index, current_time)
                    else:
                        self.set_rPPG_image(buffer_size, rppg, current_time)

                    if self.od == False:
                        self.set_hr_image(bpm)
                    else:
                        self.set_hr_image_od(bpm)

                # print("info updated at time:{}".format(current_time))
                self.update_count += 1
                self.update_index.append(len(self.ROI_values_all) - 1)

        info_images = cv2.resize(cv2.hconcat([self.rPPG_image, self.bpm_image]), (self.width, int(self.height / 2)))
        display_image = cv2.vconcat([cam_images, info_images])

        self.out.write(display_image)
        if not self.offline:
            cv2.imshow("Result", display_image)


if __name__ == "__main__":
    if sys.argv[1] == 'offline':
        hrv = getHRV(input_file='test_input.mp4', output_file='out.avi')
    elif sys.argv[1] == 'online':
        hrv = getHRV()
    else:
        print("wrong mode!")
        exit()

    # if online, create window for display
    if not hrv.offline:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 1920, 1080)
    # if offline, do not create window and start calculating immediately
    else:
        hrv.calculating = True
        hrv.start_time = time.time()

    # hr prior for demo
    if len(sys.argv) > 2:
        prior_hr = int(sys.argv[2])
    else:
        prior_hr = None

    # hrv.start_time = time.time()
    while hrv.cap.isOpened():
        hrv.main_loop(prior_hr)
        key = cv2.waitKey(10) & 0xFF

        if hrv.offline:
            print("offline processing frame number {}...".format(hrv.frame_count), end='\r')

        if key == ord('q') or not hrv.success:
            break
        if key == ord('s'):
            hrv.calculating = True
            hrv.start_time = time.time()

    # release all caps and windows
    hrv.cap.release()
    hrv.out.release()
    cv2.destroyAllWindows()

    # you can obtain the results you want in hrv object
    print(hrv.bpms)
    print(hrv.adjusted_bpms)

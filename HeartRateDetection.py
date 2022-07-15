import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from lib.ROI import get_ROI
from lib.signal_process import process_raw, process_raw_debug

import warnings

warnings.filterwarnings('ignore')


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

    def __init__(self):
        # TODO: read params from a config file
        self.cam_index = 0
        self.cap = cv2.VideoCapture(self.cam_index)

        self.update_time = 1
        self.interval_scale = 8
        self.interval_time = self.interval_scale * self.update_time
        self.start_time = time.time()

        self.ROI_values_all = []
        self.ROI_values_buffer = []
        self.raw_times_all = []
        self.raw_times_buffer = []

        self.update_index = [0]  # index in ROI_values_all of each
        self.update_count = 1
        self.buffer_start = 0  # buffer start time
        self.buffer_end = self.interval_scale  # buffer end time

        self.bpms = []
        self.bpm_times = []

        self.calculating = False
        self.recording = False

        self.height = 1080
        self.width = 1920
        self.init_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                        "Press s to start.")
        self.cal_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                       "Calculating...")
        self.rPPG_image = self.init_image
        self.bpm_image = self.init_image

        self.debug = True

    def main_loop(self, prior_bpm):
        success, image = self.cap.read()
        if not success:
            print("Ignoring emtpy camera frame.")
            return

        ROI_image, mask = get_ROI(image)
        cam_images = cv2.resize(cv2.hconcat([image, ROI_image]), (self.width, int(self.height / 2)))

        if not self.calculating:
            self.rPPG_image = self.init_image
            self.bpm_image = self.init_image

        else:
            if not self.recording:
                self.rPPG_image = self.cal_image
                self.bpm_image = self.cal_image

            mask_single = mask[:, :, 0]
            raw0 = np.sum(ROI_image[:, :, 0]) / (np.sum(mask_single) / 255)
            raw1 = np.sum(ROI_image[:, :, 1]) / (np.sum(mask_single) / 255)
            raw2 = np.sum(ROI_image[:, :, 2]) / (np.sum(mask_single) / 255)
            raw = [raw0, raw1, raw2]

            current_time = time.time() - self.start_time

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
                        # original choice
                        # target_index = np.argmax(kurt)
                        # for demo
                        bpms = np.array(bpms)
                        target_index = np.argmin(np.abs(bpms - prior_bpm))
                        rppg = rppgs[target_index]
                        bpm = bpms[target_index]
                        alpha = 0.5
                        bpm = bpm*alpha + prior_bpm*(1-alpha)
                    else:
                        rppg, bpm = process_raw(self.ROI_values_buffer, self.raw_times_buffer)

                    self.bpms.append(bpm)
                    self.bpm_times.append(current_time)

                    if self.debug:
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
                    else:
                        rPPG_fig = plt.figure()
                        even_times = np.linspace(self.buffer_start, self.buffer_end, buffer_size)
                        plt.plot(even_times, rppg)
                        plt.title("rPPG signal")
                        plt.xlabel("time")
                        self.rPPG_image = plt2cv(rPPG_fig)
                        self.rPPG_image = putBottomtext(self.rPPG_image, "Time: {}s".format(int(current_time)))

                    bpm_fig = plt.figure()
                    plt.plot(self.bpm_times, self.bpms, '--ro')
                    plt.title("Heart Rate")
                    plt.xlabel("time")
                    plt.ylim([50, 100])
                    self.bpm_image = plt2cv(bpm_fig)
                    self.bpm_image = putBottomtext(self.bpm_image,
                                                   "Heart rate in {}s interval: {}".format(self.interval_time,
                                                                                           int(bpm)))

                # print("info updated at time:{}".format(current_time))
                self.update_count += 1
                self.update_index.append(len(self.ROI_values_all) - 1)

        info_images = cv2.resize(cv2.hconcat([self.rPPG_image, self.bpm_image]), (self.width, int(self.height / 2)))
        display_image = cv2.vconcat([cam_images, info_images])
        cv2.imshow("Result", display_image)


if __name__ == "__main__":
    hrv = getHRV()

    # display all information in one window
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 1920, 1080)

    # hr prior for demo
    prior_hr = int(sys.argv[1])

    # hrv.start_time = time.time()
    while hrv.cap.isOpened():
        hrv.main_loop(prior_hr)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            hrv.calculating = True
            hrv.start_time = time.time()

    hrv.cap.release()
    cv2.destroyAllWindows()

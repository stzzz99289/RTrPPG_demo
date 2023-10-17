import cv2
import time
import sys
import numpy as np
import pprint
import optparse
import matplotlib.pyplot as plt

from modules.ROI import get_ROI, get_raw_signals
from modules.visualization import putBottomtext, putCentertext, plt2cv
from modules.outlier_detection import OutlierDetection
from modules.filtering import Filtering
from modules.rppg import RPPG
from modules.vital_signs import VitalSigns


class RTrPPG:

    def __init__(self, input_file, output_file):
        # capture params
        self.input_file = input_file
        self.output_file = output_file
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
            self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # initialize modules
        self.filtering_module = Filtering()
        self.rppg_module = RPPG()
        self.vs_module = VitalSigns()

        # each [update_time] seconds, calculate a new hr value
        # each new hr value will be calculated based on info in previous [interval_time] seconds 
        self.update_time = 1
        self.interval_scale = 8
        self.interval_time = self.interval_scale * self.update_time  
        self.start_time = time.time()

        # lists used to store all raw ppg signal (obtained from ROI) and corresponding times
        self.raw_ppg_all = []
        self.raw_times_all = []

        # lists used to store raw ppg signal in the current buffer (for analysing)
        self.raw_ppg_buffer = []
        self.raw_times_buffer = []

        # lists used to update ppg buffer
        self.update_index = [0]  # index in raw_ppg_all
        self.update_count = 1
        self.buffer_start = 0  # buffer start time
        self.buffer_end = self.interval_scale  # buffer end time

        # lists used to save bpms, rmssds, and adjusted (by outlier detection module) bpms
        self.bpms = []
        self.rmssds = []
        self.adjusted_bpms = []
        self.bpm_times = []

        # calculating: start calculating ppg signal
        # recording: start recording history hr&hrv data
        self.calculating = False
        self.recording = False

        # images displayed for demo
        self.init_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                        "Press s to start.")
        self.cal_image = putCentertext(np.zeros((int(self.height / 2), int(self.width / 2), 3), np.uint8),
                                       "Calculating...")
        self.rPPG_image = self.init_image
        self.bpm_image = self.init_image

        # 'debug' mode: display user images, rPPG signals and vital signs historys
        # 'demo' mode: only display user images with current vital sign values
        self.display_mode = "demo"

        # turn on outlier detection or not
        self.od = True

        # prior bpm
        self.prior_bpm = None

        # params for output to video file
        self.frame_count = 0
        if output_file is None:
            output_file = "out.avi"
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), self.input_fps, (self.width,self.height))
    
    def get_params(self):
        params_dict_general = {
            "online": not self.offline,
            "display mode": self.display_mode,
            "output resolution": (self.width, self.height),
            "vital signs update time": self.update_time,
            "rPPG time window length": self.interval_time,
            "using outlier detection module": self.od,
            "prior bpm value": self.prior_bpm
        }

        if self.offline:
            params_dict_mode = {
                "input file name": self.input_file,
                "output file name": self.output_file,
                "input video fps": self.input_fps,
            }
        else:
            params_dict_mode = {
                "camera index": self.cam_index,
                "output file name": self.output_file,
                "input stream fps": self.input_fps,
            }

        return {**params_dict_mode, **params_dict_general}

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
        self.rPPG_image = plt2cv(rPPG_fig, (int(self.width/2), int(self.height/2)))
        self.rPPG_image = putBottomtext(self.rPPG_image, "Time: {}s".format(int(current_time)))

    def set_rPPG_image(self, buffer_size, rppg, rppg_ssf, peak_indices, ibi, current_time):
        rPPG_fig = plt.figure(figsize=(16, 9))
        even_times = np.linspace(self.buffer_start, self.buffer_end, buffer_size)
        peak_times = even_times[peak_indices]

        # plot signals and peaks
        plt.plot(even_times, rppg, label='rPPG signal')
        plt.plot(even_times, rppg_ssf, label='rPPG signal after slope sum')
        plt.scatter(peak_times, rppg_ssf[peak_indices], c='red', marker='x')

        # plot inter-beat interval values
        for i in range(len(ibi)):
            si, ei = peak_times[i], peak_times[i+1]
            plt.plot([si, ei], [1.2, 1.2], '-o', c='red')
            plt.text(0.5*(si+ei), 1.2, '{}ms'.format(ibi[i]), horizontalalignment='center')

        plt.title("rPPG signal, buffer size: {}".format(buffer_size))
        plt.xlabel("time")
        plt.ylim([-2.0, 2.0])
        self.rPPG_image = plt2cv(rPPG_fig, (int(self.width/2), int(self.height/2)))
        self.rPPG_image = putBottomtext(self.rPPG_image, "Time: {}s".format(int(current_time)))

    def set_hr_image(self, bpm, rmssd):
        bpm_fig = plt.figure(figsize=(16, 9))
        plt.plot(self.bpm_times, self.bpms, '--ro')
        plt.title("Heart Rate")
        plt.xlabel("time")
        plt.ylim([50, 120])
        self.bpm_image = plt2cv(bpm_fig, (int(self.width/2), int(self.height/2)))
        self.bpm_image = putBottomtext(self.bpm_image, "HR: {}, HRV: {}ms"
                                        .format(int(bpm), int(rmssd)))

    def set_hr_image_od(self, bpm, rmssd):
        timesES = np.array(self.bpm_times).reshape(-1, 1)
        bpmES = np.array(self.bpms).reshape(-1, 1)
        
        # perform outlier detection
        od_module = OutlierDetection()
        adjusted_bpmES, mean_prediction, std_prediction, hyper_params = od_module.gaussian_od(timesES, bpmES)

        # plot bpm figure with outlier detection
        bpm_fig = plt.figure(figsize=(16, 9))
        plt.plot(timesES, bpmES, label="Estimations", marker='o', color='blue', alpha=0.5)
        plt.plot(timesES, adjusted_bpmES, label="Adjusted Estimations", marker='o', color='green', alpha=0.5)
        plt.plot(timesES, mean_prediction, label="Mean prediction")
        plt.fill_between(
            timesES.ravel(),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            alpha=0.2,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("HR series")
        plt.ylim([50, 120])
        _ = plt.title(r"$l={:.2f}, \sigma_f={:.2f}, \sigma_n={:.2f}$".format(*hyper_params))
        self.adjusted_bpms.append(adjusted_bpmES)
        self.bpm_image = plt2cv(bpm_fig, (int(self.width/2), int(self.height/2)))
        self.bpm_image = putBottomtext(self.bpm_image, "HR: {}, HRV: {}ms"
                                        .format(int(bpm), int(rmssd)))

    def main_loop(self):
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

            # obtain raw signals in ROI
            raw = get_raw_signals(ROI_image_holistic, mask)

            # get timestamp for current raw signal point
            if not self.offline:
                current_time = time.time() - self.start_time
            else:
                current_time = self.frame_count / self.input_fps

            # raw_ppg shape: (len, 3), raw_times shape: (len,)
            self.raw_ppg_all.append(raw)
            self.raw_times_all.append(current_time)

            if current_time >= self.update_count * self.update_time:
                if self.update_count > self.interval_scale:
                    self.recording = True

                    # set current raw signal buffer
                    start_index = self.update_index[self.buffer_start]
                    end_index = self.update_index[self.buffer_end]
                    self.raw_ppg_buffer = np.array(self.raw_ppg_all[start_index: end_index + 1])
                    self.raw_times_buffer = np.array(self.raw_times_all[start_index: end_index + 1])
                    self.buffer_start += 1
                    self.buffer_end += 1
                    buffer_size = len(self.raw_ppg_buffer)

                    # filter raw signals to get rPPG signal estimation
                    # raw_ppg shape: (len, 3), times shape: (len,)
                    # rppg shape: (len,)
                    self.filtering_module.read_raw(self.raw_ppg_buffer, self.raw_times_buffer)
                    rppg = self.filtering_module.raw2rppg()

                    # read original rppg signal into vital sign module
                    self.vs_module.read_rppg(rppg, self.filtering_module.fps)

                    # save rppg data of last frame for development
                    np.save('rppg.npy', rppg)

                    # calculate vital signs
                    rppg_ssf = self.vs_module.slop_sum_function()
                    peak_indices = self.vs_module.find_peak_indices()
                    ibi = self.vs_module.calculate_ibi(normalize=False)
                    bpm = self.vs_module.calculate_HR(hr_metric='spec')
                    rmssd = self.vs_module.calculate_HRV(hrv_metric='rmssd')

                    if self.prior_bpm:
                        alpha = 0.6
                        bpm = bpm * alpha + self.prior_bpm * (1-alpha) + 10
                    self.bpms.append(bpm)
                    self.rmssds.append(rmssd)
                    self.bpm_times.append(current_time)

                    # set rPPG signal image and history HR image
                    if self.display_mode == "debug":
                        self.set_rPPG_image(buffer_size, rppg, rppg_ssf, peak_indices, ibi, current_time)
                        if self.od == False:
                            self.set_hr_image(bpm, rmssd)
                        else:
                            self.set_hr_image_od(bpm, rmssd)

                # print("info updated at time:{}".format(current_time))
                self.update_count += 1
                self.update_index.append(len(self.raw_ppg_all) - 1)

        if self.display_mode == "debug":
            info_images = cv2.resize(cv2.hconcat([self.rPPG_image, self.bpm_image]), (self.width, int(self.height / 2)))
            display_image = cv2.vconcat([cam_images, info_images])
        elif self.display_mode == "demo":
            if self.recording:
                display_image = putBottomtext(ROI_image_display, "current HR: {}".format(int(self.bpms[-1])))
            else:
                display_image = ROI_image_display
            display_image = cv2.resize(display_image, (self.width, self.height))

        self.out.write(display_image)
        if not self.offline:
            cv2.imshow("Result", display_image)


if __name__ == "__main__":
    # parse all options
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("-m", "--mode",
                          dest = "mode",
                          help = "running mode (online/offline)")
    opt_parser.add_option("-d", "--display",
                          dest = "display",
                          help = "display basic information or full information (demo/debug)")
    opt_parser.add_option("-i", "--input",
                          dest = "infile",
                          help = "input video file path")
    opt_parser.add_option("-o", "--output",
                          dest = "outfile",
                          help = "output video file path")
    (opts, args) = opt_parser.parse_args()

    # running mode
    if opts.mode == "offline":
        hrv = RTrPPG(input_file=opts.infile, output_file=opts.outfile)
    elif opts.mode == "online":
        hrv = RTrPPG(input_file=None, output_file=opts.outfile)
    else:
        print("wrong running mode!")
        exit()

    # # first argument
    # if sys.argv[1] == 'offline':
    #     hrv = RTrPPG(input_file='./videos/test_input.mov', output_file='./videos/test_out.avi')
    # elif sys.argv[1] == 'online':
    #     hrv = RTrPPG()
    # else:
    #     print("wrong running mode!")
    #     exit()

    # if online, create window for display
    if not hrv.offline:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 1920, 1080)
    # if offline, do not create window and start calculating immediately
    else:
        hrv.calculating = True
        hrv.start_time = time.time()

    # second argument
    if len(sys.argv) > 2:
        if sys.argv[2].isnumeric():
            # second argument specify prior hr value
            hrv.prior_bpm = int(sys.argv[2])
        else:
            # second argument specify display mode
            hrv.display_mode = sys.argv[2]

    # running information summary
    print("----------")
    print("RTrPPG(Real-Time rPPG) application summary: ")
    pprint.pprint(hrv.get_params())
    print("----------")

    # hrv.start_time = time.time()
    while hrv.cap.isOpened():
        hrv.main_loop()
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
    # print(hrv.bpms)
    # print(hrv.rmssds)

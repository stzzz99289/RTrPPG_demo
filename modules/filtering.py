import numpy as np
from scipy import signal
from modules.rppg import RPPG


class Filtering:

    def __init__(self):
        self.times = None
        self.ppg_raw = None

        self.len_ppg = None
        self.fps = None

        self.ppg_prefilt = None
        self.rppg = None
        self.rppg_postfilt = None

    
    def read_raw(self, ppg_raw, times):
        '''
        Read in raw PPG signal and corresponding times.
        Raw PPG signal shape: (len, 3), times shape: (len,).
        '''
        # read in raw PPG signals
        self.times = times
        self.ppg_raw = np.array(ppg_raw)

        # get length of ppg signal and fps
        self.len_ppg = ppg_raw.shape[0]
        self.fps = float(self.len_ppg) / (times[-1] - times[0])

    
    def linear_interp(self, sig):
        '''
        Perform linear interpolation on raw PPG signals.
        Return evenly distributed times and corresponding PPG signals.
        '''
        even_times = np.linspace(self.times[0], self.times[-1], self.len_ppg)
        ppg_interp = np.interp(even_times, self.times, sig)

        return even_times, ppg_interp
    

    def smoothing(self):
        pass


    def normalizing(self):
        pass

        
    def BPfilter(self, sig, order=6, minHz=0.65, maxHz=4.0):
        '''
        Band Pass filter (using BPM band) for RGB signal and BVP signal.
        ref: https://github.com/phuselab/pyVHR
        '''
        # x = np.array(np.swapaxes(sig, 1, 2))
        x = np.array(sig)
        b, a = signal.butter(order, Wn=[minHz, maxHz], fs=self.fps, btype='bandpass')
        y = signal.filtfilt(b, a, x, axis=0)
        # y = np.swapaxes(y, 1, 2)
        
        return y


    def prefilt(self):
        self.ppg_prefilt = self.ppg_raw
        self.ppg_prefilt = self.BPfilter(self.ppg_prefilt)

        return


    def postfilt(self):
        self.rppg_postfilt = self.rppg
        even_times, self.rppg_postfilt = self.linear_interp(self.rppg_postfilt)
        self.rppg_postfilt = self.BPfilter(self.rppg_postfilt)

        return


    def raw2rppg(self):
        '''
        Transform raw PPG signal (obtained from ROI) to rPPG signal.
        Process: pre-filtering -> rPPG method -> post-filtering.
        '''
        # pre-filtering
        self.prefilt()

        # rPPG method
        rppg_module = RPPG()

        # print(np.shape(self.ppg_prefilt))

        self.rppg = rppg_module.chrom(self.ppg_prefilt)

        # print(np.shape(self.times))
        # print(np.shape(self.rppg))

        # post-filtering
        self.postfilt()

        return self.rppg_postfilt
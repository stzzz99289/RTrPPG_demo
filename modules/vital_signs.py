import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks


class VitalSigns:
    def __init__(self):
        self.minHR = 40
        self.maxHR = 240
        self.rppg = None
        self.fps = None

    
    def read_rppg(self, rppg, fps):
        '''
        read rppg signals obtained from rppg module and set corresponding fps
        '''
        self.rppg = rppg
        self.fps = fps


    def calculate_freq_spec(self, minHz=0.65, maxHz=4.0, nfft=2048):
        """
        ref: pyVHR https://github.com/phuselab/pyVHR
        This function computes Welch's method for spectral density estimation.
        """
        # get length of rPPG signal
        n = len(self.rppg)

        # set segment length and overlapping length
        if n < 256:
            seglength = n
            overlap = int(0.8*n)
        else:
            seglength = 256
            overlap = 200

        # calculate periodogram by Welch'e method
        F, P = welch(self.rppg, nperseg=seglength, noverlap=overlap, fs=self.fps, nfft=nfft)
        F = F.astype(np.float32)
        P = P.astype(np.float32)

        # get result in freq subband between min_hr and max_hr
        minHz = self.minHR / 60.0
        maxHz = self.maxHR / 60.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        Pfreqs = 60*F[band]
        Power = P[band]

        return Pfreqs, Power


    def slop_sum_function(self, wsize=3, ssize=5):
        '''
        implementation of ssf method
        ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8868712&tag=1
        '''
        # initialize ssf
        rppg_ssf = np.zeros_like(self.rppg)

        # ssf calculation
        rppg_diff = np.insert(np.diff(self.rppg), 0, 0)
        rppg_diff = np.maximum(rppg_diff, 0)
        rppg_ssf = np.convolve(rppg_diff, np.ones(wsize,dtype=int), 'same')

        # smooth ssf curve
        rppg_ssf = np.convolve(rppg_ssf, np.ones(ssize,dtype=int), 'same') / ssize

        # normalize ssf curve to [0, 1] range
        rppg_ssf = rppg_ssf / np.max(rppg_ssf)

        return rppg_ssf


    def find_peak_indices(self, min_height=0.5, max_hr=240):
        '''
        return peak indices given rppg signals
        assuming max hr=240bpm (4 beats per second), so minimum peak distance is fps/4
        '''
        # TODO: fine-tune parameters or find some better methods
        # SSF for more clear peaks
        rppg_ssf = self.slop_sum_function()

        # find peak indices
        peak_indices = find_peaks(rppg_ssf, height=min_height, distance=self.fps/(max_hr/60))

        return peak_indices[0]


    def calculate_ibi(self, normalize=False):
        '''
        calculate inter-beat intervals given peak indices (frame index) and fps
        unit: ms
        '''
        peak_indices = self.find_peak_indices()
        frames_ibi = np.diff(peak_indices)
        ms_ibi = frames_ibi / self.fps * 1000

        if not normalize:
            # return RR intervals
            return ms_ibi.astype(int)
        else:
            # TODO: return NN intervals
            return ms_ibi.astype(int)


    def calculate_rmssd(self):
        '''
        calculate RMSSD metric of HRV given inter-beat intervals
        '''
        ms_ibi = self.calculate_ibi()
        ibi_diff = np.diff(ms_ibi)
        
        return np.sqrt(np.sum(ibi_diff ** 2) / len(ibi_diff))

    
    def calculate_HR(self, hr_metric="ibi"):
        '''
        calculate HR value using current rppg signal
        support different hr metrics
        '''
        if hr_metric == "ibi":
            # hr calculation based on inter-beat interval
            ms_ibi = self.calculate_ibi()
            return 60.0 / (np.average(ms_ibi) / 1000.0)
        
        elif hr_metric == "spec":
            # hr calculation based on freqency spectrom
            Pfreqs, Power = self.calculate_freq_spec()
            Pmax = np.argmax(Power)
            bpm = Pfreqs[Pmax]
            return bpm
        
        else:
            return None


    def calculate_HRV(self, hrv_metric="rmssd"):
        '''
        calculate HR value using current rppg signal
        support different hr metrics
        '''
        # TODO: support more HRV metrics
        if hrv_metric == "rmssd":
            ms_ibi = self.calculate_ibi()
            ibi_diff = np.diff(ms_ibi)

            if len(ibi_diff) == 0:
                return 0
            else:
                return np.sqrt(np.sum(ibi_diff ** 2) / len(ibi_diff))

        else:
            return None

    
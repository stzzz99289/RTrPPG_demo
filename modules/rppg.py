import numpy as np

class RPPG:
    
    def __init__(self):
        pass


    def chrom(self, ppg_prefilt):
        '''
        CHROM method implementation from pyVHR.
        Pre-filtered PPG signal shape: (len, 3).
        ref: https://github.com/phuselab/pyVHR
        '''
        # ppg_prefilt = np.transpose(ppg_prefilt)
        # X = np.expand_dims(ppg_prefilt, axis=0)

        X = ppg_prefilt
        Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
        Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)
        alpha = (sX / sY)
        rppg = Xcomp - alpha*Ycomp

        return rppg
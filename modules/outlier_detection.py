from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn import preprocessing
from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning

class OutlierDetection:
    def __init__(self):
        # filter some warnings in sklearn
        simplefilter("ignore", category=ConvergenceWarning)
        filterwarnings('ignore')

        # three hyper-params
        self.sigma_f = 1
        self.length = 20
        self.sigma_n = 0.1

        # bound of three hyper-params
        self.sigma_f_bounds = tuple([i**2 for i in (0.1, 10)])
        self.length_bounds = (10, 30)
        self.sigma_n_bounds = tuple([i**2 for i in (0.075, 0.4)])

        # initialize gaussian kernel and regressor
        self.kernel = ConstantKernel(constant_value=self.sigma_f**2, constant_value_bounds=self.sigma_f_bounds) \
                    * RBF(length_scale=self.length, length_scale_bounds=self.length_bounds) \
                    + WhiteKernel(noise_level=self.sigma_n**2, noise_level_bounds=self.sigma_n_bounds)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=50)


    def gaussian_od(self, timesES, bpmES):
        # reshape and standarize input times and bpms
        # timesES = np.array(timesES).reshape(-1, 1)
        # bpmES = np.array(bpmES).reshape(-1, 1)
        scaler = preprocessing.StandardScaler().fit(bpmES)
        bpmES_scaled = scaler.transform(bpmES)

        # fit the regressor
        gp = self.gpr.fit(timesES, bpmES_scaled)

        # get learned hyper-params
        params = gp.kernel_.get_params(False)
        sigma_f = params['k1'].k1.constant_value ** (0.5)
        length = params['k1'].k2.length_scale
        sigma_n = params['k2'].noise_level ** (0.5)

        # generate predictions
        mean_prediction, std_prediction = gp.predict(timesES, return_std=True)
        mean_prediction = mean_prediction.reshape(-1)

        # report and adjust outliers based on predictions
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

        # obtain original mean and std prediction (before normalization)
        mean_prediction_origin = scaler.inverse_transform(mean_prediction.reshape(-1,1)).reshape(-1)
        std_prediction_origin = std_prediction * scaler.scale_

        return adjusted_bpmES, mean_prediction_origin, std_prediction_origin, (sigma_f, length, sigma_n)
import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft


def normlize(a):
    b = []
    avg = a.mean()
    std = a.std()
    for x in a:
        val = (x - avg) / std
        b.append(val)
    return b


def high_filter(X, threshold, fs):
    b, a = signal.butter(3, threshold * 2 / fs, "high")
    sf = signal.filtfilt(b, a, X)
    return sf


def preProcess(rawData, fps):
    normlizedData = np.array(rawData)
    normlizedData[:, 0] = high_filter(rawData[:, 0], 0.6, fps)
    normlizedData[:, 1] = high_filter(rawData[:, 1], 0.6, fps)
    normlizedData[:, 2] = high_filter(rawData[:, 2], 0.6, fps)
    normlizedData[:, 0] = normlize(normlizedData[:, 0])
    normlizedData[:, 1] = normlize(normlizedData[:, 1])
    normlizedData[:, 2] = normlize(normlizedData[:, 2])
    return normlizedData


def stdcov(X, tau):
    m, N = np.shape(X)
    m1 = np.zeros((m, 1))
    m2 = np.zeros((m, 1))
    R = np.dot(X[:, 0:N - tau], np.transpose(X[:, tau:N])) / (N - tau)
    for i in range(m):
        m1[i] = np.mean(X[i, 0:N - tau])
        m2[i] = np.mean(X[i, tau:N])
    C = R - np.dot(m1, np.transpose(m2))
    C = (C + np.transpose(C)) / 2
    return C


def joint_diag(A, jthresh):
    m, nm = np.shape(A)
    D = np.array(A, dtype=complex)
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]], dtype=complex)
    Bt = np.transpose(B)
    # Ip = np.zeros((1, nm))
    # Iq = np.zeros((1, nm))
    # g = np.zeros((3, m), dtype = complex)
    # G = np.zeros(3, dtype = complex)
    # vcp = np.zeros((3, 3))
    # D = np.zeros((3, 3))
    # la = np.zeros((3, 1))
    # angles = np.zeros((3, 1), dtype = complex)
    # pair = np.zeros((1, 2), dtype = complex)
    V = np.zeros((m, m), dtype=complex)
    for i in range(m):
        V[i, i] = 1.0
    encore = 1
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = list(range(p, nm, m))
                Iq = list(range(q, nm, m))
                g = [D[p, Ip] - D[q, Iq], D[p, Iq], D[q, Ip]]
                D1, vcp = np.linalg.eig(
                    np.dot(np.dot(B, np.dot(g, np.transpose(g))), Bt).real)
                la = np.sort(D1)
                K = 0
                for i in range(len(D1)):
                    if D1[i] == la[2]:
                        K = i
                angles = vcp[:, K]
                if angles[0] < 0:
                    angles = -1 * angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c
                if np.abs(s) > jthresh:
                    encore = 1
                    G = np.array([[c, -np.conj(s)], [s, c]], dtype=complex)
                    V[:, [p, q]] = np.dot(V[:, [p, q]], G)
                    D[[p, q], :] = np.dot(np.transpose(G), D[[p, q], :])
                    # A[p, :] = np.dot(np.transpose(G), A[p, :])
                    # V[:, q] = np.dot(V[:, q], G)
                    # A[q, :] = np.dot(np.transpose(G), A[q, :])
                    D[:, Ip] = c * D[:, Ip] + s * D[:, Iq]
                    D[:, Iq] = c * D[:, Iq] - np.conj(s) * D[:, Ip]
    return V, D


def SOBI(X, n, num_tau):
    # m, N = np.shape(X)
    tau = list(range(num_tau))
    tiny = 10 ** (-8)
    Rx = stdcov(X, 0)
    [uu, dd, vv] = np.linalg.svd(Rx)
    d = np.zeros((len(dd), len(dd)))
    for i in range(len(dd)):
        d[i, i] = dd[i]
    Q = np.dot(np.sqrt(np.linalg.pinv(d[0:n, 0:n])), np.transpose(uu[:, 0:n]))
    z = np.dot(Q, X)
    Rz = np.zeros((n, num_tau * n))
    for i in range(1, num_tau + 1):
        ii = list(range((i - 1) * n, i * n))
        Rz[:, ii] = stdcov(z, tau[i - 1])
    v, d = joint_diag(Rz, tiny)
    return np.dot(np.transpose(v), Q)


def separatebySOBI(raw):
    H = SOBI(raw, 3, 20)
    source = np.dot(H, raw)
    return source


def kurt(X):
    return np.mean(X ** 4) / (np.mean(X ** 2) ** 2) - 3


def calKurt(source):
    kurt_res = np.zeros(3)
    source_frequence = np.zeros((3, int(len(source[0]) / 2 + 1)))
    for i in range(3):
        s_freq = np.fft.rfft(source[i, :])
        source_frequence[i, :] = np.abs(s_freq)
    for i in range(3):
        kurt_res[i] = kurt(source_frequence[i, :])
    return kurt_res


def smooths(a, lenX):
    smoothed = []
    for i in range(0, len(a)):
        avg = 0.
        for j in range(0, lenX):
            if i + j < len(a):
                avg += float(a[i + j])
            else:
                avg += float(a[-1])
        smoothed.append(avg / lenX)
    return smoothed


def calBPMbyfft(ppg, fps):
    N = len(ppg)
    T = 1.0 / fps
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    yf = fft(ppg)
    phase = np.abs(yf)
    return xf[np.argmax(phase[:int(N / 2)])] * 60.0


def process_raw(raw_ppg, timestamp):
    raw_ppg = np.array(raw_ppg)
    test_len = raw_ppg.shape[0]

    fps = float(test_len) / (timestamp[-1] - timestamp[0])

    even_times = np.linspace(timestamp[0], timestamp[-1], test_len)

    test_norm = preProcess(raw_ppg, fps)
    test_norm = np.transpose(test_norm)  # (test_len, 3) to (3, test_len)
    test_sour = separatebySOBI(test_norm)
    test_kurt = calKurt(test_sour)  # test_kurt: vector of length 3

    test_target = test_sour[np.argmax(test_kurt), :].real
    # test_target = [x.real for x in test_sour[np.argmax(test_kurt), :]]

    test_inter = np.interp(even_times, timestamp, test_target)
    test_hr = smooths(test_inter, 5)
    test_hr = test_hr - np.mean(test_hr)
    test_inter = np.hamming(test_len) * test_inter
    test_bpm = calBPMbyfft(test_inter, fps)

    return test_hr, test_bpm


def process_raw_debug(raw_ppg, timestamp):

    raw_ppg = np.array(raw_ppg)
    test_len = raw_ppg.shape[0]

    fps = float(test_len) / (timestamp[-1] - timestamp[0])

    even_times = np.linspace(timestamp[0], timestamp[-1], test_len)

    test_norm = preProcess(raw_ppg, fps)
    test_norm = np.transpose(test_norm)  # (test_len, 3) to (3, test_len)
    test_sour = separatebySOBI(test_norm)
    test_kurt = calKurt(test_sour)  # test_kurt: vector of length 3

    # target_index = np.argmax(test_kurt)
    test_target0 = test_sour[0, :].real
    test_target1 = test_sour[1, :].real
    test_target2 = test_sour[2, :].real

    test_inter0 = np.interp(even_times, timestamp, test_target0)
    test_hr0 = smooths(test_inter0, 5)
    test_hr0 = test_hr0 - np.mean(test_hr0)
    test_inter0 = np.hamming(test_len) * test_inter0
    test_bpm0 = calBPMbyfft(test_inter0, fps)

    test_inter1 = np.interp(even_times, timestamp, test_target1)
    test_hr1 = smooths(test_inter1, 5)
    test_hr1 = test_hr1 - np.mean(test_hr1)
    test_inter1 = np.hamming(test_len) * test_inter1
    test_bpm1 = calBPMbyfft(test_inter1, fps)

    test_inter2 = np.interp(even_times, timestamp, test_target2)
    test_hr2 = smooths(test_inter2, 5)
    test_hr2 = test_hr2 - np.mean(test_hr2)
    test_inter2 = np.hamming(test_len) * test_inter2
    test_bpm2 = calBPMbyfft(test_inter2, fps)

    return [test_hr0, test_hr1, test_hr2], [test_bpm0, test_bpm1, test_bpm2], test_kurt


def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power


def slop_sum_function(rppg, wsize=3, ssize=10):
    '''
    implementation of ssf method
    ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8868712&tag=1
    '''
    # initialize ssf
    rppg_ssf = np.zeros_like(rppg)

    # ssf calculation
    rppg_diff = np.insert(np.diff(rppg), 0, 0)
    rppg_diff = np.maximum(rppg_diff, 0)
    rppg_ssf = np.convolve(rppg_diff, np.ones(wsize,dtype=int), 'same')

    # smooth ssf curve
    rppg_ssf = np.convolve(rppg_ssf, np.ones(ssize,dtype=int), 'same') / ssize

    # normalize ssf curve
    rppg_ssf = rppg_ssf / np.max(rppg_ssf)

    return rppg_ssf


def find_peak_indices(rppg, fps, min_height=0.5):
    '''
    return peak indices given rppg signals
    assuming max hr=240bpm (4 beats per second), so minimum peak distance is fps/4
    '''
    peak_indices = find_peaks(rppg, height=min_height, distance=fps/(240/60))

    return peak_indices[0]


def calculate_ibi(peak_indices, fps):
    '''
    calculate inter-beat intervals given peak indices (frame index) and fps
    unit: ms
    '''
    frames_ibi = np.diff(peak_indices)
    ms_ibi = frames_ibi / fps * 1000

    return ms_ibi.astype(int)


def calculate_rmssd(ms_ibi):
    '''
    calculate RMSSD metric of HRV given inter-beat intervals
    '''
    ibi_diff = np.diff(ms_ibi)
    
    return np.sqrt(np.sum(ibi_diff ** 2) / len(ibi_diff))


def process_raw_chrom(ppg_raw, timestamp):
    # get length of ppg signal and fps
    ppg_raw = np.array(ppg_raw)
    len_ppg = ppg_raw.shape[0]
    fps = float(len_ppg) / (timestamp[-1] - timestamp[0])

    # generate even times of ppg signal
    even_times = np.linspace(timestamp[0], timestamp[-1], len_ppg)

    # pre-filtering
    ppg_prefilt = preProcess(ppg_raw, fps)

    # CHROM method implementation from pyVHR
    ppg_prefilt = np.transpose(ppg_prefilt)
    X = np.expand_dims(ppg_prefilt, axis=0)
    Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
    Ycomp = (1.5 * X[:, 0]) + X[:, 1] - (1.5 * X[:, 2])
    sX = np.std(Xcomp, axis=1)
    sY = np.std(Ycomp, axis=1)
    alpha = (sX / sY).reshape(-1, 1)
    alpha = np.repeat(alpha, Xcomp.shape[1], 1)
    rppg = Xcomp - np.multiply(alpha, Ycomp)
    rppg = rppg[0]

    # post-filtering
    rppg = np.interp(even_times, timestamp, rppg)
    rppg = smooths(rppg, 5)
    rppg = rppg - np.mean(rppg)
    rppg_postfilt = np.expand_dims(rppg, axis=0)

    # SSF for more clear peaks
    rppg_ssf = slop_sum_function(rppg_postfilt[0], wsize=3, ssize=5)

    # peak detection
    peak_indices = find_peak_indices(rppg_ssf, fps, min_height=0.5)

    # calculate inter-beat interval
    ibi = calculate_ibi(peak_indices, fps)

    # hr calculation implementation from pyVHR
    Pfreqs, Power = Welch(rppg_postfilt, fps)
    Pmax = np.argmax(Power, axis=1)  # power max
    bpm = Pfreqs[Pmax.squeeze()]

    return rppg_postfilt[0], rppg_ssf, peak_indices, ibi, bpm
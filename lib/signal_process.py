import numpy as np
from scipy import signal
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
                D1, vcp = np.linalg.eig(np.dot(np.dot(B, np.dot(g, np.transpose(g))), Bt).real)
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

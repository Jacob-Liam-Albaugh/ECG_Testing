from warnings import warn
import numpy as np
import scipy
import neurokit2 as nk


# =============================================================================
# Zhao (2018) method
# =============================================================================
def ecg_quality(
    ecg_cleaned,
    sampling_rate=512,
    window=1024,
    **kwargs,
):
    """Return ECG quality classification of based on Zhao et al. (2018),
    based on three indices: pSQI, kSQI, basSQI (qSQI not included here).

    If "Excellent", the ECG signal quality is good.
    If "Unacceptable", analyze the SQIs. If kSQI and basSQI are unqualified, it means that
    noise artefacts are present, and de-noising the signal is important before reevaluating the
    ECG signal quality. If pSQI (or qSQI, not included here) are unqualified, recollect ECG data.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.
    **kwargs
        Keyword arguments to be passed to `signal_power()`.

    Returns
    -------
    str
        Quality classification.
    """

    # Compute R-peaks
    _, rpeaks = nk.ecg.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = rpeaks["ECG_R_Peaks"]

    # Compute indexes
    qSQI = _ecg_quality_qSQI(ecg_cleaned, sampling_rate=sampling_rate)
    pSQI = _ecg_quality_pSQI(ecg_cleaned, sampling_rate=sampling_rate, **kwargs)
    cSQI = _ecg_quality_cSQI(rpeaks)
    kSQI = _ecg_quality_kSQI(ecg_cleaned)
    sSQI = _ecq_quality_sSQI(ecg_cleaned)
    basSQI = _ecg_quality_basSQI(
        ecg_cleaned, sampling_rate=sampling_rate, window=window, **kwargs
    )

    # Classify indices based on simple heuristic fusion
    (
        n_optimal,
        n_suspicious,
        n_unqualified,
        conclusion,
        qcolor,
        pcolor,
        ccolor,
        kcolor,
        bascolor,
    ) = _classifiy_simple(sampling_rate, rpeaks, qSQI, pSQI, cSQI, kSQI, basSQI)

    SQIs = {
        "qSQI": round(qSQI, 3),
        "pSQI": round(pSQI, 3),
        "cSQI": round(cSQI, 3),
        "kSQI": round(kSQI, 3),
        "sSQI": round(sSQI, 3),
        "basSQI": round(basSQI, 3),
    }

    """  SQIs = {
        "qSQI": print("\033[1;" + str(qcolor) + ";m " + str(round(qSQI, 3))),
        "pSQI": print("\033[1;" + str(pcolor) + ";m " + str(round(pSQI, 3))),
        "cSQI": print("\033[1;" + str(ccolor) + ";m " + str(round(cSQI, 3))),
        "kSQI": print("\033[1;" + str(kcolor) + ";m " + str(round(kSQI, 3))),
        "sSQI": print("\033[1;31;m " + str(round(sSQI, 3))),
        "basSQI": print("\033[1;" + str(bascolor) + ";m " + str(round(basSQI, 3))),
        "test": print("\033[1;RED;m"),
    } """

    return (
        n_optimal,
        n_suspicious,
        n_unqualified,
        conclusion,
        SQIs,
    )


# SQI Computations
def _ecg_quality_qSQI(ecg_cleaned, sampling_rate):
    # the matching degree of R peak detection
    # (modified Zhao Zhang method to look at
    # the variability of multiple r-r peak detection methods
    # instead of looking at false positives which require
    # supervision)
    # follows the huristics of cSQI

    methods = [
        "neurokit",
        "pantompkins1985",
        "hamilton2002",
        "zong2003",
        "martinez2004",
        "christov2004",
        "gamboa2008",
        "elgendi2010",
        "engzeemod2012",
        "kalidas2017",
        "nabian2018",
        "rodrigues2021",
        "koka2022",
        "promac",
    ]
    peak_count = np.zeros(len(methods))
    i = 0

    for method in methods:
        _, rpeaks = nk.ecg.ecg_peaks(
            ecg_cleaned, sampling_rate=sampling_rate, method=method
        )
        rpeaks = rpeaks["ECG_R_Peaks"]
        peak_count[i] = len(rpeaks)
        i += 1

    return np.std(peak_count) / np.mean(peak_count)


def _ecg_quality_pSQI(
    ecg_cleaned,
    sampling_rate=512,
    window=1024,
    num_spectrum=[5, 15],
    dem_spectrum=[5, 40],
    **kwargs,
):
    # Power Spectrum Distribution of QRS Wave.
    psd = nk.signal.signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        normalize=False,
        window=window,
        **kwargs,
    )

    num_power = psd.iloc[0][0]
    dem_power = psd.iloc[0][1]

    return num_power / dem_power


def _ecg_quality_cSQI(rpeaks):
    # Variability in the R-R Interval defined by the standard deviation devided by the mean of the R-R interval.
    if len(rpeaks) < 2:
        return KeyError("NeuroKit error: ecg_quality(): Not enough R-peaks detected.")
    else:
        return np.std(np.diff(rpeaks)) / np.mean(np.diff(rpeaks))


def _ecg_quality_kSQI(ecg_cleaned):
    # Return the kurtosis of the signal
    return scipy.stats.kurtosis(ecg_cleaned, fisher=True)


def _ecq_quality_sSQI(ecg_cleaned):
    # Skewness of the QRS Wave.
    return scipy.stats.skew(ecg_cleaned)


def _ecg_quality_basSQI(
    ecg_cleaned,
    sampling_rate=512,
    window=1024,
    num_spectrum=[0, 1],
    dem_spectrum=[0, 40],
    **kwargs,
):
    """Relative Power in the Baseline."""
    psd = nk.signal.signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs,
    )

    num_power = psd.iloc[0][0]
    dem_power = psd.iloc[0][1]

    return (1 - num_power) / dem_power


# Classificaitons
def _classifiy_simple(sampling_rate, rpeaks, qSQI, pSQI, cSQI, kSQI, basSQI):
    # Classify indices based on simple heuristic fusion
    # First stage rules (0 = unqualified, 1 = suspicious, 2 = optimal
    optimal = 32  # red
    suspicious = 33  # yellow
    unqualified = 31  # green

    # Get the maximum bpm
    if len(rpeaks) > 1:
        heart_rate = 60000.0 / (1000.0 / sampling_rate * np.min(np.diff(rpeaks)))
    else:
        heart_rate = 1

    # qSQI classification
    if qSQI < 0.45:
        qSQI_class = optimal
    elif 0.45 <= qSQI and qSQI <= 0.64:
        qSQI_class = suspicious
    else:
        qSQI_class = unqualified

    # pSQI classification
    if heart_rate < 130:
        l1, l2, l3 = 0.5, 0.8, 0.4
    else:
        l1, l2, l3 = 0.4, 0.7, 0.3

    if l1 < pSQI and pSQI < l2:
        pSQI_class = optimal
    elif l3 < pSQI and pSQI < l1:
        pSQI_class = suspicious
    else:
        pSQI_class = unqualified

    # cSQI classification
    if cSQI < 0.45:
        cSQI_class = optimal
    elif 0.45 <= cSQI and cSQI <= 0.64:
        cSQI_class = suspicious
    else:
        cSQI_class = unqualified

    # kSQI classification
    if kSQI > 5:
        kSQI_class = optimal
    else:
        kSQI_class = unqualified

    # basSQI classification
    if 0.95 <= basSQI and basSQI <= 1:
        basSQI_class = optimal
    elif basSQI < 0.9:
        basSQI_class = unqualified
    else:
        basSQI_class = suspicious

    class_matrix = np.array(
        [qSQI_class, pSQI_class, cSQI_class, kSQI_class, basSQI_class]
    )

    n_optimal = len(np.where(class_matrix == optimal)[0])
    n_suspicious = len(np.where(class_matrix == suspicious)[0])
    n_unqualified = len(np.where(class_matrix == unqualified)[0])

    if (
        n_unqualified >= 3
        or (n_unqualified == 2 and n_suspicious >= 1)
        or (n_unqualified == 1 and n_suspicious == 3)
    ):
        conclusion = "Unacceptable"
    elif n_optimal >= 3 and n_unqualified == 0:
        conclusion = "Excellent"
    else:
        conclusion = "Barely acceptable"

    return (
        n_optimal,
        n_suspicious,
        n_unqualified,
        conclusion,
        qSQI_class,
        pSQI_class,
        cSQI_class,
        kSQI_class,
        basSQI_class,
    )


# Fuzzy Clasisfication function
def _classify_fuzzy(pSQI, kSQI, basSQI):
    # *R1 left out because of lack of qSQI

    # pSQI
    # UpH
    if pSQI <= 0.25:
        UpH = 0
    elif pSQI >= 0.35:
        UpH = 1
    else:
        UpH = 0.1 * (pSQI - 0.25)

    # UpI
    if pSQI < 0.18:
        UpI = 0
    elif pSQI >= 0.32:
        UpI = 0
    elif pSQI >= 0.18 and pSQI < 0.22:
        UpI = 25 * (pSQI - 0.18)
    elif pSQI >= 0.22 and pSQI < 0.28:
        UpI = 1
    else:
        UpI = 25 * (0.32 - pSQI)

    # UpJ
    if pSQI < 0.15:
        UpJ = 1
    elif pSQI > 0.25:
        UpJ = 0
    else:
        UpJ = 0.1 * (0.25 - pSQI)

    # Get R2
    R2 = np.array([UpH, UpI, UpJ])

    # kSQI
    # Get R3
    if kSQI > 5:
        R3 = np.array([1, 0, 0])
    else:
        R3 = np.array([0, 0, 1])

    # basSQI
    # UbH
    if basSQI <= 90:
        UbH = 0
    elif basSQI >= 95:
        UbH = basSQI / 100.0
    else:
        UbH = 1.0 / (1 + (1 / np.power(0.8718 * (basSQI - 90), 2)))

    # UbJ
    if basSQI <= 85:
        UbJ = 1
    else:
        UbJ = 1.0 / (1 + np.power((basSQI - 85) / 5.0, 2))

    # UbI
    UbI = 1.0 / (1 + np.power((basSQI - 95) / 2.5, 2))

    # Get R4
    R4 = np.array([UbH, UbI, UbJ])

    # evaluation matrix R (remove R1 because of lack of qSQI)
    # R = np.vstack([R1, R2, R3, R4])
    R = np.vstack([R2, R3, R4])

    # weight vector W (remove first weight because of lack of qSQI)
    # W = np.array([0.4, 0.4, 0.1, 0.1])
    W = np.array([0.6, 0.2, 0.2])

    S = np.array([np.sum((R[:, 0] * W)), np.sum((R[:, 1] * W)), np.sum((R[:, 2] * W))])

    # classify
    V = np.sum(np.power(S, 2) * [1, 2, 3]) / np.sum(np.power(S, 2))

    if V < 1.5:
        conclusion = "Excellent"
    elif V >= 2.40:
        conclusion = "Unnacceptable"
    else:
        conclusion = "Barely acceptable"

    return conclusion

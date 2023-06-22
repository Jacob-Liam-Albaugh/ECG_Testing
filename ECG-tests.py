# Get an example signal
import scipy  # new in scipy 1.10.0, used to be in scipy.misc
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zhoa_zhang_ecg_sqi import ecg_quality

ecg_ctr_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/01/ECG_3M_Control.csv"
)
ECG_1x1_WI_ts_g_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/01/ECG_1x1_WI_ts_g.csv"
)
ECG_3x1_WI_ts_g_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/01/ECG_3x1_WI_ts_g.csv"
)

ECG_1x1_CT_ts_ng_nf_ada_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/03/ECG_1x1_CT_ts_ng_nf_ada.csv"
)
ECG_1x1_CT_ts_ng_nf_lab_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/03/ECG_1x1_CT_ts_ng_nf_lab.csv"
)
ECG_3x1_CT_cn_g_wf_ada_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/03/ECG_3x1_CT_cn_g_wf_ada.csv"
)
ECG_3x1_CT_fs_g_wf_ada_data = pd.read_csv(
    "/Users/liam/Library/Mobile Documents/com~apple~CloudDocs/Purdue/Classes/UnderGrad Research/NeoWarm/ECG Testing/ECG_data/03/ECG_3x1_CT_fs_g_wf_ada.csv"
)

ecg_ctr = ecg_ctr_data[" ECG Filtered (mV)"].tolist()
ECG_1x1_WI = ECG_1x1_WI_ts_g_data[" ECG Filtered (mV)"].tolist()
ECG_3x1_WI = ECG_3x1_WI_ts_g_data[" ECG Filtered (mV)"].tolist()
ECG_1x1_CT_ts_ng_nf_ada = ECG_1x1_CT_ts_ng_nf_ada_data[" ECG Filtered (mV)"].tolist()
ECG_1x1_CT_ts_ng_nf_lab = ECG_1x1_CT_ts_ng_nf_lab_data[" ECG Filtered (mV)"].tolist()
ECG_3x1_CT_cn = ECG_3x1_CT_cn_g_wf_ada_data[" ECG Filtered (mV)"].tolist()
ECG_3x1_CT_fs = ECG_3x1_CT_fs_g_wf_ada_data[" ECG Filtered (mV)"].tolist()


ecg_scipy = scipy.datasets.electrocardiogram()
ecg_ctr = ecg_ctr[512:10752]
ECG_1x1_WI = ECG_1x1_WI[512:10752]  # [256:10496]
ECG_3x1_WI = ECG_3x1_WI[3584:13834]
ECG_1x1_CT_ts_ng_nf_ada = ECG_1x1_CT_ts_ng_nf_ada[640:4480]
ECG_1x1_CT_ts_ng_nf_lab = ECG_1x1_CT_ts_ng_nf_lab[640:4480]
ECG_3x1_CT_cn = ECG_3x1_CT_cn[1280:7680]
ECG_3x1_CT_fs = ECG_3x1_CT_fs[1280:6400]
# nk.ecg_clean(ecg_ctr, sampling_rate=512)


# ECG Processing
ecg_ctr_pro, _ = nk.ecg_process(ecg_ctr, sampling_rate=512)
ECG_1x1_WI_pro, _ = nk.ecg_process(ECG_1x1_WI, sampling_rate=512)
ECG_3x1_WI_pro, _ = nk.ecg_process(ECG_3x1_WI, sampling_rate=512)
ECG_1x1_CT_ts_ng_nf_ada_pro, _ = nk.ecg_process(
    ECG_1x1_CT_ts_ng_nf_ada, sampling_rate=128
)
ECG_1x1_CT_ts_ng_nf_lab_pro, _ = nk.ecg_process(
    ECG_1x1_CT_ts_ng_nf_lab, sampling_rate=128
)
ECG_3x1_CT_cn_pro, _ = nk.ecg_process(ECG_3x1_CT_cn, sampling_rate=128)
ECG_3x1_CT_fs_pro, _ = nk.ecg_process(ECG_3x1_CT_fs, sampling_rate=128)

"""
# ECG Plott
nk.ecg_plot(ecg_ctr_pro, sampling_rate=512)
nk.ecg_plot(ECG_1x1_WI_pro, sampling_rate=512)
nk.ecg_plot(ECG_3x1_WI_pro, sampling_rate=512)
nk.ecg_plot(ECG_1x1_CT_ts_ng_nf_ada_pro, sampling_rate=128)
nk.ecg_plot(ECG_1x1_CT_ts_ng_nf_lab_pro, sampling_rate=128)
nk.ecg_plot(ECG_3x1_CT_cn_pro, sampling_rate=128)
nk.ecg_plot(ECG_3x1_CT_fs_pro, sampling_rate=128)
# plt.show()

## NeoKit ECG Quality assesments
# Aerage qualtiy from 0 to 1, 1 being the best on an array
ecg_ctr_q = nk.ecg_quality(ecg_ctr, rpeaks=None, sampling_rate=512, approach=None)

# Average of quality array
ecg_ctr_q_avg = np.average(ecg_ctr_q)

# Median of quality array
ecg_ctr_q_median = np.median(ecg_ctr_q)
"""

# zhao method returns Unacceptable, Barely Acceptable or Excellent
"""ecg_ctr_q_zhao = nk.ecg_quality(
    ecg_ctr,
    rpeaks=None,
    sampling_rate=512,
    method="zhao2018",
    approach=None,
)
"""
ecg_ctr_q_zhao = ecg_quality(
    ecg_ctr,
    sampling_rate=512,
)
print("\n\necg_ctr_q_zhao: ", ecg_ctr_q_zhao, "\n")
"""
ecg_scipy_q_zhao = ecg_quality(ecg_scipy, sampling_rate=512)
print("ecg_scipy_q_zhao: ", ecg_scipy_q_zhao, "\n")

ECG_1x1_WI_q_zhao = ecg_quality(
    ECG_1x1_WI,
    rpeaks=None,
    sampling_rate=512,
    method="zhao2018",
    approach=None,
)
print("ECG_1x1_WI_q_zhao: ", ECG_1x1_WI_q_zhao, "\n")

ECG_3x1_WI_q_zhao = ecg_quality(
    ECG_3x1_WI,
    rpeaks=None,
    sampling_rate=512,
    method="zhao2018",
    approach=None,
)
print("ECG_3x1_WI_q_zhao: ", ECG_3x1_WI_q_zhao, "\n")
 
ECG_1x1_CT_ts_ng_nf_ada_q_zhao = ecg_quality(
    ECG_1x1_CT_ts_ng_nf_ada,
    rpeaks=None,
    sampling_rate=128,
    method="zhao2018",
    approach=None,
)
print("ECG_1x1_CT_ts_ng_nf_ada_q_zhao: ", ECG_1x1_CT_ts_ng_nf_ada_q_zhao, "\n")

ECG_1x1_CT_ts_ng_nf_lab_q_zhao = ecg_quality(
    ECG_1x1_CT_ts_ng_nf_lab,
    rpeaks=None,
    sampling_rate=128,
    method="zhao2018",
    approach=None,
)
print("\nECG_1x1_CT_ts_ng_nf_lab_q_zhao: ", ECG_1x1_CT_ts_ng_nf_lab_q_zhao, "\n")
"""
ECG_3x1_CT_cn_q_zhao = ecg_quality(
    ECG_3x1_CT_cn,
    rpeaks=None,
    sampling_rate=128,
    method="zhao2018",
    approach=None,
)
print("ECG_3x1_CT_cn_q_zhao: ", ECG_3x1_CT_cn_q_zhao, "\n")

ECG_3x1_CT_fs_q_zhao = ecg_quality(
    ECG_3x1_CT_fs,
    rpeaks=None,
    sampling_rate=128,
    method="zhao2018",
    approach=None,
)
print("ECG_3x1_CT_fs_q_zhao: ", ECG_3x1_CT_fs_q_zhao, "\n\n")

print(
    "\nECG Quality\n",
)
print(
    "Control: ",
    #    "\n  Average: ",
    #    ecg_ctr_q_avg,
    #    "\n  Median:",
    #    ecg_ctr_q_median,
    #    "\n  Assesment:",
    ecg_ctr_q_zhao,
)
print(
    "1x1 wire:  ",
    #    "\n  Average: ",
    #    ecg_1x1_q_avg,
    #    "\n  Median:",
    #    ecg_1x1_q_median,
    #    "\n  Assesment:",
    ECG_1x1_WI_q_zhao,
)
print(
    "3x1 wire:  ",
    #    "\n  Average: ",
    #    ecg_3x1_q_avg,
    #    "\n  Median:",
    #    ecg_3x1_q_median,
    #    "\n  Assesment:",
    ECG_3x1_WI_q_zhao,
)
print(
    "1x1 conductive thread top sewn no gel no fabric adafruit string:  ",
    ECG_1x1_CT_ts_ng_nf_ada_q_zhao,
)
print(
    "1x1 conductive thread top sewn no gel no fabric lab string:  ",
    ECG_1x1_CT_ts_ng_nf_lab_q_zhao,
)
print(
    "3x1 conductive thread corner sewn gel with fabric adafruit string:  ",
    ECG_3x1_CT_cn_q_zhao,
)

print(
    "3x1 conductive thread full sewn gell with fabric ada string:  ",
    ECG_3x1_CT_fs_q_zhao,
)


print("\n\n")

# plt.show()
""" with np.printoptions(threshold=np.inf):
    print("\n\nFull Control quality array:\n", ecg_ctr_q)
    print("\n\nFull 1x1 quality array:\n", ecg_1x1_q)
    print("\n\nFull 3x1 quality array:\n", ecg_3x1_q)
"""

# Zhao Zhang Methodology (2018)

# Get an example signal
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
from zhoa_zhang_ecg_sqi import ecg_quality as zzh

# ECG Data CSV reading
ecg_ctr_data = pd.read_csv("./ecg_data/01/ecg_3M_Control.csv")
ecg_1x1_WI_ts_g_data = pd.read_csv("./ecg_data/01/ecg_1x1_WI_ts_g.csv")
ecg_3x1_WI_ts_g_data = pd.read_csv("./ecg_data/01/ecg_3x1_WI_ts_g.csv")
ecg_1x1_CT_cs_g_nf_data = pd.read_csv("./ecg_data/04/ecg_1x1_CT_cs_g_nf.csv")
ecg_1x1_CT_ts_g_nf_data = pd.read_csv("./ecg_data/04/ecg_1x1_CT_ts_g_nf.csv")
ecg_3x1_CT_cs_g_nf_data = pd.read_csv("./ecg_data/04/ecg_3x1_CT_cs_g_nf.csv")
ecg_3x1_CT_fs_g_nf_data = pd.read_csv("./ecg_data/04/ecg_3x1_CT_fs_g_nf.csv")

# ECG Data CSV to list
ecg_ctr = ecg_ctr_data[" ECG Filtered (mV)"].tolist()
ecg_1x1_WI = ecg_1x1_WI_ts_g_data[" ECG Filtered (mV)"].tolist()
ecg_3x1_WI = ecg_3x1_WI_ts_g_data[" ECG Filtered (mV)"].tolist()
ecg_1x1_CT_cs = ecg_1x1_CT_cs_g_nf_data[" ECG Filtered (mV)"].tolist()
ecg_1x1_CT_ts = ecg_1x1_CT_ts_g_nf_data[" ECG Filtered (mV)"].tolist()
ecg_3x1_CT_cs = ecg_3x1_CT_cs_g_nf_data[" ECG Filtered (mV)"].tolist()
ecg_3x1_CT_fs = ecg_3x1_CT_fs_g_nf_data[" ECG Filtered (mV)"].tolist()

# ECG Data truncation
range = 10752 * 2
ecg_ctr = ecg_ctr[512 : (512 + range)]
ecg_1x1_WI = ecg_1x1_WI[512 : (512 + range)]
ecg_3x1_WI = ecg_3x1_WI[3584 : (3584 + range)]
ecg_1x1_CT_cs = ecg_1x1_CT_cs[18944 : (18944 + range)]
ecg_1x1_CT_ts = ecg_1x1_CT_ts[36864 : (36864 + range)]
ecg_3x1_CT_cs = ecg_3x1_CT_cs[16384 : (16384 + range)]
ecg_3x1_CT_fs = ecg_3x1_CT_fs[2560 : (2560 + range)]

# ECG Data Cleaning
ecg_ctr_clean = nk.ecg_clean(ecg_ctr, sampling_rate=512)
ecg_1x1_WI_clean = nk.ecg_clean(ecg_1x1_WI, sampling_rate=512)
ecg_3x1_WI_clean = nk.ecg_clean(ecg_3x1_WI, sampling_rate=512)
ecg_1x1_CT_cs_clean = nk.ecg_clean(ecg_1x1_CT_cs, sampling_rate=512)
ecg_1x1_CT_ts_clean = nk.ecg_clean(ecg_1x1_CT_ts, sampling_rate=512)
ecg_3x1_CT_cs_clean = nk.ecg_clean(ecg_3x1_CT_cs, sampling_rate=512)
ecg_3x1_CT_fs_clean = nk.ecg_clean(ecg_3x1_CT_fs, sampling_rate=512)

# ECG Processing
ecg_ctr_pro, _ = nk.ecg_process(ecg_ctr_clean, sampling_rate=512)
ecg_1x1_WI_pro, _ = nk.ecg_process(ecg_1x1_WI_clean, sampling_rate=512)
ecg_3x1_WI_pro, _ = nk.ecg_process(ecg_3x1_WI_clean, sampling_rate=512)
ecg_1x1_CT_cs_pro, _ = nk.ecg_process(ecg_1x1_CT_cs_clean, sampling_rate=512)
ecg_1x1_CT_ts_pro, _ = nk.ecg_process(ecg_1x1_CT_ts_clean, sampling_rate=512)
ecg_3x1_CT_cs_pro, _ = nk.ecg_process(ecg_3x1_CT_cs_clean, sampling_rate=512)
ecg_3x1_CT_fs_pro, _ = nk.ecg_process(ecg_3x1_CT_fs_clean, sampling_rate=512)

# ECG Plott
nk.ecg_plot(ecg_ctr_pro, sampling_rate=512)
plt.title("ecg_ctr")
nk.ecg_plot(ecg_1x1_WI_pro, sampling_rate=512)
plt.title("ecg_1x1_WI")
nk.ecg_plot(ecg_3x1_WI_pro, sampling_rate=512)
plt.title("ecg_3x1_WI")
nk.ecg_plot(ecg_1x1_CT_cs_pro, sampling_rate=512)
plt.title("ecg_1x1_CT_cs")
nk.ecg_plot(ecg_1x1_CT_ts_pro, sampling_rate=512)
plt.title("ecg_1x1_CT_ts")
nk.ecg_plot(ecg_3x1_CT_cs_pro, sampling_rate=512)
plt.title("ecg_3x1_CT_cs")
nk.ecg_plot(ecg_3x1_CT_fs_pro, sampling_rate=512)
plt.title("ecg_3x1_CT_fs")

# ECG Quality Analysis
ecg_ctr_q_zhao = zzh(ecg_ctr, sampling_rate=512)
ecg_1x1_WI_q_zhao = zzh(ecg_1x1_WI, sampling_rate=512)
ecg_3x1_WI_q_zhao = zzh(ecg_3x1_WI, sampling_rate=512)
ecg_1x1_CT_cs_q_zhao = zzh(ecg_1x1_CT_cs, sampling_rate=512)
ecg_1x1_CT_ts_q_zhao = zzh(ecg_1x1_CT_ts, sampling_rate=512)
ecg_3x1_CT_cs_q_zhao = zzh(ecg_3x1_CT_cs, sampling_rate=512)
ecg_3x1_CT_fs_q_zhao = zzh(ecg_3x1_CT_fs, sampling_rate=512)

# ECG Quality print
print("\nECG Quality:")
print("\tecg_ctr_q_zhao: \n\t\t", ecg_ctr_q_zhao, "\n")
print("\tecg_1x1_WI_q_zhao: \n\t\t", ecg_1x1_WI_q_zhao, "\n")
print("\tecg_3x1_WI_q_zhao: \n\t\t", ecg_3x1_WI_q_zhao, "\n")
print("\tecg_1x1_CT_cs_q_zhao: \n\t\t", ecg_1x1_CT_cs_q_zhao, "\n")
print("\tecg_1x1_CT_ts_q_zhao: \n\t\t", ecg_1x1_CT_ts_q_zhao, "\n")
print("\tecg_3x1_CT_cs_q_zhao: \n\t\t", ecg_3x1_CT_cs_q_zhao, "\n")
print("\tecg_3x1_CT_fs_q_zhao: \n\t\t", ecg_3x1_CT_fs_q_zhao, "\n")

# plt.show()

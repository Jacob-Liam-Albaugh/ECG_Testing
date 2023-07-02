import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
from zhoa_zhang_ecg_sqi import ecg_quality as zzh

# ECG Data names
ECG_NAMES = [
    "ecg_ctr",
    "ecg_1x1_WI",
    "ecg_3x1_WI",
    "ecg_1x1_CT_cs",
    "ecg_1x1_CT_ts",
    "ecg_3x1_CT_cs",
    "ecg_3x1_CT_fs",
]

# ECG Data CSV reading to list
ecg_ctr = pd.read_csv("./ecg_data/ecg_3M_Control.csv")[" ECG Filtered (mV)"].tolist()
ecg_1x1_WI = pd.read_csv("./ecg_data/ecg_1x1_WI_ts_g.csv")[
    " ECG Filtered (mV)"
].tolist()
ecg_3x1_WI = pd.read_csv("./ecg_data/ecg_3x1_WI_ts_g.csv")[
    " ECG Filtered (mV)"
].tolist()
ecg_1x1_CT_cs = pd.read_csv("./ecg_data/ecg_1x1_CT_cs_g_nf.csv")[
    " ECG Filtered (mV)"
].tolist()
ecg_1x1_CT_ts = pd.read_csv("./ecg_data/ecg_1x1_CT_ts_g_nf.csv")[
    " ECG Filtered (mV)"
].tolist()
ecg_3x1_CT_cs = pd.read_csv("./ecg_data/ecg_3x1_CT_cs_g_nf.csv")[
    " ECG Filtered (mV)"
].tolist()
ecg_3x1_CT_fs = pd.read_csv("./ecg_data/ecg_3x1_CT_fs_g_nf.csv")[
    " ECG Filtered (mV)"
].tolist()

# ECG Data truncation
range = 10752 * 2
ECG_DATA = [
    ecg_ctr[512 : (512 + range)],
    ecg_1x1_WI[512 : (512 + range)],
    ecg_3x1_WI[3584 : (3584 + range)],
    ecg_1x1_CT_cs[18944 : (18944 + range)],
    ecg_1x1_CT_ts[36864 : (36864 + range)],
    ecg_3x1_CT_cs[21024 : (21024 + range)],
    ecg_3x1_CT_fs[2560 : (2560 + range)],
]

# ECG Data Cleaning
ECG_clean = []
for name in ECG_NAMES:
    ecg_clean = nk.ecg_clean(ECG_DATA[ECG_NAMES.index(name)], sampling_rate=512)
    ECG_clean.append(ecg_clean)

# ECG Processing
ECG_pro = []
for clean in ECG_clean:
    pro, _ = nk.ecg_process(clean, sampling_rate=512)
    ECG_pro.append(pro)

# ECG Plot
idx = 0
for pro in ECG_pro:
    nk.ecg_plot(pro, sampling_rate=512)
    plt.title(ECG_NAMES[idx])
    idx += 1

# ECG Quality Analysis
ECG_Zhao = []
for data in ECG_DATA:
    ECG_Zhao.append(zzh(data, sampling_rate=512))

# ECG Quality print
print("\nECG Quality:", end="")
for data in ECG_DATA:
    (
        n_optimal,
        n_suspicious,
        n_unqualified,
        conclusion,
        SQIs,
    ) = zzh(data, sampling_rate=512)

    print(
        "\n   \033[0;37m %s: (O: %d, S: %d, U: %d) => %s\n"
        % (
            ECG_NAMES[ECG_DATA.index(data)],
            n_optimal,
            n_suspicious,
            n_unqualified,
            conclusion,
        ),
        end="",
    )
    print("\t", end="")

    for sqi in SQIs:
        print("%s " % (sqi), end="")

plt.show()

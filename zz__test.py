import neurokit2 as nk
import pandas as pd
from zhoa_zhang_ecg_sqi import ecg_quality as zzh

ecg_ctr_data = pd.read_csv("./ecg_data/01/ecg_3M_Control.csv")
ecg_ctr = ecg_ctr_data[" ECG Filtered (mV)"].tolist()
range = 10752 * 2
ecg_ctr = ecg_ctr[512 : (512 + range)]


(
    n_optimal,
    n_suspicious,
    n_unqualified,
    conclusion,
    SQIs,
) = zzh(ecg_ctr, sampling_rate=512)

print("\nECG Quality:")
print(
    "\tecg_ctr_q_zhao: \n\t\t",
    n_optimal,
    n_suspicious,
    n_unqualified,
    conclusion,
)


print(" %s, %s, %s, %s, %s, %s, %s", SQIs)

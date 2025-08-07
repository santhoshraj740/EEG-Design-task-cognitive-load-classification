import scipy.io
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import re
from scipy.signal import welch
from scipy.stats import entropy
import antropy as ant

participant_num = 2
data_path = "C:/Users/ssr17/Skybrain_neurotech/" 
filename_mat = f"Data_Design_Sub_{participant_num}.mat"
file_path_mat = os.path.join(data_path, filename_mat)
sfreq = 250
save_dir_preprocessed = 'preprocessed_data'
output_csv_features = 'features_extracted.csv'


BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 45)
}

SAMPLING_RATE = 250


def preprocess_all_sessions(mat_data, sfreq, save_dir,
                            filter_l_freq=1, filter_h_freq=40,
                            epoch_length=2.0, epoch_overlap=1.0):

    os.makedirs(save_dir, exist_ok=True)

    for key in mat_data.keys():
        if key.startswith('__'):
            continue

        print(f"Processing {key}")
        eeg_data = mat_data[key]

        # Create MNE Raw object
        ch_names = [f'EEG{i+1}' for i in range(eeg_data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info, verbose=False) 

        raw.filter(filter_l_freq, filter_h_freq, fir_design='firwin', verbose=False)

        data = raw.get_data()
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        raw._data = data

        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length,
                                              overlap=epoch_overlap, preload=True, verbose=False)

        filename = os.path.join(save_dir, f"{key}_epochs.npy")
        np.save(filename, epochs.get_data())
        print(f"Saved {filename}\n")

    print("All sessions processed and saved.")

def bandpower(epoch, fs, band):
    fmin, fmax = band
    freqs, psd = welch(epoch, fs=fs, nperseg=epoch.shape[-1])
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.sum(psd[idx_band])

def compute_hjorth(epoch):
    """Compute Hjorth activity, mobility, and complexity for a 1D array."""
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)
    var_zero = np.var(epoch)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
    complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0
    return activity, mobility, complexity

def extract_features_from_epoch(epoch):
    features = {}
    for i, channel_data in enumerate(epoch):
        for band_name, band_range in BANDS.items():
            power = bandpower(channel_data, SAMPLING_RATE, band_range)
            features[f'ch_{i+1}_{band_name}_power'] = power
        act, mob, comp = compute_hjorth(channel_data)
        features[f'ch_{i+1}_hjorth_activity'] = act
        features[f'ch_{i+1}_hjorth_mobility'] = mob
        features[f'ch_{i+1}_hjorth_complexity'] = comp
        try:
            features[f'ch_{i+1}_spectral_entropy'] = ant.spectral_entropy(channel_data, sf=SAMPLING_RATE, method='welch', normalize=True)
        except Exception as e:
            features[f'ch_{i+1}_spectral_entropy'] = np.nan
    return features

if __name__ == '__main__':
    try:
        mat = scipy.io.loadmat(file_path_mat)
        print(f"Successfully loaded {file_path_mat}")
    except FileNotFoundError:
        print(f"Error: {file_path_mat} not found. Please check the data_path and filename_mat.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the .mat file: {e}")
        exit()

    preprocess_all_sessions(mat, sfreq, save_dir_preprocessed)

    all_features = []
    for filename in os.listdir(save_dir_preprocessed):
        if filename.endswith('.npy'):
            file_path = os.path.join(save_dir_preprocessed, filename)
            data = np.load(file_path)  

            print(f"Processing {filename} - shape: {data.shape}")
            label_parts = filename.replace('_epochs.npy', '').split('_')
            if 'RST' in filename:
                session = "_".join(label_parts[:2]) 
                label = label_parts[3] if len(label_parts) > 3 else 'unknown' 
            else:
                session = "_".join(label_parts[:3])  # e.g., Design_2_1
                label = label_parts[3] if len(label_parts) > 3 else 'unknown'


            for epoch_idx, epoch in enumerate(data):
                features = extract_features_from_epoch(epoch)
                features['epoch'] = epoch_idx
                features['session'] = session
                features['label'] = label
                all_features.append(features)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv_features, index=False)
    print(f"\n✅ Feature extraction complete. Saved to {output_csv_features}")

    df = pd.read_csv(output_csv_features)
    df.loc[df['session'].str.contains('RST1', na=False), 'label'] = 'RST1'
    df.loc[df['session'].str.contains('RST2', na=False), 'label'] = 'RST2'
    df.to_csv(output_csv_features, index=False)
    print(f"✅ Labels corrected and saved to {output_csv_features}")


    df = pd.read_csv(output_csv_features)

    bands_for_plot = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    band_power_data = {band: [] for band in bands_for_plot}
    labels_for_plot = []

    for i, row in df.iterrows():
        for band in bands_for_plot:
            band_cols = [col for col in df.columns if re.match(fr'ch_\d+_{band}_power$', col)]
            if band_cols:
                band_power_data[band].append(row[band_cols].mean())
            else:
                band_power_data[band].append(np.nan) 
        labels_for_plot.append(row['label'])

    avg_df = pd.DataFrame(band_power_data)
    avg_df['Label'] = labels_for_plot

    df_melted = avg_df.melt(id_vars="Label", var_name="Band", value_name="Power")

    plt.figure(figsize=(12, 7)) 
    sns.barplot(data=df_melted, x="Label", y="Power", hue="Band", ci="sd", capsize=0.1, palette="viridis")

    plt.title("Average Band Power by Condition Label", fontsize=16)
    plt.xlabel("Condition Label", fontsize=12)
    plt.ylabel("Mean Band Power (across all channels)", fontsize=12)
    plt.legend(title="Frequency Band", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 
    plt.show()
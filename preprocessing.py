import mne
import numpy as np
import os

def preprocess_and_save_all(mat, sfreq=250, save_dir='preprocessed_data',
                             filter_l_freq=1, filter_h_freq=40,
                             ica_n_components=20, ica_random_state=42,
                             epoch_length=2.0, epoch_overlap=1.0):

    standard_1010_64_ch_names_full = [
        'Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz', 'F1', 'F2',
        'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz', 'FT7', 'FT8', 'FC1',
        'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'T7', 'T8', 'C1', 'C2',
        'C3', 'C4', 'C5', 'C6', 'Cz', 'TP7', 'TP8', 'CP1', 'CP2', 'CP3',
        'CP4', 'CP5', 'CP6', 'CPz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
        'P7', 'P8', 'P9', 'P10', 'Pz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz',
        'O1', 'O2', 'Oz', 'Iz'
    ]
    channels_to_exclude = ['Cz']
    expected_63_ch_names = [ch for ch in standard_1010_64_ch_names_full if ch not in channels_to_exclude]

    os.makedirs(save_dir, exist_ok=True)

     for key in mat:
        print(f"Preprocessing {key}")
        eeg_data = mat[key]

        if eeg_data.shape[0] != len(expected_63_ch_names):
            print(f"Skipping {key} due to unexpected channel count: {eeg_data.shape[0]}")
            continue

        raw_ch_names = [f'EEG {i+1}' for i in range(eeg_data.shape[0])]
        info = mne.create_info(ch_names=raw_ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        rename_mapping = dict(zip(raw_ch_names, expected_63_ch_names))
        raw.rename_channels(rename_mapping)

        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)

        raw.filter(filter_l_freq, filter_h_freq, verbose=False)

        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=ica_random_state)
        ica.fit(raw)

        try:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2', 'Fpz', 'AFz', 'Fz'])
            ica.exclude = eog_inds
        except Exception as e:
            print(f"EOG detection failed for {key}: {e}")

        raw_clean = ica.apply(raw.copy())

        cleaned_data = raw_clean.get_data()
        cleaned_data = (cleaned_data - np.mean(cleaned_data, axis=1, keepdims=True)) / \
                       np.std(cleaned_data, axis=1, keepdims=True)
        raw_clean._data = cleaned_data

        epochs = mne.make_fixed_length_epochs(raw_clean, duration=epoch_length,
                                              overlap=epoch_overlap, preload=True)

        npy_path = os.path.join(save_dir, f'{key}_epochs.npy')
        np.save(npy_path, epochs.get_data())
        print(f"Saved {key} to {npy_path}")

#running the function
preprocess_and_save_all(mat, sfreq=250) 

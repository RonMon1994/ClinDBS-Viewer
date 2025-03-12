




import mne
import numpy as np

def create_bipolar_eeg(raw, z_thresh=3.5, min_bipolar_std=1e-6):
    """
    Given an MNE Raw object, perform the following steps:
    1. Rename channels to a known montage (mapping E1..E64 -> standard names).
    2. Detect & mark bad channels (simple amplitude-based).
    3. Interpolate bad channels.
    4. Restrict to a specified 21-channel 10-20 subset (channels_21).
    5. Build a bipolar montage from spatial_pairs, skipping invalid channels.
    6. Check for bipolar channels with near-zero signals, remove them (and pairs).
    7. Return a Raw object (raw_bipolar_only) containing only the final bipolar channels.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data, already loaded (e.g., read_raw_egi).
    mapping : dict
        Dictionary mapping EGI channel names (E1..E64) to standard names.
    channels_21 : list of str
        List of 21 desired 10-20 channels.
    spatial_pairs : list of tuple
        List of (anode, cathode) pairs for the bipolar montage.
    z_thresh : float, optional
        Threshold for auto-detecting bad channels (default 3.5).
    min_bipolar_std : float, optional
        Minimum standard deviation for a bipolar channel to be considered valid.
    
    Returns
    -------
    raw_bipolar_only : mne.io.Raw
        A new Raw object containing only the final, valid bipolar channels.
    """




    mapping = {
    'E1': 'F10', 'E2': 'AF4', 'E3': 'F2', 'E4': 'FCz', 'E5': 'Fp2',
    'E6': 'Fz', 'E7': 'FC1', 'E8': 'AFz', 'E9': 'F1', 'E10': 'Fp1',
    'E11': 'AF3', 'E12': 'F3', 'E13': 'F5', 'E14': 'FC5', 'E15': 'FC3',
    'E16': 'C1', 'E17': 'F9', 'E18': 'F7', 'E19': 'FT7', 'E20': 'C3',
    'E21': 'CP1', 'E22': 'C5', 'E23': 'T9', 'E24': 'T7', 'E25': 'TP7',
    'E26': 'CP5', 'E27': 'P5', 'E28': 'P3', 'E29': 'TP9', 'E30': 'P7',
    'E31': 'P1', 'E32': 'P9', 'E33': 'PO3', 'E34': 'Pz', 'E35': 'O1',
    'E36': 'POz', 'E37': 'Oz', 'E38': 'PO4', 'E39': 'O2', 'E40': 'P2',
    'E41': 'CP2', 'E42': 'P4', 'E43': 'P10', 'E44': 'P8', 'E45': 'P6',
    'E46': 'CP6', 'E47': 'TP10', 'E48': 'TP8', 'E49': 'C6', 'E50': 'C4',
    'E51': 'C2', 'E52': 'T8', 'E53': 'FC4', 'E54': 'FC2', 'E55': 'T10',
    'E56': 'FT8', 'E57': 'FC6', 'E58': 'F8', 'E59': 'F6', 'E60': 'F4',
    'E61': 'E61', 'E62': 'E62', 'E63': 'E63', 'E64': 'E64',
    'VREF': 'Cz'
}
    
    channels_21 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'T7', 'T8'
]


    spatial_pairs = [
    # Left Temporal
    ('Fp1', 'F7'),  ('F7', 'T7'),   ('T7', 'P7'),  ('P7', 'O1'),
    ('Fp1', 'F3'),  ('F3', 'C3'),   ('C3', 'P3'),   ('P3', 'O1'),
    # Central Chain
    ('Fz',  'Cz'),  ('Cz', 'Pz'),
    # Right Temporal
    ('Fp2', 'F4'),  ('F4', 'C4'),   ('C4', 'P4'),   ('P4', 'O2'),
    ('Fp2', 'F8'),  ('F8', 'T8'),  ('T8', 'P8'), ('P8', 'O2')
]

    # --- STEP A: Rename channels ---
    raw.rename_channels(mapping)

    # --- STEP B: Auto-detect bad channels by amplitude threshold ---
    bads = auto_detect_bad_channels(raw, z_thresh=z_thresh)
    raw.info['bads'] = bads
    raw.interpolate_bads(reset_bads=True)

    # --- STEP C: Pick only the 21 channels in channels_21 ---
    existing_21 = [ch for ch in channels_21 if ch in raw.ch_names]
    missing_21 = [ch for ch in channels_21 if ch not in raw.ch_names]
    if missing_21:
        raise ValueError(f"Missing 10-20 channels even after interpolation: {missing_21}")
    raw.pick_channels(channels_21)

    # --- STEP D: Create Bipolar Montage ---
    # Filter out any pairs where anode/cathode is missing
    valid_pairs = []
    for (anode, cathode) in spatial_pairs:
        if anode in raw.ch_names and cathode in raw.ch_names:
            valid_pairs.append((anode, cathode))

    raw_bipolar = mne.set_bipolar_reference(
        raw,
        anode=[p[0] for p in valid_pairs],
        cathode=[p[1] for p in valid_pairs],
        drop_refs=True
    )
    new_bipolar_chs = [f"{p[0]}-{p[1]}" for p in valid_pairs]

    # --- STEP E: Check bipolar quality ---
    bad_bip_chs = check_bipolar_quality(raw_bipolar, new_bipolar_chs, min_bipolar_std)
    # Gather electrodes that appear in 'bad' channels
    omitted_electrodes = set()
    for ch_name in bad_bip_chs:
        e1, e2 = ch_name.split('-')
        omitted_electrodes.add(e1)
        omitted_electrodes.add(e2)

    # Rebuild final set of pairs
    final_pairs = []
    for pair in valid_pairs:
        if pair[0] not in omitted_electrodes and pair[1] not in omitted_electrodes:
            final_pairs.append(pair)

    # If final pairs differ, re-create a montage with them
    if len(final_pairs) < len(valid_pairs):
        raw_bipolar = mne.set_bipolar_reference(
            raw,
            anode=[p[0] for p in final_pairs],
            cathode=[p[1] for p in final_pairs],
            drop_refs=True
        )
        new_bipolar_chs = [f"{p[0]}-{p[1]}" for p in final_pairs]

    # --- STEP F: Return only the final channels ---
    raw_bipolar_only = raw_bipolar.copy()
    raw_bipolar_only.pick_channels(new_bipolar_chs)
    #bipolar_data = raw_bipolar_only.get_data()

    return raw_bipolar_only, final_pairs

def auto_detect_bad_channels(raw_data, z_thresh=3.5):
    """
    Detect channels whose signal amplitude (standard deviation) is above
    (mean + z_thresh * std). Returns a list of bad channel names.
    """
    data, _ = raw_data[:]
    ch_stds = data.std(axis=1)
    mean_std = np.mean(ch_stds)
    std_std  = np.std(ch_stds)
    cutoff   = mean_std + z_thresh * std_std

    bads = []
    for idx, ch_name in enumerate(raw_data.ch_names):
        if ch_stds[idx] > cutoff:
            bads.append(ch_name)
    return bads

def check_bipolar_quality(raw_bip, ch_names, min_std=1e-6):
    """
    Check that each channel in ch_names has a standard deviation
    above 'min_std'. Returns a list of channels that fail this check.
    """
    data = raw_bip.get_data(picks=ch_names)
    bad_new = []
    for i, ch in enumerate(ch_names):
        std_ = np.std(data[i, :])
        if std_ < min_std:
            bad_new.append(ch)
    return bad_new






















import mne

raw = mne.io.read_raw_fif("eeg_data/test_subject/test_subject_udp_stream_session_1759648284_1759648284_raw.fif")
print(raw)

import sys
import os
import mne
import csv
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QComboBox, QHBoxLayout, QCheckBox, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from create_bipolar_utils import create_bipolar_eeg 
from scipy.signal import butter, filtfilt
import matplotlib.ticker as ticker



class EEGDBSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG & DBS Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # -------------------- EEG Section --------------------
        self.eeg_controls = QHBoxLayout()
        self.figure_eeg, self.ax_eeg = plt.subplots(figsize=(12, 4))
        self.canvas_eeg = FigureCanvas(self.figure_eeg)
        self.toolbar_eeg = NavigationToolbar(self.canvas_eeg, self)
        
        self.load_eeg_button = QPushButton("Load MFF")
        self.eeg_combo_box = QComboBox()
        self.start_sample_label_eeg = QLabel("Start Sample:")
        self.start_sample_input_eeg = QLineEdit()
        self.start_sample_checkbox_eeg = QCheckBox()
        self.lowcut_eeg_label = QLabel("Lowcut (Hz):")
        self.lowcut_eeg = QLineEdit("1")
        self.highcut_eeg_label = QLabel("Highcut (Hz):")
        self.highcut_eeg = QLineEdit("50")
        self.band_pass_eeg_checkbox = QCheckBox("Apply Band-Pass Filter")
        self.zoom_in_eeg = QPushButton("Zoom In")
        self.zoom_out_eeg = QPushButton("Zoom Out")

        # Store EEG combo box options as a class attribute
        self.eeg_combo_box.clear()
        self.eeg_combo_box.addItems(["All signals", "Left Temporal", "Central Chain", "Right Temporal"])


        # Connecting EEG Buttons to Functions
        self.load_eeg_button.clicked.connect(self.load_eeg_mff)
        self.eeg_combo_box.currentTextChanged.connect(self.update_eeg_plot)
        self.band_pass_eeg_checkbox.stateChanged.connect(self.apply_bandpass_eeg)
        self.zoom_in_eeg.clicked.connect(lambda: self.adjust_eeg_zoom(1.2))
        self.zoom_out_eeg.clicked.connect(lambda: self.adjust_eeg_zoom(0.8))
        self.start_sample_checkbox_eeg.stateChanged.connect(self.apply_start_sample_eeg)

        # Organizing EEG Controls
        self.eeg_controls.addWidget(self.load_eeg_button)
        self.eeg_controls.addWidget(self.eeg_combo_box)
        self.eeg_controls.addWidget(self.start_sample_label_eeg)
        self.eeg_controls.addWidget(self.start_sample_input_eeg)
        self.eeg_controls.addWidget(self.start_sample_checkbox_eeg)
        self.eeg_controls.addWidget(self.lowcut_eeg_label)
        self.eeg_controls.addWidget(self.lowcut_eeg)
        self.eeg_controls.addWidget(self.highcut_eeg_label)
        self.eeg_controls.addWidget(self.highcut_eeg)
        self.eeg_controls.addWidget(self.band_pass_eeg_checkbox)
        self.eeg_controls.addWidget(self.zoom_in_eeg)
        self.eeg_controls.addWidget(self.zoom_out_eeg)

        # Initializing EEG Variables
        self.raw_bipolar_only = None
        self.channel_data_eeg = {}
        self.channel_data_eeg_original = {}
        self.sample_rate_eeg = None
        self.mff_folder_path = None
        self.zoom_factor_eeg = 1.0
        self.spatial_pairs = None


        # -------------------- Shared Section --------------------
        self.shared_controls = QHBoxLayout()
        self.enable_marking_button = QCheckBox("Enable Marking")
        self.delete_marked_event_button = QPushButton("Delete Last Marked Event")
        self.nav_left = QPushButton("⬅ Left")
        self.nav_right = QPushButton("Right ➡")
        self.save_marked_events_button = QPushButton("Save Marked Events")
        self.load_marked_events_button = QPushButton("Load Marked Events")
        self.x_axis_length_label = QLabel("X-axis Length:")
        self.x_axis_length = QLineEdit()
        self.apply_x_axis_checkbox = QCheckBox("Apply X-axis Length")
        
        # Connecting Shared Buttons
        self.enable_marking_button.stateChanged.connect(self.toggle_marking_mode)
        self.delete_marked_event_button.clicked.connect(self.delete_last_marked_event)
        self.save_marked_events_button.clicked.connect(self.save_marked_events)
        self.load_marked_events_button.clicked.connect(self.load_marked_events)

        # Organizing Shared Controls
        self.shared_controls.addWidget(self.enable_marking_button)
        self.shared_controls.addWidget(self.delete_marked_event_button)
        self.shared_controls.addWidget(self.nav_left)
        self.shared_controls.addWidget(self.nav_right)
        self.shared_controls.addWidget(self.save_marked_events_button)
        self.shared_controls.addWidget(self.load_marked_events_button)
        self.shared_controls.addWidget(self.x_axis_length_label)
        self.shared_controls.addWidget(self.x_axis_length)
        self.shared_controls.addWidget(self.apply_x_axis_checkbox)

        # Ensure these variables are initialized in the constructor
        self.current_start_time_eeg = 0.0  # Start time for EEG display
        self.current_start_time_dbs = 0.0  # Start time for DBS display

        self.window_size_eeg = 20  # Display window size in seconds for EEG
        self.window_size_dbs = 20  # Display window size in seconds for DBS

        self.nav_left.clicked.connect(lambda: self.shift_both_views(-2))
        self.nav_right.clicked.connect(lambda: self.shift_both_views(2))

        self.apply_x_axis_checkbox.stateChanged.connect(self.update_plots_apply_x_axis_length)
        

        # -------------------- DBS Section --------------------
        self.dbs_controls = QHBoxLayout()
        self.figure_dbs, self.ax_dbs = plt.subplots(figsize=(12, 4))
        self.canvas_dbs = FigureCanvas(self.figure_dbs)
        self.toolbar_dbs = NavigationToolbar(self.canvas_dbs, self)

        self.load_dbs_button = QPushButton("Load JSON")
        self.dbs_combo_box = QComboBox()
        self.start_sample_label_dbs = QLabel("Start Sample:")
        self.start_sample_input_dbs = QLineEdit()
        self.start_sample_checkbox_dbs = QCheckBox()
        self.lowcut_dbs_label = QLabel("Lowcut (Hz):")
        self.lowcut_dbs = QLineEdit("1")
        self.highcut_dbs_label = QLabel("Highcut (Hz):")
        self.highcut_dbs = QLineEdit("50")
        self.band_pass_dbs_checkbox = QCheckBox("Apply Band-Pass Filter")
        self.zoom_in_dbs = QPushButton("Zoom In")
        self.zoom_out_dbs = QPushButton("Zoom Out")

        # Store DBS combo box options as a class attribute
        self.dbs_combo_options = [
            "Original Channels",
            "Montage Channels",
            "Right Side (Original and Montage)",
            "Left Side (Original and Montage)",
            "Right Side (Original Only)",
            "Left Side (Original Only)",
            "Right Side (Montage Only)",
            "Left Side (Montage Only)"
        ]

        # Connecting DBS Buttons to Functions
        self.load_dbs_button.clicked.connect(self.load_dbs_json)
        self.dbs_combo_box.currentTextChanged.connect(self.update_dbs_plot)
        self.band_pass_dbs_checkbox.stateChanged.connect(self.apply_bandpass_dbs)
        self.zoom_in_dbs.clicked.connect(lambda: self.adjust_dbs_zoom(1.2))
        self.zoom_out_dbs.clicked.connect(lambda: self.adjust_dbs_zoom(0.8))
        self.start_sample_checkbox_dbs.stateChanged.connect(self.apply_start_sample_dbs)

        # Organizing DBS Controls
        self.dbs_controls.addWidget(self.load_dbs_button)
        self.dbs_controls.addWidget(self.dbs_combo_box)
        self.dbs_controls.addWidget(self.start_sample_label_dbs)
        self.dbs_controls.addWidget(self.start_sample_input_dbs)
        self.dbs_controls.addWidget(self.start_sample_checkbox_dbs)
        self.dbs_controls.addWidget(self.lowcut_dbs_label)
        self.dbs_controls.addWidget(self.lowcut_dbs)
        self.dbs_controls.addWidget(self.highcut_dbs_label)
        self.dbs_controls.addWidget(self.highcut_dbs)
        self.dbs_controls.addWidget(self.band_pass_dbs_checkbox)
        self.dbs_controls.addWidget(self.zoom_in_dbs)
        self.dbs_controls.addWidget(self.zoom_out_dbs)

        # Adding layouts to the main GUI
        self.layout.addLayout(self.eeg_controls)
        self.layout.addWidget(self.toolbar_eeg)
        self.layout.addWidget(self.canvas_eeg)
        self.layout.addLayout(self.shared_controls)
        self.layout.addWidget(self.toolbar_dbs)
        self.layout.addWidget(self.canvas_dbs)
        self.layout.addLayout(self.dbs_controls)

        # Initializing DBS Variables
        self.channel_data_dbs = {}
        self.channel_data_dbs_original = {} 
        self.montage_data_dbs = {}  
        self.montage_data_dbs_original = {}
        self.sample_rate_dbs = 250 
        self.json_file_path = None  
        self.zoom_factor_dbs = 1.0 
    
    # ----------------- EEG-Specific Functions -----------------
    def load_eeg_mff(self):
        """Loads EEG data from an MFF file and processes it."""
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select MFF Folder", "")
        if not folder_path:
            print("❌ No folder selected. Operation cancelled.")
            return  # User canceled selection

        try:
            print(f"📂 Loading EEG data from: {folder_path}")

            # ✅ Attempt to read the EEG file
            raw = mne.io.read_raw_egi(folder_path, preload=True)
            print(f"✔ EEG data loaded. Found {len(raw.ch_names)} channels.")
            print(f"🔍 EEG Sample rate: {raw.info['sfreq']} Hz")

            # ✅ Convert to bipolar montage
            self.raw_bipolar_only, self.spatial_pairs = create_bipolar_eeg(raw)
            self.sample_rate_eeg = int(self.raw_bipolar_only.info['sfreq'])
            print(f"✔ Bipolar montage created. New sample rate: {self.sample_rate_eeg} Hz")

            # ✅ Extract channel data
            self.channel_data_eeg = self.extract_channel_data_eeg()

            if not self.channel_data_eeg:
                print("⚠ Warning: No EEG data found!")
                return  # Avoid processing empty data
            
            # ✅ Print debug info: number of channels
            print(f"📊 Extracted {len(self.channel_data_eeg)} EEG channels.")

            # ✅ Check first 10 samples of the first channel
            first_channel = list(self.channel_data_eeg.keys())[0]
            print(f"📊 First channel: {first_channel} - First 10 samples: {self.channel_data_eeg[first_channel][:10]}")

            # ✅ Store a copy of the original data
            self.channel_data_eeg_original = copy.deepcopy(self.channel_data_eeg)
            print("📁 Original EEG data stored as backup.")

            # ✅ Reset relevant states
            self.current_start_time_eeg = 0.0
            self.marked_events_eeg = []
            self.mff_folder_path = folder_path

            # ✅ Update the EEG signal selection box
            #self.eeg_combo_box.clear()
            #self.eeg_combo_box.addItems(["All signals", "Left Temporal", "Central Chain", "Right Temporal"])

            # ✅ Update the GUI with new EEG data
            self.update_eeg_plot()

            # ✅ Update the window title with folder name
            parent_dir = os.path.basename(folder_path)
            grandparent_dir = os.path.basename(os.path.dirname(folder_path))
            short_path = f"{grandparent_dir}/{parent_dir}"
            self.setWindowTitle(f"EEG Viewer - {short_path}")

            print("🎉 ✔ EEG data loaded successfully!")

        except FileNotFoundError:
            print(f"❌ Error: MFF file not found in {folder_path}.")
        except ValueError as ve:
            print(f"❌ Error: Invalid EEG file format: {ve}")
        except Exception as e:
            print(f"❌ Unexpected error while loading EEG: {e}")

    def extract_channel_data_eeg(self):
        """Extracts EEG channel data after bipolar conversion and returns a dictionary."""

        if self.raw_bipolar_only is None:
            print("⚠ Warning: No EEG data available for extraction.")
            return {}

        try:
            bipolar_data = self.raw_bipolar_only.get_data()
            channel_names = self.raw_bipolar_only.ch_names

            if len(bipolar_data) != len(channel_names):
                print("❌ Error: Mismatch between EEG data and channel names.")
                return {}

            extracted_data = {channel_names[i]: bipolar_data[i].tolist() for i in range(len(channel_names))}
            return extracted_data

        except Exception as e:
            print(f"❌ Unexpected error extracting EEG channel data: {e}")
            return {}

    def get_signals_by_option_eeg(self, option):
        """Returns EEG signals based on the selected category."""

        if not self.channel_data_eeg:
            print("⚠ No EEG data available.")
            return [], []

        if option == "All signals":
            return list(self.channel_data_eeg.values()), list(self.channel_data_eeg.keys())

        if self.spatial_pairs is None:
            print("⚠ Warning: Spatial pairs not initialized.")
            return [], []

        # Define ranges for each category
        category_ranges = {
            "Left Temporal": self.spatial_pairs[:8],
            "Central Chain": self.spatial_pairs[8:10],
            "Right Temporal": self.spatial_pairs[10:]
        }

        if option in category_ranges:
            signals = [self.channel_data_eeg.get(f"{pair[0]}-{pair[1]}")
                    for pair in category_ranges[option] if f"{pair[0]}-{pair[1]}" in self.channel_data_eeg]

            labels = [f"{pair[0]}-{pair[1]}" for pair in category_ranges[option] if f"{pair[0]}-{pair[1]}" in self.channel_data_eeg]

            return signals, labels

        print(f"⚠ Warning: Unknown EEG category '{option}'.")
        return [], []

    def update_eeg_plot(self):
        """Updates the EEG signal display based on user selections, zoom, and marked events."""
        
        print(f"📊 Debug: Plotting EEG data - First 10 samples: {list(self.channel_data_eeg.values())[0][:10]}")
        selected_option = self.eeg_combo_box.currentText()
        if not selected_option or not self.channel_data_eeg:
            print("⚠ No EEG data loaded or no option selected.")
            return

        # ✅ Do NOT reapply the filter inside this function
        signals, labels = self.get_signals_by_option_eeg(selected_option)
        if not signals:
            print("⚠ No valid EEG signals found for selection.")
            return

        # ✅ Determine the X-axis window length
        try:
            window_size_eeg = float(self.x_axis_length.text()) if self.apply_x_axis_checkbox.isChecked() else 20
            if window_size_eeg <= 0:
                raise ValueError("Window size must be greater than 0")
        except ValueError:
            print("⚠ Invalid X-axis length. Using default 20s.")
            window_size_eeg = 20

        # ✅ Compute sample range
        start_sample = int(self.current_start_time_eeg * self.sample_rate_eeg)
        end_sample = start_sample + int(window_size_eeg * self.sample_rate_eeg)

        # ✅ Ensure sample range is within bounds
        max_length = len(next(iter(self.channel_data_eeg.values())))
        end_sample = min(end_sample, max_length)
        start_sample = max(0, end_sample - int(window_size_eeg * self.sample_rate_eeg))

        time_axis = np.arange(start_sample, end_sample) / self.sample_rate_eeg

        # ✅ Clear the figure and update plots
        self.figure_eeg.clear()
        num_signals = len(signals)

        for i, (signal, label) in enumerate(zip(signals, labels)):
            signal = signal[start_sample:end_sample]

            ax = self.figure_eeg.add_subplot(num_signals, 1, i + 1)
            ax.plot(time_axis, signal, label=label, color='b')
            ax.legend(loc="upper right")

            if i < num_signals - 1:
                ax.tick_params(axis='x', labelbottom=False)

            # ✅ Apply zoom factor
            min_val, max_val = np.min(signal), np.max(signal)
            range_val = max_val - min_val
            epsilon = 1e-6  # Small offset to avoid zero-range issues
            range_val = max(range_val, epsilon)  # Ensure range is at least epsilon
            ax.set_ylim(
                min_val - (range_val * (1 - self.zoom_factor_dbs) / 2),
                max_val + (range_val * (1 - self.zoom_factor_dbs) / 2)
            )

            ax.margins(0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # Disable scientific notation on y-axis
            ax.ticklabel_format(style='plain', axis='y')

            ax.margins(0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

            # ✅ Highlight marked events
            for start, end in self.marked_events_eeg:
                if start_sample / self.sample_rate_eeg <= start <= end_sample / self.sample_rate_eeg:
                    ax.axvspan(start, end, color='red', alpha=0.3)

        print(f"📏 Y-axis limits before updating: {ax.get_ylim()}")
        print(f"📊 Min EEG value in current window: {np.min(signal)}, Max EEG value: {np.max(signal)}")

        self.figure_eeg.tight_layout()
        self.canvas_eeg.draw()

    def apply_bandpass_eeg(self):
        """
        Applies or removes a band-pass filter to EEG signals when the checkbox is toggled.

        Fixes:
        - Prevents filtering from applying multiple times to the same signal.
        - Ensures trimming is preserved even when toggling filtering on/off.
        - Properly restores the correct signal state when disabling filtering.
        """

        print("🔍 Entering apply_bandpass_eeg()")

        if not self.channel_data_eeg_original:
            print("⚠ No EEG original data available for filtering.")
            return

        try:
            lowcut = float(self.lowcut_eeg.text())
            highcut = float(self.highcut_eeg.text())
            fs = self.sample_rate_eeg
        except ValueError:
            print("⚠ Invalid frequency values.")
            return

        if lowcut >= highcut or lowcut <= 0 or highcut >= fs / 2:
            print(f"⚠ Error: Invalid filter values (Lowcut: {lowcut}, Highcut: {highcut}, Fs: {fs})")
            return

        print(f"🟢 Before filtering: First 10 samples (original): {list(self.channel_data_eeg_original.values())[0][:10]}")
        print(f"🟢 Before filtering (current state): {list(self.channel_data_eeg.values())[0][:10]}")

        self.band_pass_eeg_checkbox.blockSignals(True)  # Prevent unintended recursive updates

        if self.band_pass_eeg_checkbox.isChecked():
            print(f"✔ Applying Band-Pass filter (Lowcut: {lowcut} Hz, Highcut: {highcut} Hz)")

            # ✅ Always filter the ORIGINAL untrimmed signal to avoid repeated filtering
            self.filtered_full_channel_data_eeg = {
                key: self._bandpass_filter(val, lowcut, highcut, fs) for key, val in self.channel_data_eeg_original.items()
            }

            # ✅ If the signal was trimmed before filtering, apply the filter to the trimmed version
            if hasattr(self, "trimmed_channel_data_eeg") and self.trimmed_channel_data_eeg:
                print("🟡 Applying filter to trimmed signal")
                self.filtered_trimmed_channel_data_eeg = {
                    key: self._bandpass_filter(val, lowcut, highcut, fs) for key, val in self.trimmed_channel_data_eeg.items()
                }
                self.channel_data_eeg = copy.deepcopy(self.filtered_trimmed_channel_data_eeg)
            else:
                print("🟢 Applying filter to full-length signal")
                self.channel_data_eeg = copy.deepcopy(self.filtered_full_channel_data_eeg)

        else:
            print("🔄 Resetting EEG signals to previous state (without filter)")

            # ✅ Restore the appropriate version based on whether trimming was applied
            if hasattr(self, "trimmed_channel_data_eeg") and self.trimmed_channel_data_eeg:
                print("🔄 Restoring trimmed EEG data without filtering")
                self.channel_data_eeg = copy.deepcopy(self.trimmed_channel_data_eeg)
            else:
                print("🔄 Restoring full-length original EEG data")
                self.channel_data_eeg = copy.deepcopy(self.channel_data_eeg_original)

        self.band_pass_eeg_checkbox.blockSignals(False)  # ✅ Re-enable checkbox interactions

        print(f"🟢 After filtering: First 10 samples (original): {list(self.channel_data_eeg_original.values())[0][:10]}")
        print(f"🟢 After filtering (current state): {list(self.channel_data_eeg.values())[0][:10]}")
        if self.band_pass_eeg_checkbox.isChecked():
            print("✔ Applying filter")
        else:
            print("🔄 Restoring previous state")


        # ✅ Refresh the plot
        self.update_eeg_plot()

    def adjust_eeg_zoom(self, factor):
        """Adjusts the zoom level of the EEG signal display."""
        
        # ✅ Ensure zoom factor stays within valid bounds
        self.zoom_factor_eeg *= factor
        self.zoom_factor_eeg = max(0.2, min(self.zoom_factor_eeg, 5.0))  # Limit between 0.2 and 5.0
        
        print(f"🔍 Adjusting EEG Zoom: {self.zoom_factor_eeg}")  # Debugging

        # ✅ Update the EEG plot with the new zoom level
        self.update_eeg_plot()

    def apply_start_sample_eeg(self):
        """
        Adjusts the EEG signal start point based on user input, allowing reset to original length.
        
        Fixes:
        - Ensures proper handling when both filtering and trimming are toggled.
        - Prevents unintended overwriting of data.
        - Resets trimmed data correctly when trimming is disabled.
        - Ensures filtering applies to the correct version (full-length or trimmed).
        """

        print("🔍 Entering apply_start_sample_eeg()")

        # ✅ Ensure EEG data is loaded before modifying it
        if not self.channel_data_eeg:
            print("⚠ No EEG data loaded.")
            return

        # ✅ CASE 1: Checkbox is unchecked → Restore previous state
        if not self.start_sample_checkbox_eeg.isChecked():
            print("🔄 Resetting to previous state...")

            # ✅ Reset trimmed data to avoid unintended persistence
            self.trimmed_channel_data_eeg = None

            # ✅ Restore the correct state based on whether filtering is active
            if self.band_pass_eeg_checkbox.isChecked() and hasattr(self, "filtered_full_channel_data_eeg"):
                print("🔄 Restoring **full filtered** EEG data")
                self.channel_data_eeg = copy.deepcopy(self.filtered_full_channel_data_eeg)  # Restore full filtered signal
            else:
                print("🔄 Restoring **original unfiltered** EEG data")
                self.channel_data_eeg = copy.deepcopy(self.channel_data_eeg_original)  # Restore full unfiltered signal

            # ✅ Update the EEG plot after resetting
            self.update_eeg_plot()
            return  # Exit early

        # ✅ CASE 2: Checkbox is checked → Trim signals based on user input
        try:
            start_sample = int(self.start_sample_input_eeg.text())  # Convert input to integer
            if start_sample < 0:
                print("⚠ Error: Start sample must be a non-negative integer.")
                return
        except ValueError:
            print("⚠ Error: Invalid start sample value (must be an integer).")
            return

        # ✅ Validate the start sample against signal length
        max_samples = len(next(iter(self.channel_data_eeg_original.values())))  # Get total samples from original data
        if start_sample >= max_samples:
            print(f"⚠ Error: Start sample ({start_sample}) exceeds total samples ({max_samples}). Resetting to previous state.")

            # ✅ Restore correct version depending on whether filtering is applied
            if self.band_pass_eeg_checkbox.isChecked():
                print("🔄 Restoring **full filtered** EEG data")
                self.channel_data_eeg = copy.deepcopy(self.filtered_full_channel_data_eeg)
            else:
                print("🔄 Restoring **original unfiltered** EEG data")
                self.channel_data_eeg = copy.deepcopy(self.channel_data_eeg_original)

        else:
            # ✅ Trim the **correct version** of the signal (filtered or unfiltered)
            if self.band_pass_eeg_checkbox.isChecked():
                print(f"✂ Trimming **filtered** EEG data from sample {start_sample}")
                self.channel_data_eeg = {key: val[start_sample:] for key, val in self.filtered_full_channel_data_eeg.items()}
            else:
                print(f"✂ Trimming **original unfiltered** EEG data from sample {start_sample}")
                self.channel_data_eeg = {key: val[start_sample:] for key, val in self.channel_data_eeg_original.items()}

        # ✅ Store the trimmed version separately for future use
        print("✅ Storing trimmed EEG data")
        self.trimmed_channel_data_eeg = copy.deepcopy(self.channel_data_eeg)

        # ✅ Update the EEG plot after trimming
        self.update_eeg_plot()

    # ----------------- DBS-Specific Functions -----------------
    def load_dbs_json(self):
        """Loads DBS data from a JSON file and updates the plot."""
        
        print("🔍 Starting DBS JSON load...")  # Debugging

        # Open file selection dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json)")
        
        if not file_path:
            print("⚠ No file selected. Operation cancelled.")
            return

        try:
            # Save file path for later use
            self.json_file_path = file_path

            # Extract the last two directories and file name for GUI title
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            short_path = f"{grandparent_dir}/{parent_dir}/{file_name}"
            self.setWindowTitle(f"DBS Viewer - {short_path}")

            # Reset necessary variables before loading new data
            self.current_start_time_dbs = 0.0
            self.marked_events_dbs = []  # Reset marked events
            
            # Read and parse JSON file
            with open(file_path, "r") as file:
                data = json.load(file)

            print(f"📄 JSON Keys: {list(data.keys())}")  # Debugging: Show JSON structure
            
            # Extracting sampling rate from JSON if available
            self.sample_rate_dbs = data.get("sampling_rate", 250)
            print(f"✔ Sampling rate: {self.sample_rate_dbs} Hz")

            # Extract and process channel data
            self.channel_data_dbs = self.extract_channel_data_dbs(data)
            
            if not self.channel_data_dbs:
                print("⚠ No valid DBS data found in the file.")
                return
            
            print("✅ Extracted DBS channel data.") # Debugging
            
            # Store a copy of the original data for reference
            self.channel_data_dbs_original = copy.deepcopy(self.channel_data_dbs)

            # ✅ Generate montage signals before UI updates
            self.montage_data_dbs = self.create_montages(self.channel_data_dbs)  # Now it gets assigned
            self.montage_data_dbs_original = copy.deepcopy(self.montage_data_dbs)
            print("✅ Created montage channels.")  # Debugging

            # Prevent unnecessary updates while modifying the combo box
            self.dbs_combo_box.blockSignals(True)

            # Update combo box options
            self.dbs_combo_box.clear()
            self.dbs_combo_box.addItems([
                "Original Channels",
                "Montage Channels",
                "Right Side (Original and Montage)",
                "Left Side (Original and Montage)",
                "Right Side (Original Only)",
                "Left Side (Original Only)",
                "Right Side (Montage Only)",
                "Left Side (Montage Only)"
            ])
            
            self.dbs_combo_box.blockSignals(False)
                
            # Update the DBS plot with the newly loaded data
            self.update_dbs_plot()
            print("✅ Updated DBS plot.")  # Debugging

            print(f"✔ DBS data loaded successfully from {file_path}")

        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON file format.")
        except FileNotFoundError:
            print(f"❌ Error: File not found - {file_path}")
        except Exception as e:
            print(f"❌ Unexpected error while loading DBS data: {e}")


        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON file format.")
        except FileNotFoundError:
            print(f"❌ Error: File not found - {file_path}")
        except Exception as e:
            print(f"❌ Unexpected error while loading DBS data: {e}")

    def extract_channel_data_dbs(self, data):
        """Extracts DBS channel data from the provided JSON structure."""
        
        # Ensure the key exists
        if "IndefiniteStreaming" not in data:
            print("⚠ Error: Missing 'IndefiniteStreaming' key in JSON data.")
            return {}

        channel_data_dbs = {}

        try:
            for item in data["IndefiniteStreaming"]:
                # Validate presence of required keys
                if "Channel" not in item or "TimeDomainData" not in item:
                    print("⚠ Warning: Missing 'Channel' or 'TimeDomainData' in a data entry. Skipping.")
                    continue

                channel_name = str(item["Channel"]).strip()  # Ensure it's a string
                time_domain_data = item["TimeDomainData"]

                # Validate time domain data
                if not isinstance(time_domain_data, list) or not time_domain_data:
                    print(f"⚠ Warning: Invalid or empty time domain data for channel {channel_name}. Skipping.")
                    continue

                # Initialize if channel does not exist
                if channel_name not in channel_data_dbs:
                    channel_data_dbs[channel_name] = []

                # Append data
                channel_data_dbs[channel_name].extend(time_domain_data)

            if not channel_data_dbs:
                print("⚠ Warning: No valid DBS data extracted.")
                return {}

            print(f"✔ Extracted {len(channel_data_dbs)} DBS channels.")  # Debugging
            return channel_data_dbs

        except Exception as e:
            print(f"❌ Unexpected error while extracting DBS data: {e}")
            return {}

    def create_montages(self, channel_data_dbs):
        """
        Creates montage signals for DBS data by computing differential signals.
        Montage signals are computed separately for right and left hemispheres.
        Returns a dictionary with the new montage signals.
        """
        
        if not channel_data_dbs:
            print("⚠ Error: No DBS channel data available for montage creation.")
            return {}

        montage_data_dbs = {}

        try:
            # Right Hemisphere Montages
            if all(channel in channel_data_dbs for channel in ["ZERO_THREE_RIGHT", "ONE_THREE_RIGHT"]):
                montage_data_dbs["ZERO_ONE_RIGHT"] = np.subtract(
                    channel_data_dbs["ZERO_THREE_RIGHT"], channel_data_dbs["ONE_THREE_RIGHT"]
                )

            if all(channel in channel_data_dbs for channel in ["ONE_THREE_RIGHT", "ZERO_TWO_RIGHT", "ZERO_THREE_RIGHT"]):
                montage_data_dbs["ONE_TWO_RIGHT"] = np.add(
                    channel_data_dbs["ONE_THREE_RIGHT"], channel_data_dbs["ZERO_TWO_RIGHT"]
                ) - channel_data_dbs["ZERO_THREE_RIGHT"]

            if all(channel in channel_data_dbs for channel in ["ZERO_THREE_RIGHT", "ZERO_TWO_RIGHT"]):
                montage_data_dbs["TWO_THREE_RIGHT"] = np.subtract(
                    channel_data_dbs["ZERO_THREE_RIGHT"], channel_data_dbs["ZERO_TWO_RIGHT"]
                )

            # Left Hemisphere Montages
            if all(channel in channel_data_dbs for channel in ["ZERO_THREE_LEFT", "ONE_THREE_LEFT"]):
                montage_data_dbs["ZERO_ONE_LEFT"] = np.subtract(
                    channel_data_dbs["ZERO_THREE_LEFT"], channel_data_dbs["ONE_THREE_LEFT"]
                )

            if all(channel in channel_data_dbs for channel in ["ONE_THREE_LEFT", "ZERO_TWO_LEFT", "ZERO_THREE_LEFT"]):
                montage_data_dbs["ONE_TWO_LEFT"] = np.add(
                    channel_data_dbs["ONE_THREE_LEFT"], channel_data_dbs["ZERO_TWO_LEFT"]
                ) - channel_data_dbs["ZERO_THREE_LEFT"]

            if all(channel in channel_data_dbs for channel in ["ZERO_TWO_LEFT", "ZERO_THREE_LEFT"]):
                montage_data_dbs["TWO_THREE_LEFT"] = np.subtract(
                    channel_data_dbs["ZERO_TWO_LEFT"], channel_data_dbs["ZERO_THREE_LEFT"]
                )

            print("✔ Montage signals successfully created.")
            return montage_data_dbs  # ✅ Return the dictionary instead of modifying self.

        except KeyError as e:
            print(f"⚠ Missing required channel data for montage: {e}")
            return {}
        except Exception as e:
            print(f"❌ Unexpected error while creating montages: {e}")
            return {}

    def get_signals_by_option_dbs(self, option):
        """
        Returns the signals and corresponding labels based on the selected option in the DBS GUI.
        Handles both original and montage channels efficiently while ensuring data integrity.
        """

        if not self.channel_data_dbs:
            print("⚠ No DBS data available.")
            return [], []

        # Check if montage data exists once at the beginning
        has_montage_data = bool(self.montage_data_dbs)

        # Helper function to fetch channels safely
        def fetch_channels(source, keys, source_name):
            missing_keys = [key for key in keys if key not in source]
            if missing_keys:
                print(f"⚠ Missing channels in {source_name}: {missing_keys}")
            return [source[key] for key in keys if key in source], [key for key in keys if key in source]

        if option == "Original Channels":
            print(f"🔎 Inside get_signals_by_option_dbs - Using {option}: First 10 samples of ZERO_THREE_LEFT: {self.channel_data_dbs['ZERO_THREE_LEFT'][:10]}")
            return list(self.channel_data_dbs.values()), list(self.channel_data_dbs.keys())

        elif option == "Montage Channels":
            if not has_montage_data:
                print("⚠ No montage data available.")
                return [], []
            return list(self.montage_data_dbs.values()), list(self.montage_data_dbs.keys())

        elif option == "Right Side (Original and Montage)":
            orig_keys = ["ZERO_THREE_RIGHT", "ONE_THREE_RIGHT", "ZERO_TWO_RIGHT"]
            montage_keys = ["ZERO_ONE_RIGHT", "ONE_TWO_RIGHT", "TWO_THREE_RIGHT"]
            orig_signals, orig_labels = fetch_channels(self.channel_data_dbs, orig_keys, "Original Data")
            montage_signals, montage_labels = fetch_channels(self.montage_data_dbs, montage_keys, "Montage Data") if has_montage_data else ([], [])
            return orig_signals + montage_signals, orig_labels + montage_labels

        elif option == "Left Side (Original and Montage)":
            orig_keys = ["ZERO_THREE_LEFT", "ONE_THREE_LEFT", "ZERO_TWO_LEFT"]
            montage_keys = ["ZERO_ONE_LEFT", "ONE_TWO_LEFT", "TWO_THREE_LEFT"]
            orig_signals, orig_labels = fetch_channels(self.channel_data_dbs, orig_keys, "Original Data")
            montage_signals, montage_labels = fetch_channels(self.montage_data_dbs, montage_keys, "Montage Data") if has_montage_data else ([], [])
            return orig_signals + montage_signals, orig_labels + montage_labels

        elif option == "Right Side (Original Only)":
            orig_keys = ["ZERO_THREE_RIGHT", "ONE_THREE_RIGHT", "ZERO_TWO_RIGHT"]
            return fetch_channels(self.channel_data_dbs, orig_keys, "Original Data")

        elif option == "Left Side (Original Only)":
            orig_keys = ["ZERO_THREE_LEFT", "ONE_THREE_LEFT", "ZERO_TWO_LEFT"]
            return fetch_channels(self.channel_data_dbs, orig_keys, "Original Data")

        elif option == "Right Side (Montage Only)":
            if not has_montage_data:
                print("⚠ No montage data available.")
                return [], []
            montage_keys = ["ZERO_ONE_RIGHT", "ONE_TWO_RIGHT", "TWO_THREE_RIGHT"]
            return fetch_channels(self.montage_data_dbs, montage_keys, "Montage Data")

        elif option == "Left Side (Montage Only)":
            if not has_montage_data:
                print("⚠ No montage data available.")
                return [], []
            montage_keys = ["ZERO_ONE_LEFT", "ONE_TWO_LEFT", "TWO_THREE_LEFT"]
            return fetch_channels(self.montage_data_dbs, montage_keys, "Montage Data")

        return [], []

    def update_dbs_plot(self):
        """
        Updates the DBS signal display with selected channels, zoom, and event markings.
        Ensures that signals are reset correctly when switching views.
        """

        selected_option = self.dbs_combo_box.currentText()
        if not selected_option or not self.channel_data_dbs:
            print("⚠ No DBS data loaded or no option selected.")
            return

        print(f"📢 DBS Zoom Factor: {self.zoom_factor_dbs}")  # Debugging

        if not self.band_pass_dbs_checkbox.isChecked():  
            # Only reset if trimming is also OFF
            if not self.start_sample_checkbox_dbs.isChecked():
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)


        print(f"📊 Before get_signals_by_option_dbs - First 10 samples of ZERO_THREE_LEFT: {self.channel_data_dbs['ZERO_THREE_LEFT'][:10]}")
        # ✅ Get the signals and labels based on user selection
        #print(f"🔎 Fetching signals for {selected_option}: First 10 samples of ZERO_THREE_LEFT: {self.channel_data_dbs['ZERO_THREE_LEFT'][:10]}")
        signals, labels = self.get_signals_by_option_dbs(selected_option)

        if not signals or not labels:
            print("⚠ No valid DBS signals found for selection.")
            return

        # ✅ Define the X-axis window length
        try:
            window_size_dbs = float(self.x_axis_length.text()) if self.apply_x_axis_checkbox.isChecked() else 20
            if window_size_dbs <= 0:
                raise ValueError("Window size must be greater than 0")
        except ValueError:
            print("⚠ Invalid X-axis length. Using default 20s.")
            window_size_dbs = 20

        # ✅ Compute sample range
        start_sample = int(self.current_start_time_dbs * self.sample_rate_dbs)
        end_sample = start_sample + int(window_size_dbs * self.sample_rate_dbs)

        # ✅ Ensure sample range is within bounds
        max_length = len(next(iter(self.channel_data_dbs.values())))
        end_sample = min(end_sample, max_length)
        start_sample = max(0, end_sample - int(window_size_dbs * self.sample_rate_dbs))

        time_axis = np.arange(start_sample, end_sample) / self.sample_rate_dbs

        # ✅ Clear figure (consider optimization by only updating plots)
        #print(f"📊 Final Plotting Data for ZERO_THREE_LEFT: {signals[0][:10]}")
        self.figure_dbs.clear()
        num_signals = len(signals)

        for i, (signal, label) in enumerate(zip(signals, labels)):
            signal = signal[start_sample:end_sample]

            ax = self.figure_dbs.add_subplot(num_signals, 1, i + 1)
            ax.plot(time_axis, signal, label=label, color='b')
            ax.legend(loc="upper right")
            #print(f"🖊 Plotting {label}: First 10 samples: {signal[:10]}")

            if i < num_signals - 1:
                ax.tick_params(axis='x', labelbottom=False)

            # ✅ Apply zoom factor and prevent min==max errors
            min_val, max_val = np.min(signal), np.max(signal)
            range_val = max_val - min_val
            epsilon = 1e-6  # Small offset to avoid zero-range issues
            ax.set_ylim(
                min_val - (range_val * (1 - self.zoom_factor_dbs) / 2) - epsilon,
                max_val + (range_val * (1 - self.zoom_factor_dbs) / 2) + epsilon
            )

            ax.margins(0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

            # ✅ Mark events on the plot
            for start, end in self.marked_events_dbs:
                if start_sample / self.sample_rate_dbs <= start <= end_sample / self.sample_rate_dbs:
                    ax.axvspan(start, end, color='red', alpha=0.3)

        self.figure_dbs.tight_layout()
        self.canvas_dbs.draw()
        print("📊 DBS plot updated successfully.")

    def apply_bandpass_dbs(self):
        """
        Applies or removes a band-pass filter to DBS signals when the checkbox is toggled.
        
        Ensures that:
        - Filtering is applied correctly to either the full signal or the already-trimmed version.
        - Trimming is preserved even after filtering.
        - Resetting filtering restores the correct state, whether trimmed or full-length.
        - It prevents unwanted cumulative processing.
        """

        print("🔍 Entering apply_bandpass_dbs()")
        
        if not self.channel_data_dbs_original:
            print("⚠ No DBS original data available for filtering.")
            return

        try:
            lowcut = float(self.lowcut_dbs.text())
            highcut = float(self.highcut_dbs.text())
            fs = self.sample_rate_dbs
        except ValueError:
            print("⚠ Invalid frequency values.")
            return

        if lowcut >= highcut or lowcut <= 0 or highcut >= fs / 2:
            print(f"⚠ Error: Invalid filter values (Lowcut: {lowcut}, Highcut: {highcut}, Fs: {fs})")
            return

        # ✅ Temporarily block signals to prevent unintended recursive updates
        self.band_pass_dbs_checkbox.blockSignals(True)

        if self.band_pass_dbs_checkbox.isChecked():
            print(f"✔ Applying Band-Pass filter (Lowcut: {lowcut} Hz, Highcut: {highcut} Hz)")

            # ✅ Determine whether to apply filtering on trimmed or full-length signals
            if hasattr(self, "trimmed_channel_data_dbs") and self.trimmed_channel_data_dbs:
                print("🟡 Filtering the trimmed version of the signal")
                self.channel_data_dbs = copy.deepcopy(self.trimmed_channel_data_dbs)
                self.montage_data_dbs = copy.deepcopy(self.trimmed_montage_data_dbs) if hasattr(self, "trimmed_montage_data_dbs") else {}
            else:
                print("🟢 Filtering the full-length signal")
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)

            # ✅ Apply filtering to all DBS channels
            for key in self.channel_data_dbs:
                self.channel_data_dbs[key] = self._bandpass_filter(self.channel_data_dbs[key], lowcut, highcut, fs)

            # ✅ Apply filtering to montage channels if they exist
            if self.montage_data_dbs:
                for key in self.montage_data_dbs:
                    self.montage_data_dbs[key] = self._bandpass_filter(self.montage_data_dbs[key], lowcut, highcut, fs)

            # ✅ Store the filtered version for future trimming
            self.filtered_channel_data_dbs = copy.deepcopy(self.channel_data_dbs)
            self.filtered_montage_data_dbs = copy.deepcopy(self.montage_data_dbs) if self.montage_data_dbs else {}

        else:
            print("🔄 Resetting DBS signals to previous state (without filter)")
            
            # ✅ Restore the correct state based on whether trimming was applied
            if hasattr(self, "trimmed_channel_data_dbs") and self.trimmed_channel_data_dbs:
                print("🔄 Restoring trimmed data without filtering.")
                self.channel_data_dbs = copy.deepcopy(self.trimmed_channel_data_dbs)
                self.montage_data_dbs = copy.deepcopy(self.trimmed_montage_data_dbs) if hasattr(self, "trimmed_montage_data_dbs") else {}
            else:
                print("🔄 Restoring full unfiltered data.")
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)

        self.band_pass_dbs_checkbox.blockSignals(False)  # ✅ Unblock checkbox signals

        # ✅ Refresh the plot
        print("📊 Calling update_dbs_plot()...")
        self.update_dbs_plot()

    def adjust_dbs_zoom(self, factor):
        """Adjusts the zoom level of the DBS signal display."""
        
        # ✅ Ensure zoom factor stays within valid bounds
        self.zoom_factor_dbs *= factor
        self.zoom_factor_dbs = max(0.2, min(self.zoom_factor_dbs, 5.0))  # Limit between 0.2 and 5.0

        print(f"🔍 Adjusting DBS Zoom: {self.zoom_factor_dbs}")  # Debugging

        # ✅ Update the DBS plot with the new zoom level
        self.update_dbs_plot()

    def apply_start_sample_dbs(self):
        """
        Adjusts the start point of DBS signals (both original and montage) based on user input.
        
        This function ensures that:
        - The user can reset the signals to their original length by unchecking the checkbox.
        - Trimming applies to both **filtered and unfiltered** versions without overriding previous processing.
        - The function correctly maintains filtering even after trimming.
        - It updates the DBS plot after changes.
        """

        print("🔍 Entering apply_start_sample_dbs()")

        # ✅ Ensure DBS data is loaded before modifying it
        if not self.channel_data_dbs:
            print("⚠ No DBS data loaded.")
            return

        # ✅ Case 1: Checkbox is unchecked → Restore original signals
        if not self.start_sample_checkbox_dbs.isChecked():
            # Restore the correct state based on whether filtering is applied
            if hasattr(self, "filtered_channel_data_dbs") and self.band_pass_dbs_checkbox.isChecked():
                print("🔄 Resetting to **filtered** data.")
                self.channel_data_dbs = copy.deepcopy(self.filtered_channel_data_dbs)
                self.montage_data_dbs = copy.deepcopy(self.filtered_montage_data_dbs) if hasattr(self, "filtered_montage_data_dbs") else {}
            else:
                print("🔄 Resetting to **original unfiltered** data.")
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)

            # ✅ Also reset stored trimmed data to avoid unintended cumulative trims
            self.trimmed_channel_data_dbs = None
            self.trimmed_montage_data_dbs = None

            print("✔ DBS signals restored to previous state.")
            self.update_dbs_plot()
            return  # Exit early

        # ✅ Case 2: Checkbox is checked → Trim signals based on user input
        try:
            start_sample = int(self.start_sample_input_dbs.text())  # Convert input to integer
            if start_sample < 0:
                print("⚠ Error: Start sample must be a non-negative integer.")
                return
        except ValueError:
            print("⚠ Error: Invalid start sample value (must be an integer).")
            return

        # ✅ Validate the start sample against signal length
        max_samples = len(next(iter(self.channel_data_dbs_original.values())))  # Get total samples from the original data
        if start_sample >= max_samples:
            print(f"⚠ Error: Start sample ({start_sample}) exceeds total samples ({max_samples}). Resetting to previous state.")
            
            # Reset based on whether filtering is active
            if hasattr(self, "filtered_channel_data_dbs") and self.band_pass_dbs_checkbox.isChecked():
                self.channel_data_dbs = copy.deepcopy(self.filtered_channel_data_dbs)
                self.montage_data_dbs = copy.deepcopy(self.filtered_montage_data_dbs) if hasattr(self, "filtered_montage_data_dbs") else {}
            else:
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)
        
        else:
            # ✅ Trim the **current version** of the signal (filtered or unfiltered)
            if hasattr(self, "filtered_channel_data_dbs") and self.band_pass_dbs_checkbox.isChecked():
                print("✂ Trimming **filtered** data.")
                self.channel_data_dbs = copy.deepcopy(self.filtered_channel_data_dbs)
                self.montage_data_dbs = copy.deepcopy(self.filtered_montage_data_dbs) if hasattr(self, "filtered_montage_data_dbs") else {}

            else:
                print("✂ Trimming **original unfiltered** data.")
                self.channel_data_dbs = copy.deepcopy(self.channel_data_dbs_original)
                self.montage_data_dbs = copy.deepcopy(self.montage_data_dbs_original)

            # ✅ Apply trimming on the selected version
            for key in self.channel_data_dbs:
                self.channel_data_dbs[key] = self.channel_data_dbs[key][start_sample:]

            if self.montage_data_dbs:
                for key in self.montage_data_dbs:
                    self.montage_data_dbs[key] = self.montage_data_dbs[key][start_sample:]

            print(f"✔ DBS signals updated - starting from sample {start_sample}.")

        # ✅ Store the trimmed version so filtering is applied to the trimmed signal
        self.trimmed_channel_data_dbs = copy.deepcopy(self.channel_data_dbs)
        self.trimmed_montage_data_dbs = copy.deepcopy(self.montage_data_dbs) if self.montage_data_dbs else {}

        # ✅ Update the DBS plot after trimming
        self.update_dbs_plot()

    # ----------------- Shared Functions (Both EEG & DBS) -----------------
    def toggle_marking_mode(self):
        """Enable marking mode for marking interictal events synchronously on both EEG and DBS plots."""
        self.marking_enabled = self.enable_marking_button.isChecked()
        
        if self.marking_enabled:
            # Connect mouse events for both EEG and DBS plots
            self.eeg_press_cid = self.canvas_eeg.mpl_connect('button_press_event', self.on_press)
            self.eeg_release_cid = self.canvas_eeg.mpl_connect('button_release_event', self.on_release)

            self.dbs_press_cid = self.canvas_dbs.mpl_connect('button_press_event', self.on_press)
            self.dbs_release_cid = self.canvas_dbs.mpl_connect('button_release_event', self.on_release)
        else:
            # Disconnect events when marking mode is turned off
            self.canvas_eeg.mpl_disconnect(self.eeg_press_cid)
            self.canvas_eeg.mpl_disconnect(self.eeg_release_cid)
            self.canvas_dbs.mpl_disconnect(self.dbs_press_cid)
            self.canvas_dbs.mpl_disconnect(self.dbs_release_cid)

    def on_press(self, event):
        """Start marking an interictal event."""
        if self.marking_enabled and event.inaxes:
            self.start_time = event.xdata

    def on_release(self, event):
        """Finish marking an interictal event and apply marking to both EEG and DBS."""
        if self.marking_enabled and event.inaxes and self.start_time is not None:
            end_time = event.xdata

            # Store the marked event for both EEG and DBS
            self.marked_events_eeg.append((self.start_time, end_time))
            self.marked_events_dbs.append((self.start_time, end_time))

            # Refresh both plots to show the new marked event
            self.update_eeg_plot()
            self.update_dbs_plot()

            # Reset marking mode
            self.start_time = None
            self.marking_enabled = False
            self.enable_marking_button.setChecked(False)  # Reset the checkbox

            print(f"Marked event on EEG & DBS: {self.marked_events_eeg[-1]}")

    def shift_both_views(self, shift):
        """
        Moves both EEG and DBS views forward or backward by the given shift in seconds.
        
        Parameters:
        shift (int): Number of seconds to shift the view (+ for forward, - for backward).
        """

        # Shift EEG view if EEG data exists
        if self.channel_data_eeg:
            max_time_eeg = (len(next(iter(self.channel_data_eeg.values()))) / self.sample_rate_eeg) - self.window_size_eeg
            self.current_start_time_eeg = max(0, min(self.current_start_time_eeg + shift, max_time_eeg))

        # Shift DBS view if DBS data exists
        if self.channel_data_dbs:
            max_time_dbs = (len(next(iter(self.channel_data_dbs.values()))) / self.sample_rate_dbs) - self.window_size_dbs
            self.current_start_time_dbs = max(0, min(self.current_start_time_dbs + shift, max_time_dbs))

        # Debugging print statements
        print(f"📊 Shifting both views by {shift} seconds")
        print(f"🔹 EEG Start Time: {self.current_start_time_eeg}s")
        print(f"🔹 DBS Start Time: {self.current_start_time_dbs}s")

        # Update both plots
        self.update_eeg_plot()
        self.update_dbs_plot()

    def delete_last_marked_event(self):
        """Delete the last marked event from both EEG and DBS views."""
        
        if self.marked_events_eeg and self.marked_events_dbs:
            # Remove the last event from both lists
            last_eeg_event = self.marked_events_eeg.pop()
            last_dbs_event = self.marked_events_dbs.pop()

            # Print for debugging
            print(f"❌ Deleted last marked event: EEG {last_eeg_event}, DBS {last_dbs_event}")

            # Update both plots
            self.update_eeg_plot()
            self.update_dbs_plot()
        else:
            print("⚠ No marked events to delete.")

    def save_marked_events(self):
        """Save marked events to a user-specified CSV file."""
        
        if not self.marked_events_eeg or not self.marked_events_dbs:
            print("⚠ No marked events to save.")
            return

        # Open a file dialog to let the user choose the save location and filename
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Marked Events", "", "CSV Files (*.csv);;All Files (*)", options=options
        )

        # If the user cancels, do nothing
        if not file_path:
            print("⚠ Save operation cancelled by the user.")
            return

        # Ensure the filename ends with ".csv"
        if not file_path.endswith(".csv"):
            file_path += ".csv"

        # Write marked events to the chosen file
        try:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Start Time (s)", "End Time (s)"])  # Header row
                
                for (start, end) in self.marked_events_eeg:  # Same for both EEG & DBS
                    writer.writerow([f"{start:.3f}", f"{end:.3f}"])
            
            print(f"✅ Marked events saved to {file_path}")

        except Exception as e:
            print(f"❌ Error saving marked events: {e}")

    def load_marked_events(self):
        """Load marked events from a CSV file and display them on both EEG and DBS plots."""
        
        # Open file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Marked Events File", "", "CSV Files (*.csv);;All Files (*)")

        # If the user cancels, do nothing
        if not file_path:
            print("⚠ Load operation cancelled by the user.")
            return

        # Clear existing marked events
        self.marked_events_eeg = []
        self.marked_events_dbs = []

        try:
            # Read the CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) < 2:
                        continue  # Skip malformed rows
                    
                    start, end = map(float, row[:2])
                    self.marked_events_eeg.append((start, end))
                    self.marked_events_dbs.append((start, end))  # Same times for both EEG & DBS
            
            # Refresh both plots
            self.update_eeg_plot()
            self.update_dbs_plot()

            print(f"✅ Loaded marked events from {file_path}")

        except Exception as e:
            print(f"❌ Error loading marked events: {e}")

    def update_plots_apply_x_axis_length(self):
        """Updates both EEG and DBS plots when X-axis settings change."""
        self.update_eeg_plot()
        self.update_dbs_plot()

    def _bandpass_filter(self, signal, lowcut, highcut, fs, order=4):
        """Internal function to apply a band-pass filter to a signal."""
        
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        if low >= high or low <= 0 or high >= 1:
            print(f"⚠ Filter Error: Invalid values (Lowcut: {lowcut}, Highcut: {highcut})")
            return signal  # Return the original signal if filter values are incorrect

        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGDBSViewer()
    window.show()
    sys.exit(app.exec_())














































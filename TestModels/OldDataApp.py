import time
from bitalino import BITalino
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
from PIL import Image, ImageTk

class EEGRecorder:
    def __init__(self, master):
        self.master = master
        master.title("EEG Recorder")
        master.geometry("1200x800")

        # BITalino setup
        self.macAddress = "0C:43:14:24:78:EA"
        self.batteryThreshold = 30
        self.acqChannels = [3]
        self.samplingRate = 100
        self.nSamples = 10

        # Data storage
        self.display_seconds = 10
        self.max_points = self.display_seconds * self.samplingRate
        self.data = deque(maxlen=self.max_points)
        self.x_data = deque(maxlen=self.max_points)
        self.recording_data = []

        # Main frame
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left padding frame
        left_padding = ttk.Frame(main_frame, width=100)
        left_padding.pack(side=tk.LEFT, fill=tk.Y)

        # Center frame for valence-arousal image and countdown
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, padx=50, pady=200, fill=tk.BOTH, expand=True)

        # Load and display valence-arousal image
        self.va_image = Image.open("valence_arousal.png")
        self.va_image = self.va_image.resize((400, 400), Image.LANCZOS)
        self.va_photo = ImageTk.PhotoImage(self.va_image)
        self.va_canvas = tk.Canvas(center_frame, width=400, height=400)
        self.va_canvas.pack(fill=tk.BOTH, expand=True)
        self.va_canvas.create_image(0, 0, anchor=tk.NW, image=self.va_photo)
        self.va_canvas.bind("<Motion>", self.update_va_coordinates)

        # Countdown label
        self.countdown_var = tk.StringVar(value="")
        self.countdown_label = ttk.Label(center_frame, textvariable=self.countdown_var, font=('Arial', 18))
        self.countdown_label.pack(pady=10)

        # Keyboard instructions
        instructions = "Press 'S' to start recording, 'Q' to stop recording"
        self.instructions_label = ttk.Label(center_frame, text=instructions, font=('Arial', 12))
        self.instructions_label.pack(pady=10)

        # Right frame for EEG plot, controls, and logger
        self.right_frame = ttk.Frame(main_frame, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(self.right_frame)
        control_frame.pack(pady=10)

        # Connection status
        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 14))
        self.status_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Control buttons
        button_style = ttk.Style()
        button_style.configure('TButton', font=('Arial', 14), padding=15)

        self.connect_button = ttk.Button(control_frame, text="Connect", command=self.connect, style='TButton')
        self.connect_button.grid(row=1, column=0, padx=10)

        self.disconnect_button = ttk.Button(control_frame, text="Disconnect", command=self.disconnect, state=tk.DISABLED, style='TButton')
        self.disconnect_button.grid(row=1, column=1, padx=10)

        # Toggle buttons
        self.toggle_graph_button = ttk.Button(control_frame, text="Toggle Graph", command=self.toggle_graph, style='TButton')
        self.toggle_graph_button.grid(row=2, column=0, padx=10, pady=10)

        self.toggle_logger_button = ttk.Button(control_frame, text="Toggle Logger", command=self.toggle_logger, style='TButton')
        self.toggle_logger_button.grid(row=2, column=1, padx=10, pady=10)

        # Plotting setup
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.line, = self.ax.plot([], [])
        self.ax.set_title("Real-time EEG Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("EEG Value")
        self.ax.set_xlim(0, self.display_seconds)

        # Create a canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Logger
        self.logger = scrolledtext.ScrolledText(self.right_frame, height=10)
        self.logger.pack(pady=10, fill=tk.BOTH, expand=True)

        self.is_connected = False
        self.is_recording = False
        self.device = None
        self.valence = 0
        self.arousal = 0

        # Bind keyboard events
        self.master.bind('<s>', self.start_recording_countdown)
        self.master.bind('<q>', self.stop_recording)

    def connect(self):
        if not self.is_connected:
            try:
                self.device = BITalino(self.macAddress)
                self.device.battery(self.batteryThreshold)
                print(self.device.version())
                self.device.start(self.samplingRate, self.acqChannels)
                self.is_connected = True
                self.status_var.set("Connected")
                self.connect_button.config(state=tk.DISABLED)
                self.disconnect_button.config(state=tk.NORMAL)
                
                # Start updating plot in a separate thread
                self.update_thread = threading.Thread(target=self.update_plot_loop)
                self.update_thread.start()
            except Exception as e:
                print(f"Error connecting to device: {e}")

    def disconnect(self):
        if self.is_connected:
            self.is_connected = False
            self.status_var.set("Disconnected")
            self.connect_button.config(state=tk.NORMAL)
            self.disconnect_button.config(state=tk.DISABLED)
            self.is_recording = False
            
            if self.device:
                self.device.stop()
                self.device.close()
                self.device = None

    def start_recording_countdown(self, event=None):
        if self.is_connected and not self.is_recording:
            self.countdown_thread = threading.Thread(target=self.countdown)
            self.countdown_thread.start()

    def countdown(self):
        for i in range(5, 0, -1):
            self.countdown_var.set(f"Recording starts in {i}")
            time.sleep(1)
        self.countdown_var.set("Recording...")
        self.is_recording = True
        self.recording_data = []

    def stop_recording(self, event=None):
        if self.is_recording:
            self.is_recording = False
            self.countdown_var.set("Recording stopped")
            self.save_recording()

    def update_plot_loop(self):
        while self.is_connected:
            try:
                samples = self.device.read(self.nSamples)
                current_time = time.time()
                for sample in samples:
                    eeg_value = sample[-1]
                    self.data.append(eeg_value)
                    self.x_data.append(current_time)
                    if self.is_recording:
                        self.recording_data.append((eeg_value, self.valence, self.arousal))
                    self.log_data(eeg_value, self.valence, self.arousal)

                self.update_plot()
                time.sleep(0.01)  # Small delay to prevent overwhelming the GUI
            except Exception as e:
                print(f"Error reading data: {e}")
                self.disconnect()
                break

    def update_plot(self):
        if len(self.x_data) > 1:
            x_array = np.array(self.x_data)
            x_array -= x_array[-1] - self.display_seconds  # Shift x-axis to show last 10 seconds
            self.line.set_data(x_array, self.data)
            self.ax.set_xlim(0, self.display_seconds)
            y_min, y_max = min(self.data), max(self.data)
            y_range = y_max - y_min
            y_buffer = y_range * 0.1
            self.ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
            self.canvas.draw()

    def update_va_coordinates(self, event):
        self.valence = max(-1, min(1, 2 * (event.x / 400) - 1))  # Normalize to -1 to 1
        self.arousal = max(-1, min(1, 1 - 2 * (event.y / 400)))  # Normalize to -1 to 1, invert y-axis

    def log_data(self, eeg, valence, arousal):
        log_text = f"EEG: {eeg:.2f}, Valence: {valence:.2f}, Arousal: {arousal:.2f}\n"
        self.logger.insert(tk.END, log_text)
        self.logger.see(tk.END)  # Scroll to the end

    def save_recording(self):
        if self.recording_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if file_path:
                with open(file_path, 'w') as f:
                    for eeg, valence, arousal in self.recording_data:
                        f.write(f"{eeg},{valence:.4f},{arousal:.4f}\n")
                print(f"Recording saved to {file_path}")
        else:
            print("No data to save.")

    def toggle_graph(self):
        if self.canvas_widget.winfo_viewable():
            self.canvas_widget.pack_forget()
        else:
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def toggle_logger(self):
        if self.logger.winfo_viewable():
            self.logger.pack_forget()
        else:
            self.logger.pack(pady=10, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGRecorder(root)
    root.mainloop()
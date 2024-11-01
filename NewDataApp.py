import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from DSI import Headset, IfStringThenNormalString, SampleCallback, MessageCallback, DSIException
import threading
from PIL import Image, ImageTk

# Global variables
global_main_window = None
recording = False
countdown_active = False
valence_arousal_data = []
valence, arousal = 0, 0
sample_counter = 0  # Counter for effective sampling

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("DSI Headset Data Logger")
        self.root.geometry("1200x800")
        
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)
        
        self.top_frame = ttk.Frame(self.paned_window)
        self.bottom_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.top_frame, weight=1)
        self.paned_window.add(self.bottom_frame, weight=1)
        
        self.top_left_frame = ttk.Frame(self.top_frame)
        self.top_right_frame = ttk.Frame(self.top_frame)
        self.top_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.top_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        
        # Load and display the valence-arousal image
        self.va_image = Image.open("valence_arousal.png")
        self.va_image = self.va_image.resize((400, 400), Image.LANCZOS)
        self.va_photo = ImageTk.PhotoImage(self.va_image)
        self.va_canvas = tk.Canvas(self.top_left_frame, width=400, height=400)
        self.va_canvas.pack(padx=20, pady=20)
        self.va_canvas.create_image(200, 200, image=self.va_photo)
        
        self.cursor_label = ttk.Label(self.top_left_frame, text="Cursor: (0.00, 0.00)")
        self.cursor_label.pack()
        
        self.countdown_label = ttk.Label(self.top_left_frame, text="")
        self.countdown_label.pack()
        
        # Add instructions label
        self.instructions_label = ttk.Label(self.top_left_frame, text="Press 's' to start recording, 'q' to stop and save data", wraplength=300)
        self.instructions_label.pack(pady=10)
        
        self.va_canvas.bind("<Motion>", self.update_cursor_position)
        self.root.bind("s", self.start_countdown)
        self.root.bind("q", self.stop_recording)
        
        self.text_area = scrolledtext.ScrolledText(self.bottom_frame, wrap=tk.WORD, height=10)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Logger toggle button
        self.logger_toggle_var = tk.BooleanVar(value=True)
        self.logger_toggle_button = ttk.Checkbutton(self.top_right_frame, text="Toggle Logger", variable=self.logger_toggle_var, command=self.toggle_logger)
        self.logger_toggle_button.pack(pady=10)

    def toggle_logger(self):
        if self.logger_toggle_var.get():
            self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        else:
            self.text_area.pack_forget()

    def update_text(self, text):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)

    def update_cursor_position(self, event):
        global valence, arousal
        x = (event.x - 200) / 200  # Convert to -1 to 1 range
        y = -(event.y - 200) / 200  # Convert to -1 to 1 range and invert y-axis
        x = max(-1, min(1, x))  # Clamp values between -1 and 1
        y = max(-1, min(1, y))
        self.cursor_label.config(text=f"Cursor: ({x:.2f}, {y:.2f})")
        valence, arousal = x, y
        
    def start_countdown(self, event):
        global countdown_active
        if not countdown_active:
            countdown_active = True
            self.countdown(5)
    
    def countdown(self, count):
        global countdown_active, recording
        if count > 0:
            self.countdown_label.config(text=f"Recording in {count}...")
            self.root.after(1000, self.countdown, count - 1)
        else:
            self.countdown_label.config(text="Recording!")
            recording = True
            countdown_active = False
    
    def stop_recording(self, event):
        global recording
        if recording:
            recording = False
            self.countdown_label.config(text="Recording stopped")
            self.save_data()
    
    def save_data(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            with open(filename, "w") as f:
                for i, data in enumerate(valence_arousal_data):
                    if i % 3 == 0:  # Save every third data point
                        channels_data = " ".join([f"{name}={value:+08.2f}" for name, value in data['channels'].items()])
                        f.write(f"{data['time']},{data['valence']},{data['arousal']}, {channels_data}\n")
            self.update_text(f"Data saved to {filename}")
            valence_arousal_data.clear()

@SampleCallback
def on_sample_received(headsetPtr, packetTime, userData):
    global recording, valence, arousal, sample_counter
    headset = Headset(headsetPtr)
    output = f"{packetTime:.3f}: Valence={valence:.2f}, Arousal={arousal:.2f}"
    channel_data = {}
    for channel in headset.Channels():
        channel_name = IfStringThenNormalString(channel.GetName())
        channel_value = channel.ReadBuffered()
        output += f" {channel_name}={channel_value:+08.2f}"
        channel_data[channel_name] = channel_value
        
    if global_main_window:
        global_main_window.root.after(0, global_main_window.update_text, output)
        
        if recording:
            valence_arousal_data.append({
                "time": packetTime,
                "valence": valence,
                "arousal": arousal,
                "channels": channel_data
            })
            sample_counter += 1

@MessageCallback
def on_message_received(msg, lvl=0):
    if lvl <= 3:
        message = f"DSI Message (level {lvl}): {IfStringThenNormalString(msg)}"
        if global_main_window:
            global_main_window.root.after(0, global_main_window.update_text, message)
    return 1

def connect_and_acquire_data(port, reference=''):
    try:
        headset = Headset()
        headset.SetMessageCallback(on_message_received)
        headset.Connect(port)
        
        if reference.lower().startswith('imp'):
            headset.SetSampleCallback(on_sample_received, 0)
            headset.StartImpedanceDriver()
        else:
            headset.SetSampleCallback(on_sample_received, 0)
            if reference:
                headset.SetDefaultReference(reference, True)
        
        headset.StartDataAcquisition()
        message = "Data acquisition started..."
        if global_main_window:
            global_main_window.root.after(0, global_main_window.update_text, message)
        
        while True:
            headset.Idle(0.1)  # Reduced idle time for responsiveness

    except DSIException as e:
        error_message = f"Error occurred: {e}"
        if global_main_window:
            global_main_window.root.after(0, global_main_window.update_text, error_message)

if __name__ == '__main__':
    root = tk.Tk()
    main_window = MainWindow(root)
    global_main_window = main_window  # Set the global reference

    args = sys.argv
    if sys.platform.lower().startswith('win'):
        # default_port = 'COM7' # pc
        default_port = 'COM5' # laptop
    else:
        default_port = '/dev/cu.DSI7-0009.BluetoothSeri'

    port = args[1] if len(args) > 1 else default_port
    ref = args[2] if len(args) > 2 else ''

    threading.Thread(target=connect_and_acquire_data, args=(port, ref), daemon=True).start()

    root.mainloop()
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from DSI import Headset, IfStringThenNormalString, SampleCallback, MessageCallback, DSIException
import threading
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import pywt
import os

SAMPLING_RATE = 100 # Hz
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Using device: {device}")
# print(f"Device name: {torch.cuda.get_device_name(0)}")


    
def pca_for_tensor_array(tensor_array, n_components, center=True, normalize=False, eps=1e-7):
    """
    Apply PCA to an array of PyTorch tensors.

    Args:
    tensor_array (list): List of PyTorch tensors, each of shape (n, m) where n is the number of samples
                         and m is the number of features.
    n_components (int): Number of principal components to keep.
    center (bool): Whether to center the data before applying PCA. Default is True.
    normalize (bool): Whether to normalize the data before applying PCA. Default is False.
    eps (float): Small value to add for numerical stability in normalization. Default is 1e-7.

    Returns:
    list: List of PyTorch tensors after PCA transformation, each of shape (n, n_components).
    dict: Dictionary containing 'components' (the PCA components), 'mean' (mean of the data),
          and 'std' (standard deviation of the data, if normalization was applied).
    """
    # Concatenate all tensors in the array
    combined_tensor = torch.cat(tensor_array, dim=0)
    
    # Center the data
    if center:
        mean = torch.mean(combined_tensor, dim=0, keepdim=True)
        combined_tensor = combined_tensor - mean
    else:
        mean = None

    # Normalize the data
    if normalize:
        std = torch.std(combined_tensor, dim=0, keepdim=True)
        combined_tensor = combined_tensor / (std + eps)
    else:
        std = None

    # Perform PCA
    U, S, V = torch.svd(combined_tensor)
    
    # Select top n_components
    components = V[:, :n_components]
    
    # Project data onto principal components
    projected_data = torch.matmul(combined_tensor, components)
    
    # Split the projected data back into separate tensors
    result_tensors = []
    start_idx = 0
    for tensor in tensor_array:
        end_idx = start_idx + tensor.shape[0]
        result_tensors.append(projected_data[start_idx:end_idx])
        start_idx = end_idx
    
    # Prepare return values
    pca_info = {
        'components': components,
        'mean': mean,
        'std': std
    }
    
    return result_tensors, pca_info

def apply_pca_transform(new_data, pca_info, center=True, normalize=False, eps=1e-7):
    """
    Apply PCA transformation to new data using pre-computed PCA information.

    Args:
    new_data (torch.Tensor): New data to transform, of shape (n, m).
    pca_info (dict): Dictionary containing PCA information from pca_for_tensor_array.
    center (bool): Whether to center the data before applying PCA. Default is True.
    normalize (bool): Whether to normalize the data before applying PCA. Default is False.
    eps (float): Small value to add for numerical stability in normalization. Default is 1e-7.

    Returns:
    torch.Tensor: Transformed data of shape (n, n_components).
    """
    if center and pca_info['mean'] is not None:
        new_data = new_data - pca_info['mean']
    
    if normalize and pca_info['std'] is not None:
        new_data = new_data / (pca_info['std'] + eps)
    
    return torch.matmul(new_data, pca_info['components'])

def save_pca_info(pca_info, filename):
    """
    Save PCA information to a file.

    Args:
    pca_info (dict): Dictionary containing PCA information.
    filename (str): Name of the file to save the PCA information.
    """
    torch.save(pca_info, filename)

def load_pca_info(filename):
    """
    Load PCA information from a file.

    Args:
    filename (str): Name of the file to load the PCA information from.

    Returns:
    dict: Dictionary containing PCA information.
    """
    return torch.load(filename)

def uwt(eeg_tensor, levels=5, normalize=True):
    
    assert len(eeg_tensor.shape) == 2 and eeg_tensor.shape[1] == 1, "Input should be a [n, 1] tensor"
    
    device = eeg_tensor.device # Store the device of the input tensor
    eeg_np = eeg_tensor.cpu().numpy().flatten() # Move input to CPU for numpy operations
    
    # wavelet = pywt.Wavelet('db4')
    wavelet = pywt.Wavelet('db20')
    # wavelet = pywt.Wavelet('gaus')
    # print(pywt.wavelist())
    dec_lo, dec_hi, _, _ = wavelet.filter_bank
    
    pad_len = (len(dec_lo) - 1) * (2**levels)
    eeg_padded = np.pad(eeg_np, (pad_len, pad_len), mode='reflect')
    
    detail_coeffs = []
    
    approx = eeg_padded
    for i in range(levels):
        filter_len = len(dec_lo) * (2**i)
        lo = np.zeros(filter_len)
        hi = np.zeros(filter_len)
        lo[0::2**i] = dec_lo
        hi[0::2**i] = dec_hi
        
        detail = np.convolve(approx, hi, mode='same')
        approx = np.convolve(approx, lo, mode='same')
        
        if normalize:
            detail = detail / np.sqrt(2**i)
        
        detail_coeffs.append(detail[pad_len:-pad_len])
    
    approx_coeffs = approx[pad_len:-pad_len]
    if normalize:
        approx_coeffs = approx_coeffs / np.sqrt(2**levels)
    
    # Convert to PyTorch tensors and move to the original device
    gamma = torch.tensor(detail_coeffs[0], device=device).view(-1, 1).float()
    beta = torch.tensor(detail_coeffs[1], device=device).view(-1, 1).float()
    alpha = torch.tensor(detail_coeffs[2], device=device).view(-1, 1).float()
    theta = torch.tensor(detail_coeffs[3], device=device).view(-1, 1).float()
    delta = torch.tensor(approx_coeffs, device=device).view(-1, 1).float()
    
    return gamma, beta, alpha, theta, delta

def multiple_uwt(datas_array, levels=5, normalize=True, preserve_data=False):
    """
    Perform Undecimated Wavelet Transform (UWT) on each column of each tensor in the input array.
    
    Args:
    datas_array (list): List of torch.Tensor, each of shape (n, m) where n is the number of time steps
                        and m is the number of EEG channels.
    levels (int): Number of decomposition levels for UWT. Default is 5.
    normalize (bool): Whether to normalize the data before UWT. Default is True.
    preserve_data (bool): Whether to include the original data in the output. Default is False.
    
    Returns:
    list: List of torch.Tensor. If preserve_data is True, each tensor has shape (n, 6m) where the 
          first m columns are the original data, and the remaining 5m columns are the UWT decompositions
          (gamma, beta, alpha, theta, delta) for each of the m channels. If preserve_data is False, 
          each tensor has shape (n, 5m) containing only the UWT decompositions.
    """
    combined_array = []
    for data in datas_array:
        n, m = data.shape
        uwt_results = []
        
        for channel in range(m):
            channel_data = data[:, channel].unsqueeze(1)  # (n, 1)
            data_g, data_b, data_a, data_t, data_d = uwt(eeg_tensor=channel_data, levels=levels, normalize=normalize)
            channel_uwt = torch.cat((data_g, data_b, data_a, data_t, data_d), dim=1)  # (n, 5)
            uwt_results.append(channel_uwt)
        
        all_uwt = torch.cat(uwt_results, dim=1)  # (n, 5m)
        
        if preserve_data:
            combined = torch.cat((data, all_uwt), dim=1)  # (n, 6m)
        else:
            combined = all_uwt  # (n, 5m)
        
        combined_array.append(combined)
    
    return combined_array


def load_mlp(file_path):
    # Load the model
    model_state_dict = torch.load(file_path, map_location=torch.device('cuda'))
    
    # Recreate the MLP structure (you need to know the layer sizes)
    neuron_layers = [model_state_dict[f'layers.{i}.weight'].shape[1] for i in range(len(model_state_dict) // 2)]
    neuron_layers.append(model_state_dict[f'layers.{len(neuron_layers) - 1}.weight'].shape[0])
    
    model = MLP(neuron_layers).to('cuda')
    
    # Load the state dict into the model
    model.load_state_dict(model_state_dict)
    
    return model

def load_ltc(file_path):
    # Load the model
    model_state_dict = torch.load(file_path, map_location=torch.device('cuda'))
    
    # Recreate the LTC structure (you need to know the layer sizes)
    ltc_layers = [model_state_dict[f'layers.{i}.weight'].shape[1] for i in range(len(model_state_dict) // 2)]
    ltc_layers.append(model_state_dict[f'layers.{len(ltc_layers) - 1}.weight'].shape[0])
    
    model = LTC(ltc_layers).to('cuda')
    
    # Load the state dict into the model
    model.load_state_dict(model_state_dict)
    
    return model

def load_pca_info(filename):
    return torch.load(filename, map_location=torch.device('cuda'))


class MLP(nn.Module):
    def __init__(self, neuron_layers):
        super(MLP, self).__init__()
        
        if len(neuron_layers) < 2:
            raise ValueError("At least input and output layer sizes must be specified")
        
        self.layers = nn.ModuleList()
        for i in range(len(neuron_layers) - 1):
            self.layers.append(nn.Linear(neuron_layers[i], neuron_layers[i+1], bias=True))
    
    def to(self, device):
        super(MLP, self).to(device)
        return self
    
    def forward(self, x):
        x = x.to(device)
        # x shape: [samples, features]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.tanh(x)
        return x  # Return shape: [samples,  outputs]
    
    def create_outputs_sequence(self, hidden_state_sequence):
        hidden_state_sequence = hidden_state_sequence.to(device)
        with torch.no_grad():
            return self.forward(hidden_state_sequence)
    
    def train_model(self, training_data, labels, optimizer, criterion, epochs):
        # clear_cuda_cache()
        training_data = training_data.to(device)
        labels = labels.to(device)
        
        training_data = training_data.detach()
        labels = labels.detach()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(training_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                print(f'[Epoch {epoch + 1}] loss: {loss.item():.5f}')

        print('Finished Training')
        
    def advanced_train_model(self, datas_array, labels_array, optimizer, criterion, epochs):
        datas_array = [data.to(device) for data in datas_array]
        labels_array = [label.to(device) for label in labels_array]
        
        combined_data = torch.cat(datas_array, dim=0)
        combined_labels = torch.cat(labels_array, dim=0)

        for epoch in range(epochs):
            # num_samples = combined_data.shape[0]
            # perm = torch.randperm(num_samples)

            # shuffled_data = combined_data[perm]
            # shuffled_labels = combined_labels[perm]
            
            optimizer.zero_grad()
            
            # outputs = self(shuffled_data)            
            outputs = self(combined_data)            
            
            # loss = criterion(outputs, shuffled_labels)
            loss = criterion(outputs, combined_labels)
            
            loss.backward()
            
            # # Divide gradients by the batch size (full batch)
            # for param in self.parameters():
            #     if param.grad is not None:
            #         param.grad / combined_data.shape[0]
        
            optimizer.step()

            if epoch % 1 == 0:
                print(f'[Epoch {epoch + 1}] loss: {loss.item():.5f}')

        print('Finished Advanced Training')

class LTC(nn.Module):
    def __init__(self, LTC_layers):
        super(LTC, self).__init__()
        
        self.activation = nn.ReLU()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(LTC_layers) - 1):
            layer = nn.Linear(LTC_layers[i], LTC_layers[i+1], bias=True)
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            self.layers.append(layer) 
        
    def forward(self, input, hidden):
        hidden = hidden.view(input.shape[0], -1, self.layers[-1].out_features)
        
        # print(input.shape)
        print(hidden.shape)
        
        combined = torch.cat((input, hidden), dim=2)
        
        for layer in self.layers:
            combined = layer(combined)
            combined = self.activation(combined)
            
        return combined

    def create_hidden_state_sequences(self, data_array, tau, A, step_size):
        device = next(self.parameters()).device
        max_sequence_length = max(data.shape[0] for data in data_array)
        batch_size = len(data_array)
        
        self.eval()
        
        # Create a padded tensor for all sequences
        padded_data = torch.zeros((batch_size, max_sequence_length, data_array[0].shape[1]), device=device)
        for i, data in enumerate(data_array):
            padded_data[i, :data.shape[0], :] = data
        
        # Create a mask for valid timesteps
        mask = torch.arange(max_sequence_length, device=device)[None, :] < torch.tensor([len(data) for data in data_array], device=device)[:, None]
        
        hidden_state_sequences = torch.zeros((batch_size, max_sequence_length, self.layers[-1].out_features), device=device)
        hidden_state = torch.zeros((batch_size, 1, self.layers[-1].out_features), device=device)
        
        for i in range(max_sequence_length):
            input = padded_data[:, i:i+1, :]
            
            ltc_output = self.forward(input, hidden_state)
            
            # Fused step
            num = hidden_state + step_size * ltc_output * A
            den = 1 + step_size * (1/tau + ltc_output)
            new_hidden_state = num / den
            
            # Update only valid sequences
            hidden_state = torch.where(mask[:, i:i+1, None], new_hidden_state, hidden_state)
            hidden_state_sequences[:, i:i+1, :] = hidden_state
        
        # Create a list of tensors, each containing the non-padded sequence
        result = [seq[:mask[i].sum()] for i, seq in enumerate(hidden_state_sequences)]
        
        return result

    def forward_2d(self, input, hidden):
        """
        Forward pass for 2D input tensors.
        
        Args:
        input (torch.Tensor): 2D input tensor of shape (sequence_length, features)
        hidden (torch.Tensor): 2D hidden state tensor of shape (1, hidden_features)
        
        Returns:
        torch.Tensor: 2D output tensor of shape (sequence_length, hidden_features)
        """
        sequence_length, features = input.shape
        hidden = hidden.view(1, 1, -1).expand(1, sequence_length, -1)
        input = input.unsqueeze(0)  # Add batch dimension
        
        combined = torch.cat((input, hidden), dim=2)
        
        for layer in self.layers:
            combined = layer(combined)
            combined = self.activation(combined)
        
        return combined.squeeze(0)  # Remove batch dimension

    def create_hidden_state_2d(self, data, tau, A, step_size):
        """
        Create hidden state sequence for a single 2D input tensor.
        
        Args:
        data (torch.Tensor): 2D input tensor of shape (sequence_length, features)
        tau (float): Time constant
        A (float): Amplitude factor
        step_size (float): Step size for integration
        
        Returns:
        torch.Tensor: 2D hidden state tensor of shape (sequence_length, hidden_features)
        """
        device = next(self.parameters()).device
        sequence_length, _ = data.shape
        
        self.eval()
        
        hidden_state_sequence = torch.zeros((sequence_length, self.layers[-1].out_features), device=device)
        hidden_state = torch.zeros((1, self.layers[-1].out_features), device=device)
        
        for i in range(sequence_length):
            input = data[i:i+1, :]  # Select single timestep
            
            ltc_output = self.forward_2d(input.unsqueeze(0), hidden_state)
            
            # Fused step
            num = hidden_state + step_size * ltc_output * A
            den = 1 + step_size * (1/tau + ltc_output)
            new_hidden_state = num / den
            
            hidden_state = new_hidden_state
            hidden_state_sequence[i, :] = hidden_state.squeeze()
        
        return hidden_state_sequence


# Global variables
global_main_window = None
recording = False
countdown_active = False
valence_arousal_data = []
valence, arousal = 0, 0
sample_counter = 0  # Counter for effective sampling
buffer_data = torch.zeros(1, 7, device=device)  # PyTorch tensor of size 1x7 for EEG data
step_size = 0.01
tau = 0.3
A = 1

mlp = load_mlp(r"C:\Users\danie\OneDrive\Desktop\EEGStuff-main81-80-main\MLP-3HS-OUT.pth")
ltc = load_ltc(r"C:\Users\danie\OneDrive\Desktop\EEGStuff-main81-80-main\LTC-T099-A1.pth")
pca_info = load_pca_info(r"C:\Users\danie\OneDrive\Desktop\EEGStuff-main81-80-main\PCA_info.pkl")

buffer_hidden_state = torch.zeros(1, ltc.layers[-1].out_features, device=device)

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
        
        # Create the red dot
        self.dot = self.va_canvas.create_oval(0, 0, 10, 10, fill='red', outline='red')
        self.initialise_dot_position()  # Initial position
        
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
    
    def initialise_dot_position(self):
        x = 180
        y = 200
        self.va_canvas.coords(self.dot, x, y, x+10, y+10)  # Update dot position
    
    def update_dot_position(self, buffer_data):
        global buffer_hidden_state
        
        # Apply UWT
        buffer_data_uwt = multiple_uwt([buffer_data], preserve_data=True)
        
        # Apply PCA
        buffer_data_pca = apply_pca_transform(buffer_data_uwt[0], pca_info)
        
        valence, arousal = 0, 0
        
        with torch.no_grad():
            ltc_output = ltc.forward_2d(buffer_data_pca, buffer_hidden_state)
            
            # Fused step
            num = buffer_hidden_state + step_size * ltc_output * A
            den = 1 + step_size * (1/tau + ltc_output)
            buffer_hidden_state = num / den
            
            # Generate output
            output = mlp.forward(buffer_hidden_state)
            valence, arousal = output[0, 0].item(), output[0, 1].item()
            
        
        # Convert valence and arousal to canvas coordinates
        x = (valence + 1) * 200  # Convert from [-1, 1] to [0, 400]
        y = (1 - arousal) * 200  # Convert from [-1, 1] to [400, 0] (inverted y-axis)
        
        # Update dot position
        self.va_canvas.coords(self.dot, x-5, y-5, x+5, y+5)
        
        # Update the cursor label
        self.cursor_label.config(text=f"Predicted: ({valence:.2f}, {arousal:.2f})")
        
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
                for data in valence_arousal_data:
                    f.write(f"{data['time']},{data['valence']},{data['arousal']},{','.join(map(str, data['eeg_data']))}\n")
            self.update_text(f"Data saved to {filename}")
            valence_arousal_data.clear()

@SampleCallback
def on_sample_received(headsetPtr, packetTime, userData):
    global recording, valence, arousal, sample_counter, global_main_window, buffer_data
    headset = Headset(headsetPtr)
    output = f"{packetTime:.3f}: Valence={valence:.2f}, Arousal={arousal:.2f}"
    
    channels = headset.Channels()
    channel_names = ['Pz', 'F4', 'C4', 'P4', 'P3', 'C3', 'F3']
    
    # Find the index for each channel and read its value
    eeg_data = []
    for channel_name in channel_names:
        channel_index = next((i for i, ch in enumerate(channels) if IfStringThenNormalString(ch.GetName()) == channel_name), None)
        if channel_index is not None:
            value = channels[channel_index].ReadBuffered()
            eeg_data.append(value)
            output += f" {channel_name}={value:+08.2f}"
        else:
            eeg_data.append(0)  # If channel not found, append 0
            output += f" {channel_name}=N/A"
    
    # Update buffer_data
    buffer_data = torch.roll(buffer_data, shifts=-1, dims=0)
    buffer_data[-1] = torch.tensor(eeg_data)
    
    # # Print the contents of buffer_data for debugging
    # print("Buffer Data:")
    # print(buffer_data)
    # print("---")
    
    if global_main_window:
        global_main_window.root.after(0, global_main_window.update_text, output)
        global_main_window.root.after(0, lambda: global_main_window.update_dot_position(buffer_data))
        
        if recording:
            valence_arousal_data.append({
                "time": packetTime,
                "valence": valence,
                "arousal": arousal,
                "eeg_data": eeg_data
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
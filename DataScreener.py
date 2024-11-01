import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from tkinter import filedialog, Tk
import sys

def read_eeg_file(file_path):
    """Read and parse EEG data file"""
    data = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        try:
            parts = line.strip().split(',')
            timestamp = float(parts[0])
            valence = float(parts[1])
            arousal = float(parts[2])
            
            channel_data = {
                'TimeStamp': timestamp,
                'Valence': valence,
                'Arousal': arousal
            }
            
            eeg_parts = parts[3].split()
            for part in eeg_parts:
                if not part.startswith('TRG='):
                    name, value = part.split('=')
                    channel_data[name] = float(value)
            
            data.append(channel_data)
            
        except Exception as e:
            print(f"Error processing line: {str(e)}")
            continue
    
    return pd.DataFrame(data)

def plot_eeg_data(file_path):
    """Plot EEG data with interactive buttons"""
    df = read_eeg_file(file_path)
    
    if df.empty:
        print("No data was parsed from the file!")
        return
    
    # Create figure in fullscreen mode
    plt.rcParams['figure.figsize'] = [19.2, 10.8]
    fig = plt.figure()
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except:
        try:
            manager.window.showMaximized()
        except:
            try:
                manager.resize(*manager.window.maxsize())
            except:
                print("Could not maximize window")
    
    # Create subplots
    axs = fig.subplots(9, 1, sharex=True)
    fig.suptitle(f'EEG Data: {os.path.basename(file_path)}', fontsize=12)
    
    # Channels to plot
    channels = ['Pz', 'F4', 'C4', 'P4', 'P3', 'C3', 'F3']
    
    # Plot EEG channels
    for idx, channel in enumerate(channels):
        axs[idx].plot(df['TimeStamp'], df[channel])
        axs[idx].set_ylabel(f'{channel} (μV)')
        axs[idx].grid(True)
    
    # Plot Valence and Arousal
    axs[7].plot(df['TimeStamp'], df['Valence'], 'g-')
    axs[7].set_ylabel('Valence')
    axs[7].grid(True)
    
    axs[8].plot(df['TimeStamp'], df['Arousal'], 'r-')
    axs[8].set_ylabel('Arousal')
    axs[8].grid(True)
    
    axs[-1].set_xlabel('Time (s)')
    plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
    
    # Create buttons
    next_ax = plt.axes([0.25, 0.02, 0.1, 0.02])
    tag_bad_ax = plt.axes([0.45, 0.02, 0.1, 0.02])
    exit_ax = plt.axes([0.65, 0.02, 0.1, 0.02])
    
    next_button = Button(next_ax, 'Preserve')
    tag_bad_button = Button(tag_bad_ax, 'Tag as Bad')
    exit_button = Button(exit_ax, 'Exit App', color='red', hovercolor='darkred')
    
    def remove_bad_tag(filepath):
        """Remove [Bad] tag from filename if it exists"""
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        if filename.startswith('[Bad]'):
            new_filename = filename[5:]  # Remove first 5 characters ('[Bad]')
            new_path = os.path.join(directory, new_filename)
            os.rename(filepath, new_path)
            print(f"Removed [Bad] tag: {new_path}")
            return new_path
        return filepath

    def preserve_file(event):
        """Remove [Bad] tag if it exists and move to next file"""
        nonlocal file_path
        file_path = remove_bad_tag(file_path)
        plt.close()
    
    def tag_as_bad(event):
        """Add [Bad] tag if it doesn't exist"""
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if not filename.startswith('[Bad]'):
            new_path = os.path.join(directory, f'[Bad]{filename}')
            os.rename(file_path, new_path)
            print(f"File tagged as bad: {new_path}")
        plt.close()
            
    def exit_app(event):
        """Exit the entire application"""
        print("Exiting application...")
        plt.close('all')
        sys.exit(0)
    
    next_button.on_clicked(preserve_file)
    tag_bad_button.on_clicked(tag_as_bad)
    exit_button.on_clicked(exit_app)
    
    plt.show()

def main():
    # Create root window and hide it
    root = Tk()
    root.withdraw()
    
    # Ask user to select directory
    directory = filedialog.askdirectory(title="Select Directory with EEG Files")
    
    if not directory:
        print("No directory selected")
        return
    
    # Process each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            print(f"Processing: {filename}")
            plot_eeg_data(file_path)

if __name__ == "__main__":
    main()
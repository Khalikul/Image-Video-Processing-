import matplotlib.pyplot as plt
import numpy as np
import wave


# Function to show the sound waves and detect spikes
def visualize(path: str, spike_threshold: int = 2000):
    # reading the audio file
    raw = wave.open(path)

    # gets the frame rate
    f_rate = raw.getframerate()
    # gets the number of channels
    n_channels = raw.getnchannels()
    # gets the sample width
    samp_width = raw.getsampwidth()
    # gets the number of frames
    n_frames = raw.getnframes()

    print(f"Frame rate: {f_rate}")
    print(f"Channels: {n_channels}")
    print(f"Sample width: {samp_width} bytes")
    print(f"Number of frames: {n_frames}")

    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    # If stereo, taking only one channel
    if n_channels > 1:
        signal = signal[::n_channels]

    # Calculate the total duration of the audio
    total_duration = n_frames / f_rate
    print(f"Total audio duration: {total_duration:.2f} seconds")

    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = np.linspace(
        0,  # start
        total_duration,
        num=len(signal)
    )

    # Detecting spikes
    # spike_times = time[np.abs(signal) > spike_threshold]
    # for spike_time in spike_times:
    #     print(f"Spike detected at {spike_time:.2f} seconds")

    # using matplotlib to plot
    # creates a new figure
    plt.figure(1)

    # title of the plot
    plt.title("Sound Wave")

    # label of x-axis
    plt.xlabel("Time")

    # actual plotting
    plt.plot(time, signal)

    # shows the plot
    # in new window
    plt.show()


if __name__ == "__main__":
    # gets the command line Value
    visualize("path")

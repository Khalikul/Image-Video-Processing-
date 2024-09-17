import os
import cv2
import numpy as np
import pandas as pd
import wave
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tabulate import tabulate

# Function to extract audio from video and save as .wav
def extract_audio(video_path, audio_output_path):
    # Load video file (this extracts the audio)
    audio = AudioSegment.from_file(video_path, format="mp4")
    # Export audio as .wav file
    audio.export(audio_output_path, format="wav")
    print(f"Audio extracted and saved as '{audio_output_path}'")


# Function to detect non-silent segments in audio
def detect_audio_events(audio_path, min_silence_len=500, silence_thresh_delta=-14, padding=50):
    # Load audio file
    audio = AudioSegment.from_wav(audio_path)
    # Define silence threshold
    silence_thresh = audio.dBFS + silence_thresh_delta
    # Detect non-silent chunks in the audio
    non_silent_chunks = detect_nonsilent(audio, min_silence_len, silence_thresh, padding)

    # Convert audio events to timestamps
    audio_events = [(start / 1000.0, end / 1000.0) for start, end in non_silent_chunks]  # convert to seconds

    return audio_events


# Function to visualize sound waves
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


# Function to capture frames with motion
def FrameCapture(video_path, output_folder):
    # Initialize camera
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None

    # Frame rate and duration
    frame_rate = vidObj.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print("Error: Frame rate is zero. Check if video file is valid.")
        return None
    frame_duration = 1 / frame_rate

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize variables
    master = None
    status_list = [None, None]
    times = []
    frame_numbers = []
    frame_number = 0

    # Creation of dataframe for motion events
    motion_df = pd.DataFrame(columns=["Start", "End", "Start Frame", "End Frame"])

    motion_frame_count = 0

    while True:
        status = 0
        success, frame0 = vidObj.read()

        if not success:
            break

        current_time = frame_number * frame_duration
        frame1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.GaussianBlur(frame1, (15, 15), 0)

        if master is None:
            master = frame2
            frame_number += 1
            continue

        frame3 = cv2.absdiff(master, frame2)
        frame4 = cv2.threshold(frame3, 128, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((2, 2), np.uint8)
        frame5 = cv2.erode(frame4, kernel, iterations=4)
        frame5 = cv2.dilate(frame5, kernel, iterations=8)
        contours, _ = cv2.findContours(frame5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            status = 1

        master = frame2
        status_list.append(status)
        status_list = status_list[-2:]

        if status_list[-1] == 1 and status_list[-2] == 0:
            times.append(current_time)
            frame_numbers.append(frame_number)
            frame_filename = os.path.join(output_folder, f"motion_frame_{motion_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame0)
            motion_frame_count += 1
        if status_list[-1] == 0 and status_list[-2] == 1:
            times.append(current_time)
            frame_numbers.append(frame_number)

        frame_number += 1

    for i in range(0, len(times), 2):
        try:
            motion_df.loc[len(motion_df)] = [times[i], times[i + 1], frame_numbers[i], frame_numbers[i + 1]]
        except IndexError:
            end_time = frame_number * frame_duration
            motion_df.loc[len(motion_df)] = [times[i], end_time, frame_numbers[i], frame_number]

    vidObj.release()
    return motion_df


# Main Function
if __name__ == '__main__':
    video_file_path = ""
    audio_file_path = ""
    output_folder = ""

    # Extract audio from video
    extract_audio(video_file_path, audio_file_path)

    # Visualize the audio file
    # visualize(audio_file_path)

    # Capture frames with motion
    motion_df = FrameCapture(video_file_path, output_folder)
    if motion_df is not None:
        motion_df.to_csv("", index=False)
        print("Motion events saved to motion_events.csv")
        print("Motion Events:")
        print(tabulate(motion_df, headers="keys", tablefmt=""))

    # Detect audio events
    audio_events = detect_audio_events(audio_file_path)
    print("Detected Audio Events (in seconds):")
    for start, end in audio_events:
        print(f"Audio event from {start:.2f}s to {end:.2f}s")

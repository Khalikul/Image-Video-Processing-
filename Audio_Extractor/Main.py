from pydub import AudioSegment

# Path to the video file
video_file_path = "path"
# Path to save the extracted audio file
audio_file_path = "path"

# Load video file (this extracts the audio)
audio = AudioSegment.from_file(video_file_path, format="mp4")

# Export audio as .wav file
audio.export(audio_file_path, format="wav")

print(f"Audio extracted and saved as '{audio_file_path}'")

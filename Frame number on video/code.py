import cv2

def display_frame_numbers(video_path):
    '''Display_frame_numbers: Function to overlay frame numbers on the video and display it.
cv2.VideoCapture: Opens the video file.
video_capture.isOpened(): Checks if the video file was opened successfully.
frame_count: Counter to keep track of the frame number.
video_capture.read(): Reads the next frame from the video.
cv2.putText: Overlays the frame number text on the frame.
cv2.imshow: Displays the current frame with the frame number.
cv2.waitKey(1): Waits for 1 millisecond between frames and checks for the 'q' key to exit.
video_capture.release(): Releases the video capture object.
cv2.destroyAllWindows(): Closes all OpenCV windows.'''
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Overlay the frame number on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        line_type = 2
        cv2.putText(frame, f"Frame: {frame_count}", position, font, font_scale, font_color, line_type)

        # Display the frame
        cv2.imshow('Video with Frame Numbers', frame)

        # Increment the frame count
        frame_count += 1

        # Exit if the user presses the 'q' key
        #cv2.waitKey(1): Waits for 1 millisecond between frames and checks for the 'q' key to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    video_capture.release()
    cv2.destroyAllWindows()

# Usage
video_path = 'Zoom5.mp4'
display_frame_numbers(video_path)

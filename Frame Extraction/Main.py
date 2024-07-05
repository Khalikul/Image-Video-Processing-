import cv2

# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    if not vidObj.isOpened():
        print(f"Error: Could not open video '{path}'")
        return

    # Used as counter variable
    count = 0

    while True:
        # vidObj object calls read function extract frames
        success, image = vidObj.read()

        # Break the loop if no frame is returned
        if not success:
            break

        # Ensure the frame is not empty
        if image is not None:
            # Saves the frames with frame-count
            cv2.imwrite(f"frame{count}.jpg", image)
            count += 1
        else:
            print(f"Warning: Received an empty frame at count {count}")
            break

    print(f"Extracted {count} frames.")

# Driver Code
if __name__ == '__main__':
    # Calling the function
    FrameCapture("path")

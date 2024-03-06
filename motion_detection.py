import cv2

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Initialize the first frame (this will be used as a reference)
_, first_frame = cap.read()

while True:
    # Read a new frame from the camera
    _, frame = cap.read()

    # Convert frames to grayscale
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the first frame
    delta_frame = cv2.absdiff(gray_first, gray_frame)

    # Apply a threshold to the difference frame
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the threshold frame to fill in small holes
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust this threshold based on your scenario
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the original and motion-detected frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Motion Detection", thresh_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

import cv2 as cv

# cap = cv.VideoCapture("/Users/nika/Downloads/soccer.mp4")
CAMERA_INDEX = 0
cap = cv.VideoCapture(CAMERA_INDEX)
tracker = cv.TrackerCSRT_create()
BB = None

# Function to track the object
def track(frame):
    # update the tracker with the current frame
    (success, box) = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        cv.putText(frame, "Tracking failure detected", (20, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return success, frame

# Main loop for video capturing and processing
while True:
    # Set the timer before reading the frame
    timer = cv.getTickCount()
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if BB is not None:
        success, frame = track(frame)

    # Calculate frames per second (fps)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    # Display the fps on the frame
    cv.putText(frame, f"FPS: {int(fps)}", (20, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display the name of the tracker algorithm
    cv.putText(frame, f"CSRT Tracker", (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv.imshow("Frame", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        BB = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, BB)

    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

import cv2
import numpy as np
import time
import HandTracking as ht
import autopy
import threading

pTime = 0               # Used to calculate frame rate
width = 640             # Width of Camera
height = 480            # Height of Camera
frameR = 100            # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates

cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
cap.set(3, width)           # Adjusting size
cap.set(4, height)

# Set modelComplexity=0 for faster CPU performance
detector = ht.handDetector(maxHands=1, modelComplexity=0)
try:
    screen_width, screen_height = autopy.screen.size()      # Getting the screen size
except:
    # Fallback if screen size cannot be determined (e.g. in headless environments)
    screen_width, screen_height = 1920, 1080

# Shared variables for threading
img_copy = None
lmlist = []
fingers = []
lock = threading.Lock()
running = True

def detect_hands():
    global img_copy, lmlist, fingers, running
    while running:
        if img_copy is not None:
            # Work on a local copy to release the lock or avoid interference
            with lock:
                local_img = img_copy.copy()

            # We use the detector to find hands
            # These methods are called only from this thread to ensure consistency
            _ = detector.findHands(local_img, draw=False)
            temp_lmlist, _ = detector.findPosition(local_img, draw=False)

            temp_fingers = []
            if len(temp_lmlist) != 0:
                temp_fingers = detector.fingersUp()

            with lock:
                lmlist = temp_lmlist
                fingers = temp_fingers

            # Small sleep to prevent high CPU usage in this loop
            time.sleep(0.01)
        else:
            time.sleep(0.1)

# Start detection thread
t = threading.Thread(target=detect_hands, daemon=True)
t.start()

while True:
    success, img = cap.read()
    if not success:
        break

    # Update image for the detection thread
    with lock:
        img_copy = img.copy()

    # Get latest results from detection thread
    with lock:
        current_lmlist = lmlist.copy()
        current_fingers = fingers.copy()

    if len(current_lmlist) != 0:
        x1, y1 = current_lmlist[8][1:]
        x2, y2 = current_lmlist[12][1:]

        # Draw the boundary box
        cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)

        # 1. If fore finger is up and middle finger is down (Moving Mode)
        if len(current_fingers) >= 3 and current_fingers[1] == 1 and current_fingers[2] == 0:
            x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            try:
                autopy.mouse.move(screen_width - curr_x, curr_y)    # Moving the cursor
            except:
                pass
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        # 2. If fore finger & middle finger both are up (Clicking Mode)
        if len(current_fingers) >= 3 and current_fingers[1] == 1 and current_fingers[2] == 1:
            # Manually calculate distance to keep the main thread independent of detector's internal state
            length = np.hypot(x2 - x1, y2 - y1)

            if length < 40:     # If both fingers are really close to each other
                cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)
                try:
                    autopy.mouse.click()    # Perform Click
                except:
                    pass

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()

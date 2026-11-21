import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import cv2

print("Step 1: opening camera...")
cap = None
for name, backend in [("DirectShow", cv2.CAP_DSHOW),
                      ("Media Foundation", cv2.CAP_MSMF),
                      ("default", cv2.CAP_ANY)]:
    c = cv2.VideoCapture(0, backend)
    if c.isOpened():
        for _ in range(15):
            ok, frame = c.read()
            if ok and frame is not None:
                print(f"Step 2: got frames using {name}")
                cap = c
                break
            time.sleep(0.1)
    if cap is not None:
        break
    c.release()

if cap is None:
    print("No backend delivered frames. Close other apps using the camera.")
    raise SystemExit

print("Step 3: opening window... look for 'Camera Test' (press Q to quit)")
cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera Test", cv2.WND_PROP_TOPMOST, 1)  # keep on top
cv2.resizeWindow("Camera Test", 640, 480)
cv2.moveWindow("Camera Test", 50, 50)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
# /// script
# dependencies = ["opencv-python", "numpy"]
# ///
import sys
from pathlib import Path
import cv2 as cv

PATTERN = (7, 9)  # inner corners (cols, rows)
out = Path("shots")
out.mkdir(exist_ok=True)

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv.VideoCapture(index, cv.CAP_AVFOUNDATION)  # Linux: cv.CAP_V4L2 + FOURCC "MJPG"
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

n = len(list(out.glob("*.png")))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    # detect on half resolution so the preview stays fluid
    small = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
    found, corners = cv.findChessboardCorners(small, PATTERN, flags=cv.CALIB_CB_FAST_CHECK)
    view = frame.copy()
    if found:
        cv.drawChessboardCorners(view, PATTERN, corners * 2, found)
    color = (0, 255, 0) if found else (0, 0, 255)
    cv.putText(view, f"{n} saved", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv.imshow("calib", cv.resize(view, None, fx=0.5, fy=0.5))

    key = cv.waitKey(1) & 0xFF
    if key == ord(" ") and found:
        cv.imwrite(str(out / f"{n:03d}.png"), frame)  # raw frame, not the annotated one
        n += 1
    elif key in (ord("q"), 27):
        break

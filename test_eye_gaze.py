import cv2
from GazeTracking.gaze_tracking import GazeTracking
import plotly.express as px 
import keyboard
from screeninfo import get_monitors

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
gaze.calibration
w, h = get_monitors()[0].width, get_monitors()[0].height
x_left_pup_coord = [0]
y_left_pup_coord = [0]

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    new_frame = gaze.annotated_frame()
    
    if gaze.pupil_left_coords():
        x_left_pup_coord.append(gaze.pupil_left_coords()[1])
        y_left_pup_coord.append(gaze.pupil_left_coords()[0])
    cv2.imshow("Demo", new_frame)
    if keyboard.is_pressed('q'):
        break

x_left_pup_coord.append(w)
y_left_pup_coord.append(h)
fig = px.scatter( x=x_left_pup_coord, y=y_left_pup_coord)
fig.show()
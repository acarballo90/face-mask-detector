import cv2
import numpy as np


# Load Haar Cascade for facial detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('data/haarcascade_nose.xml')

# User messages
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
mask = 'WEARING FACE MASK'
mask_font_color = (0, 255, 0)
no_mask = 'NO FACE MASK'
no_mask_font_color = (0, 0, 255)
inc_mask = 'WEARING FACE MASK INCORRECTLY'
thickness = 2
font_scale = 0.75


# Read video
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    print('Camera not found')

while True:
    # Get frames
    ret, frames = cap.read()

    # Convert frames into gray
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    (thresh, bw) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Detect face contour
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(bw, 1.1, 4)

    # Detect mouth contour
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

    # Detect nose contour
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)

    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(frames, "Detecting face mask", org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(frames, mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw face contour
        for (x, y, w, h) in faces:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frames[y:y + h, x:x + w]
            cv2.putText(frames, "Face", (x, y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if(len(mouth_rects) == 0 and len(nose_rects) == 0):
            cv2.putText(frames, mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
        elif(len(mouth_rects) == 0 and len(nose_rects) == 1):
            # Draw nose contour
            for (nx, ny, nw, nh) in nose_rects:
                if (y < ny < y + h):
                    cv2.putText(frames, inc_mask, org, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    cv2.rectangle(frames, (nx, ny), (nx + nh, ny + nw), (0, 255, 0), 1)
                    cv2.putText(frames, "Nose", (nx, ny - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    break
        else:
            # Draw mouth contour
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    cv2.putText(frames, no_mask, org, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    cv2.rectangle(frames, (mx, my), (mx + mh, my + mw), (0, 0, 255), 1)
                    cv2.putText(frames, "Mouth", (mx, my - 5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Draw nose contour
            for (nx, ny, nw, nh) in nose_rects:
                if(y < ny < y + h):
                    cv2.putText(frames, no_mask, org, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    cv2.rectangle(frames, (nx, ny), (nx + nh, ny + nw), (0, 255, 0), 1)
                    cv2.putText(frames, "Nose", (nx, ny - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    break

    # Show frames
    cv2.imshow('Mask Detection', frames)

    # Save video
    out.write(frames)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
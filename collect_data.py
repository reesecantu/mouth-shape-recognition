import cv2

cap = cv2.VideoCapture(0)
is_recording = False
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame while collecting data.")
        continue
    
    image = cv2.flip(image, 1)
    key = cv2.waitKey(5) & 0xFF
    if key == ord(' '): 
        is_recording = not is_recording
    
    status_text = "RECORDING" if is_recording else "WAITING"
    text_color = (0, 0, 255) if is_recording else (0, 255, 0)

    cv2.putText(
    image,
    status_text,
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    3,
    text_color,
    5,
    cv2.LINE_AA
    )
    cv2.imshow('Data Collection', image)
    if key == 27:
        break
    
cap.release()
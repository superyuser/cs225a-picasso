import cv2

for idx in range(10):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        continue

    print(f"Showing camera index {idx}. Press any key to continue.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read from index {idx}")
            break

        cv2.putText(
            frame,
            f"Camera index: {idx}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv2.imshow("Camera Preview", frame)

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()
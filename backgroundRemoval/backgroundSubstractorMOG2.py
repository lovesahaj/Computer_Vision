import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

# MOG -> Mixture of Gaussians
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    success, image = cap.read()
    # image = cv2.resize(image, (int(16/9 * 480), 480))
    fgmask = fgbg.apply(image)
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

    fgmask = cv2.bitwise_and(image, fgmask)

    cv2.imshow("Image", image)
    cv2.imshow("fgmask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    os.system("cls")

cap.release()
cv2.destroyAllWindows()

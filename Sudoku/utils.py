import numpy as np
import cv2
import torch


def preProcess(image):
    mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(mono, (11, 11), 3)
    imgThreshold = cv2.adaptiveThreshold(
        src=imgBlur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return imgThreshold


def show_image(image):
    while True:
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def biggestContour(ctr):
    biggest = np.array([])
    max_area = 0

    for c in ctr:
        area = cv2.contourArea(c)

        if area > 500:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # corners

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

            if area > max_area and len(approx) == 8:
                biggest = []
                for x in approx:
                    if x[0][0] not in (0, 799) or x[0][1] not in (0, 799):
                        biggest.append(x)
                biggest = np.array(biggest)
                max_area = area

    return biggest, max_area


def reorder(ctr):
    ctr = ctr.reshape(4, 2)
    ctr_new = np.zeros((4, 1, 2), dtype=np.int32)

    add = ctr.sum(1)
    ctr_new[0] = ctr[np.argmin(add)]
    ctr_new[3] = ctr[np.argmax(add)]

    diff = np.diff(ctr, axis=1)
    ctr_new[1] = ctr[np.argmin(diff)]
    ctr_new[2] = ctr[np.argmax(diff)]

    return ctr_new


def save_checkpoint(state, filename="mycheckpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

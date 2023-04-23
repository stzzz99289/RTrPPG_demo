import cv2
import numpy as np

def putCentertext(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = int((image.shape[1] - textsize[0]) / 2)
    textY = int((image.shape[0] + textsize[1]) / 2)
    scale = 1
    thickness = 2
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def putBottomtext(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textX = 0
    textY = image.shape[0]
    scale = 1
    thickness = 2
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def plt2cv(fig, shape):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, shape)
    return img
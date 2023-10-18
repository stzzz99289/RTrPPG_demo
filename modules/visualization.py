import cv2
import numpy as np

def putCentertext(image, text):
    # text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = int((image.shape[1] - textsize[0]) / 2)
    textY = int((image.shape[0] + textsize[1]) / 2)
    scale = 1
    thickness = 2

    # put text on the image
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def putBottomtext(image, text):
    # text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    textX = 0
    textY = image.shape[0]
    scale = 1
    thickness = 2

    # put text on the image
    image = np.ascontiguousarray(image, dtype=np.uint8)
    cv2.putText(image, text, (textX, textY), font, scale, (0, 0, 255), thickness)
    return image


def plt2cv(fig, shape):
    # draw the figure on canvas and convert to cv2 image
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    # resize the image
    img = cv2.resize(img, shape)

    # slice to remove alpha channel
    img = img[:, :, :3]

    return img
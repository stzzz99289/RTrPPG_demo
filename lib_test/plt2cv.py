# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    x = [0, 1, 2, 3, 4, 5]
    y = [5, 10, 3, 9, 8, 7]

    fig = plt.figure()
    plt.plot(x, y, 'ko-')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("plot", img)
    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# RTrPPG demo

demo code for RTrPPG, by Tianze Shi, School of Computer Science, PKU.

## dependencies

- opencv
- mediapipe
- matplotlib
- numpy
- scipy

## run

`> python main.py [run_mode] [display_mode]`

### run mode

- `online`

    - requires a webcam

- `offline`

    - takes a pre-recorded video as input

### display mode

- `debug` (show more information in four images)

    - first image: original input video frames
    - second image: input video frames with ROI region visualization
    - third image: rPPG signals in the current time window
    - fourth image: vital signs history

- `demo` (show basic information in one image)

    - original input video frames with vital signs information

## todo list

- monitoring for multiple users

- more HRV metrics

- keep rPPG signals consistent (when moving time window, rPPG signal at the same time remains the same)

- reduce effects of face covering & head movement on rPPG signal (when hand moves across the face, there is a sudden change of rPPG signal)

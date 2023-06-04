# RTrPPG demo

demo code for RTrPPG, by Tianze Shi, School of Computer Science, PKU.

## dependencies

- see file `environment.yml`

## run

- online mode (requires a webcam)

    - `> python main.py online`

- offline mode (takes a pre-recorded video as input)

    - `> python main.py offline`

- results are not stable in the wild, for a more stable result, you can set `prior_hr`

    - `> python main.py offline 70`
    - in this example, `prior_hr` is set to 70 (which is the user's average hr) and we can get a more stable hr prediction result.
    - you can set `prior_hr` based on user's preference.

## todo list

- monitoring for multiple users

- more HRV metrics

- keep rPPG signals consistent (when moving time window, rPPG signal at the same time remains the same)

- reduce effects of face covering & head movement on rPPG signal (when hand moves across the face, there is a sudden change of rPPG signal)

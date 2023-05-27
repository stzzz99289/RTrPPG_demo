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

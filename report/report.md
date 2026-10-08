# Results: PID vs RL gains, layer by layer

This report compares two controllers for the Crazyflie 2.1 Brushless on each layer of the cascade. Each one runs as a
full cascade from that layer down to the motors:

| Controller | The layer itself | The layers below |
|---|---|---|
| **PID** | hand-tuned PID | hand-tuned PID |
| **RL gains** | network that outputs the 9 PID gains of the layer (`frozen/gains/<layer>.pt`) | frozen gain networks |

All numbers come from `src/Drone_RL/uav/tools/compare_layers.py`: headless, the same random seed for both
controllers, Isaac Sim 6.1 and a 1 kHz physics step. The raw values are in [`metrics/`](metrics).

## 1. Summary

![summary](figures/summary.png)

**Tracking error on random commands.** This is the training task of the layer: 64 drones, one full episode, with
random commands coming from the PID layers above plus random offsets. It is the hard test, because the command keeps
jumping. Lower is better.

| Layer (unit) | PID | RL gains | Change |
|---|---|---|---|
| rate (deg/s) | 37.8 | **24.6** | -35 % |
| attitude, tilt (deg) | 15.5 | **5.9** | -62 % |
| velocity (m/s) | 0.85 | **0.46** | -45 % |
| position (m) | 0.125 | **0.097** | -23 % |

**Motor chatter.** The mean change of the motor commands per control step, on the same task. Lower means smoother
motors (less heat, less vibration on a real drone).

| Layer | PID | RL gains |
|---|---|---|
| rate | **0.0063** | 0.0075 |
| attitude | **0.0021** | 0.0051 |
| velocity | **0.0129** | 0.0342 |
| position | **0.0032** | 0.0202 |

Every drone of both controllers was still flying at the end of the episode (no crash, no flip).

**In short:**

- **RL gains** has the lower error on every layer. It starts from the PID (zero output is the tuned PID) and only
  reshapes it, so it learns in a few hundred iterations and never produces an unstable controller. Its weak point is
  smoothness on the outer layers: on velocity and position it changes the gains fast enough to shake the motors 2.6×
  and 6.3× more than the PID. A stronger penalty on the change of the gains is the next thing to try.
- **PID** is the smoothest and the most predictable, but the slower of the two: on fast commands, and on yaw in
  particular, it lags the most.

## 2. Step responses

Each figure shows the mean of 8 drones following a fixed test signal: the wanted value (dashed), the two controllers,
and below them the command of motor 1, to show how hard each one works the motors. The mean |error| in the legends is
over the whole test.

### 2.1 Rate layer (500 Hz): body rate steps of 30 deg/s (roll, pitch) and 90 deg/s (yaw)

![rate](figures/compare_rate.png)

| mean \|error\| (deg/s) | roll | pitch | yaw |
|---|---|---|---|
| PID | 0.24 | 0.25 | 6.49 |
| RL gains | **0.23** | **0.25** | **3.37** |

The gain network matches the PID on roll and pitch, which were already well tuned, and **halves the yaw error** by
raising the yaw gain.

### 2.2 Attitude layer (250 Hz): roll and pitch steps of 11.5 deg, a yaw step of 46 deg

![attitude](figures/compare_attitude.png)

| mean \|error\| (deg) | roll | pitch | yaw |
|---|---|---|---|
| PID | 0.47 | 0.47 | 4.73 |
| RL gains | **0.20** | **0.16** | **3.68** |

RL gains reaches the wanted angle about twice as fast as the PID, with no overshoot, and keeps the motors smooth.

### 2.3 Velocity layer (100 Hz): steps of 1 m/s in x, then in y, 0.5 m/s up, then a diagonal

![velocity](figures/compare_velocity.png)

| mean \|error\| (m/s) | vx | vy | vz |
|---|---|---|---|
| PID | 0.102 | 0.099 | 0.016 |
| RL gains | **0.100** | **0.093** | **0.007** |

The PID overshoots every step by about 25 % and rings. The gain network removes the overshoot and halves the height
error while the drone accelerates sideways, but it reaches the new speed more slowly. On clean steps the two have the
same mean error; on the random commands of the training task the gain network halves it (section 1).

### 2.4 Position layer (50 Hz): target jumps of 0.5 m, a 0.3 m climb, then a circle of 0.5 m radius at 0.5 m/s

![position](figures/compare_position.png)

| mean \|error\| (m) | x | y | z |
|---|---|---|---|
| PID | 0.119 | 0.129 | **0.012** |
| RL gains | **0.117** | **0.113** | 0.014 |

The PID overshoots the y jump by 0.2 m and rings for 2 s; the gain network settles without overshoot and follows the
circle with less lag. Both start 0.25 m away from the first target, which adds the same error to both.

## 3. Training curves

![training](figures/training.png)

One row per layer: mean episode reward, tracking error (`Metrics/layer/error`, log scale, in the units of the tables
above) and the action noise of the policy (its std, which shrinks as the policy settles). The dashed line is the
tracking error of the tuned PID on the same task. A resumed run is joined to the run it continues; the first iterations
after the restart, whose statistics only count the short episodes that end first, are left out (the small gap).

| Layer | Iterations |
|---|---|
| rate | 1500 |
| attitude | 1000 |
| velocity | 1400 |
| position | 1000 |

The gain cascade starts from the PID (above the PID line at first only because of the exploration noise) and goes below
the PID line within about 100 iterations on every layer.

## 4. How to reproduce

```bash
python src/Drone_RL/uav/tools/compare_layers.py --layer rate        # then attitude, velocity, position
python src/Drone_RL/uav/tools/report_figures.py
```

`compare_layers.py` skips the gains when a frozen network is missing and marks it in the json, so the report can be
rebuilt at any stage of training. The training curves are read from `logs/rsl_rl/`.

## 5. Limits of this comparison

- Simulation only. The controllers see the true state, with no estimator, no motor lag and no wind.
- One seed and one set of trained networks. The random-command test uses 64 drones over a full episode; the step tests
  use 8.
- The PID gains were tuned by hand, once. A better-tuned PID would narrow the gap; the gain network shows how much
  each gain could change and in which direction.

<p align="center">
  <img src="guide/media/banner.png" alt="Crazyflie 2.1 Brushless in Isaac Sim" width="720">
</p>

# Drone_RL

Reinforcement learning and classical control for the **Crazyflie 2.1 Brushless** in **Isaac Lab 3.0**.
A cascaded PID (position, velocity, attitude, rate) and the same four layers learned with PPO, one network per layer,
trained bottom-up and frozen, small enough to run on the drone's STM32 (no ROS in this repo); plus a landing task on an
ArUco marker seen by a downward camera. Physical constants come from the datasheet, papers and Bitcraze firmware.

<p align="center">
  <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID"><img src="guide/media/demo_pid_hover.gif" alt="Demo video" width="360"></a>
  <img src="guide/media/demo_figure8.gif" alt="Kinematic figure-8" width="360">
</p>

> **Demo video:** https://www.youtube.com/watch?v=YOUR_VIDEO_ID

| | |
|---|---|
| Robot | Crazyflie 2.1 Brushless, 34 g, 100 mm frame, 0.2 N max thrust per motor |
| Simulator | Isaac Sim 6.1 + Isaac Lab 3.0 |
| RL | rsl_rl 5.5.1 (PPO): `Isaac-UAV-{Rate,Attitude,Velocity,Position}-RL-v0`, `Isaac-UAV-Landing-ArUco-v0` |
| Extras | cascaded PID baseline, kinematic trajectory tool |

Documentation: [guide/readme.md](guide/readme.md) (seven pages, read in order) · [Tiếng Việt](README.vi.md)

## 1. Requirements

- Ubuntu (x86_64), NVIDIA GPU with a recent driver (developed on an RTX 3060, 12 GB)
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), Python 3.12
- `git` and [Git LFS](https://git-lfs.com/) (the `.usd` and `.pt` files are stored in LFS)

The first Isaac Sim launch asks you to accept the NVIDIA EULA and downloads extensions, so it can take a few minutes.

## 2. Install, step by step

### 2.1 Clone

```bash
sudo apt install git-lfs && git lfs install      # once per machine
mkdir -p ~/Documents/GitHub && cd ~/Documents/GitHub
git clone https://github.com/vohongquann/Drone_RL.git
cd Drone_RL && git lfs pull
```

### 2.2 Install Isaac Sim and Isaac Lab (once)

This project expects Isaac Lab as a sibling folder (`../IsaacLab`, see `[tool.uv.sources]` in `pyproject.toml`).

```bash
cd ~/Documents/GitHub
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
conda create -n env_isaaclab python=3.12 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
./isaaclab.sh -i 'isaacsim,rl[rsl-rl]'      # Isaac Sim 6.1 + Isaac Lab + rsl_rl
```

The `isaaclab` command does not exist before this step: `./isaaclab.sh -i` installs the Isaac Lab packages into the
active conda environment, and one of them provides the `isaaclab` console script. Check it with
`which isaaclab && isaaclab --help`. See the [official installation guide](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)
if a step fails. The `uv` setup declared in `pyproject.toml` is not what this project was developed with.

### 2.3 Install this project

```bash
conda activate env_isaaclab
cd ~/Documents/GitHub/Drone_RL
pip install -e . --no-deps
pip install pytest                    # test dependency
python -c "import Drone_RL.uav; print('ok')"
```

The last line must print `ok`. Every later command must run in the terminal where `env_isaaclab` is active
and from the `Drone_RL` folder.

### 2.4 Run the tests (no simulator window)

```bash
python -m pytest tests -q
```

## 3. Quick start

Fly the classical PID (open a window with `--viz kit`):

```bash
python src/Drone_RL/uav/pid_control/pid_hover_test.py --scenario hover     # hover | square
```

Follow a known trajectory kinematically and check the motors could fly it:

```bash
python src/Drone_RL/uav/pid_control/run_kinematic.py --pattern figure8   # hover | circle | figure8
```

Train the RL layers bottom-up, each on top of the frozen ones below (video clips in
`logs/rsl_rl/uav_<layer>/<run>/videos/train/`):

```bash
isaaclab train --rl_library rsl_rl --task Isaac-UAV-Rate-RL-v0 --num_envs 1024 --video --video_length 400 --video_interval 5000
isaaclab play --rl_library rsl_rl --task Isaac-UAV-Rate-RL-v0 --num_envs 1
cp logs/rsl_rl/uav_rate/<run>/exported/policy.pt src/Drone_RL/uav/rl_control/frozen/rate.pt   # then Attitude, Velocity, Position the same way
```

Details: [guide/05_rl_cascade.md](guide/05_rl_cascade.md).

## 4. How it works

Read [guide/readme.md](guide/readme.md), then the pages in order:

1. [The drone](guide/01_drone.md): every number and its source
2. [Propulsion](guide/02_propulsion.md): motor curve, mixer, force on the body
3. [Kinematic flight](guide/03_kinematic_flight.md): can the motors fly a path?
4. [PID cascade](guide/04_pid_cascade.md): position, velocity, attitude, rate
5. [RL cascade](guide/05_rl_cascade.md): the same layers learned and frozen one by one
6. [ArUco landing](guide/06_aruco_landing.md): camera, marker detection, landing policy
7. [Một bước mô phỏng, từ PPO đến PhysX](guide/07_simulation_step.md): cái gì chạy theo thứ tự nào, và Isaac Lab điều khiển PhysX ra sao

```
src/Drone_RL/
  assets/data/crazyflie/cf2x.usd   drone model (Isaac Sim Crazyflie 2.x)
  uav/uav_cfg.py                   every physical constant, with its source
  uav/mdp/                         shared MDP terms: actions (MotorAction, CascadeAction, propulsion, mixer),
                                   RL layers, commands, observations, rewards, terminations
  uav/pid_control/                 cascaded PID, layer tests, kinematic trajectory tool
  uav/rl_control/                  every RL task, one file each (rate, attitude, velocity, position, landing),
                                   PPO in agents/, frozen/ weights
tests/                             constants, mixer, PID, RL layers, ArUco (no simulator)
```

## 5. Known limitations

Not found in any published source, so these are assumptions: propeller mass, in-flight IMU vibration noise. The
simulation has no motor lag: a motor gives the thrust of its PWM at once. The
collision geometry of the USD is the Crazyflie 2.x one.

## 6. Development

```bash
python -m pytest tests -q
```

## Acknowledgements

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). Parameters from the Crazyflie 2.1 Brushless datasheet,
[Bitcraze firmware](https://github.com/bitcraze/crazyflie-firmware), Busetto et al. (arXiv:2512.14450), Folk et al.
(arXiv:2604.00343) and the Bosch BMI088 datasheet.

## License

See [LICENSE](LICENSE).

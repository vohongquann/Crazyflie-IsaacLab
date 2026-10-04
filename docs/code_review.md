# Review toàn bộ code Drone_RL, từ dễ đến khó

File này là lộ trình để tự review toàn bộ hệ thống. Đi theo thứ tự: mỗi cấp chỉ dùng những gì đã kiểm tra ở cấp trước,
nên khi tới cấp khó (một bước mô phỏng có frozen layers chạy lồng nhau) bạn đã chắc chắn mọi mảnh bên dưới đúng.

- Mỗi mục có: **file / hàm / class**, **đọc để hiểu gì**, **kiểm tra gì** (checkbox), **bẫy dễ sai**, và khi có thể
  một **lệnh kiểm tra nhanh** (vài giây, không chạy simulator, không train).
- Số dòng ghi theo code ngày 2026-09-27; nếu lệch vài dòng thì tìm theo tên hàm.
- Tài liệu song song: `guide/01..07`. Cột "Guide" ở mỗi cấp nói nên mở trang nào bên cạnh.

Tất cả lệnh Python dưới đây chạy bằng:

```bash
PY=~/miniconda3/envs/env_isaaclab/bin/python
cd ~/Documents/GitHub/Drone_RL
```

---

## Mục lục

| Cấp | Nội dung | Độ khó | Cần Isaac? | Guide |
|---|---|---|---|---|
| 0 | Chuẩn bị: quy ước, bản đồ phụ thuộc, trạng thái hiện tại | ★ | không | readme |
| 1 | Hằng số vật lý, mixer, PID cơ bản | ★ | không | 01, 02 |
| 2 | Mô hình lực đẩy: PWM → lực → trễ động cơ → wrench | ★★ | một phần | 02 |
| 3 | PID cascade và bay động học | ★★ | một phần | 03, 04 |
| 4 | Bốn lớp RL thuần torch (`layers.py`) | ★★★ | không | 05 |
| 5 | Đông cứng một lớp: `FrozenPolicy`, `freeze.py` | ★★★ | không | 05 |
| 6 | Các term MDP chạy trong Isaac Lab: action, command, observation, reward, termination | ★★★★ | có | 05, 07 |
| 7 | Cấu hình task và PPO | ★★★ | có | 05 |
| 8 | Hạ cánh ArUco | ★★★ | có | 06 |
| 9 | Toàn hệ thống theo thời gian: một bước mô phỏng | ★★★★★ | có | 07 |
| 10 | Test: cái gì đã được bảo vệ, cái gì chưa | ★★ | không | — |
| Phụ lục | File ngoài luồng, vấn đề mở, bảng theo dõi | — | — | — |

---

## Cấp 0 — Chuẩn bị

### 0.1 Chạy test trước khi đọc

```bash
$PY -m pytest tests -q          # kỳ vọng: 21 passed
```

Nếu không pass, dừng lại sửa trước: mọi thứ bên dưới giả định các test này xanh.

### 0.2 Quy ước dùng khắp nơi (nhớ trước khi đọc bất kỳ file nào)

| Quy ước | Giá trị | Kiểm ở đâu |
|---|---|---|
| Khung thân (body) | x trước, y trái, z lên | docstring `uav_cfg.py` |
| Khung thế giới (world) | z lên, trọng lực −z | — |
| Quaternion | **(x, y, z, w)** (Isaac Lab 3.0) | `kinematics.quat_to_rotmat`, `flight_test.euler_from_quat`, camera landing |
| Ma trận quay `R` | body → world, nên `R[:, :, 2]` = trục z thân nhìn từ world; `R[:, 2, :]` = trục z world nhìn từ thân | `layers.py` |
| `einsum("nji,nj->ni", R, v)` | = `Rᵀ v` (world → body) | `layers.py:120`, `attitude.py` |
| Dữ liệu robot | `robot.data.<x>` là `ProxyArray`, phải `.torch` | `FlightState.of` |
| Đơn vị lệnh động cơ | [0, 1] ↔ PWM [0, 65535] | `motor_action.py`, `cascade_action.py` |
| `CF_THRUST_COEF_G` | lực đẩy **tổng 4 động cơ** tính bằng gram | `uav_cfg.py:105`, chia 4 trong `pwm_to_thrust` |
| Physics | 500 Hz (`FW_TICK_HZ`), dt = 2 ms | `mdp/actions/constants.py` |
| Tần số policy | rate 100 Hz (decimation 5), attitude/velocity/position 50 Hz (decimation 10), landing 50 Hz | `layers.py`, `cascade_env_cfg.py:109` |
| Trọng số reward | Isaac Lab nhân với `step_dt` | `cascade_env_cfg.termination_penalty` |

### 0.3 Bản đồ phụ thuộc (ai import ai)

```text
uav_cfg.py  ──────────────┬──────────────┬───────────────────────────┐
  (hằng số, UAV_CFG)      │              │                           │
                          ▼              ▼                           ▼
mdp/actions/constants.py  mdp/actions/mixer.py   mdp/actions/propulsion.py (pwm_to_thrust, thrust_to_pwm,
                          │                      │                           motor_lag_step, Propulsion)
                          ▼                      ▼
pid_control/pid.py ──► rate/attitude/velocity/position.py ──► pid_control/cascade.py (CascadePID)
                                                                   │ (import lười trong LayerCommand.__init__)
mdp/layers.py (FlightState, 4 Layer, layers_below) ◄── thrust_to_pwm │
     │                                                             ▼
     ├──► mdp/actions/frozen_policy.py (FrozenPolicy, load_frozen)   mdp/commands.py (LayerCommand)
     │         │                                                   │
     ├──► mdp/actions/cascade_action.py (History, FrozenLayer, CascadeAction) ◄── motor_action.MotorActionCfg
     ├──► mdp/observations.py (layer_observation, ActionHistory, ArucoObservation ◄── mdp/aruco.py)
     ├──► mdp/rewards.py ◄── mdp/terminations.py
     ▼
rl_control/freeze.py (FROZEN_DIR) ◄── rl_control/cascade_env_cfg.py ◄── rate/attitude/velocity/position_env_cfg.py
rl_control/agents/*_ppo_cfg.py        rl_control/landing_env_cfg.py ◄── marker_plate.py
rl_control/__init__.py (gym.register 5 task) ◄── uav/__init__.py ◄── tasks/__init__.py (Isaac Lab tìm task ở đây)
```

Đọc theo chiều mũi tên từ trên xuống = đúng thứ tự của file này.

### 0.4 Trạng thái hiện tại (quan trọng khi review)

- **Cả 4 file `rl_control/frozen/*.pt` đều cũ (stale)** so với code hiện tại:
  - `attitude.pt`, `velocity.pt`, `position.pt`: train trước khi thêm thrust feedforward cho attitude.
  - `rate.pt`: train với offset lệnh rate hẹp (1.5, 1.5, 1 rad/s; 0.03 N). Hôm nay offset đã mở rộng tới
    (6, 6, 3) rad/s và 0.167 N, nên phải train lại từ rate.
  - Kích thước obs/action không đổi, nên `load_frozen` **không** báo lỗi: file cũ vẫn load được, chỉ là hành vi sai.
    Đây là bẫy lớn nhất của hệ thống: kiểm tra kích thước không bắt được thay đổi ngữ nghĩa.
- Những gì đã sửa hôm nay sau review (review lại các chỗ này trước cũng được):

| # | Sửa | File |
|---|---|---|
| 1 | Offset lệnh của rate task = giới hạn đầu ra của attitude layer | `rl_control/rate_env_cfg.py:41-43` |
| 2 | Metric `error` = trung bình cả episode (trước là giá trị bước cuối) | `mdp/commands.py` `reset`, `_update_metrics` |
| 3 | Guide 05, 06, 07 khớp lại với code | `guide/` |
| 4 | `SPIN_VISUAL_SCALE` 1.0 → 0.06; cascade task tắt quay cánh | `propulsion.py:58`, `cascade_action.py:161` |
| 5 | Landing bắt đầu episode ở lực hover | `motor_action.py:64-69` |
| 6 | `ActionHistory` chỉ đẩy 1 lần mỗi bước | `observations.py:65-72` |
| 7 | 3 test mới (khung nghiêng, khớp train/frozen, lịch tick) | `tests/test_rl_layers.py:64-148` |
| 8 | Lệnh được tính ngay khi reset toàn bộ (trước đó quan sát đầu tiên thấy lệnh 0) | `mdp/commands.py` `reset` |

---

## Cấp 1 — Hằng số vật lý, mixer, PID cơ bản (★)

Mục tiêu: mọi con số và mọi dấu (+/−) ở tầng thấp nhất đúng. Sai ở đây làm hỏng mọi tầng trên mà không báo lỗi.

### 1.1 `src/Drone_RL/uav/uav_cfg.py` (194 dòng)

Đọc cùng `guide/01_drone.md` (nguồn từng số) và `guide/02_propulsion.md` (công thức).

| Dòng | Tên | Kiểm tra |
|---|---|---|
| 25 | `CF_MOTOR_HALF_XY_M` | = 0.050/√2 = 0.03536 m (100 mm động cơ-động cơ theo đường chéo → 50 mm từ tâm → chiếu lên trục) |
| 29-34 | `CF_MOTOR_XY` | Thứ tự m1..m4 phải khớp joint trong `cf2x.usd`: m1 (+x, −y) trước-phải, m2 (−x, −y) sau-phải, m3 (−x, +y) sau-trái, m4 (+x, +y) trước-trái |
| 37 | `CF_MOTOR_SPIN` | (−1, +1, −1, +1): hai động cơ chéo nhau cùng chiều; +1 = quay thuận chiều kim đồng hồ nhìn từ trên → mô-men phản lực +z |
| 41 | `DRONE_MASS_TOTAL_KG` | 0.034 (datasheet, có chân, pin 350 mAh, không deck) |
| 45, 48 | `DRONE_MASS_FAN_KG`, `DRONE_MASS_GROUP1_KG` | thân = tổng − 4 cánh; cánh 0.8 g là giả định (giá trị USD) |
| 51 | `DRONE_INERTIA_DIAG` | Busetto 45 g, scale tuyến tính theo khối lượng → (1.81e-5, 1.81e-5, 2.44e-5) kg·m² |
| 54 | `DRONE_HOVER_THRUST_N` | m·g = 0.3335 N |
| 62-72 | `CF_K_ETA`, `CF_KV`, `CF_V0`, `CF_DZ`, `CF_PWM_MAX`, `CF_BATTERY_V_NOM` | so với Folk (arXiv:2604.00343, mục 6.3.1) |
| 75 | `DRONE_KM` | kM/kF = 7.73e-11/3.72e-8 = 2.08e-3 m |
| 79 | `CF_KD_DRAG` | Folk bảng 6.2 |
| 83-86 | `motor_thrust_n()` | η = KV·(V+V0)·(PWM−DZ)^(2/3), T = k_η·η²; `.clip(0.0)` dưới vùng chết |
| 89 | `CF_F_MAX_N` | 0.232 N mỗi động cơ (ngoại suy trên PWM 45000) |
| 94-105 | `_fit_thrust_curve()`, `CF_THRUST_COEF_G` | fit bậc 2 cho **tổng 4 động cơ** bằng gram; chú ý dùng g = 9.80665 khi đổi N→g |
| 109-115 | `_hover_throttle()`, `DRONE_HOVER_THROTTLE` | giải a·p²+b·p+c = 34 g → 0.502 |
| 118 | `CF_DRAG_COEF` | kd · Σ η_hover = 0.0292 N·s/m |
| 123 | `CF_MOTOR_JOINTS` | tên joint, chỉ để hiển thị |
| 127-148 | `apply_real_mass_inertia()` | chỉ sửa rigid body tên `body`; phải gọi trước khi PhysX cook |
| 151-158 | `_spawn_uav_with_nominal_mass()` | `spawn_from_usd.__wrapped__` rồi ghi khối lượng |
| 161-194 | `UAV_CFG` | `solver_velocity_iteration_count=0`, gyroscopic on, actuator "dummy" stiffness = damping = 0 (cánh không bị lái) |

Checklist:
- [ ] Mọi số khớp nguồn trong `guide/01_drone.md`.
- [ ] Thứ tự và dấu `CF_MOTOR_XY` / `CF_MOTOR_SPIN` khớp nhau và khớp USD.
- [ ] Hiểu vì sao `CF_THRUST_COEF_G` là **tổng**: một đường cong cho từng động cơ làm hover throttle bão hòa ở 1.0.
- [ ] Tỉ lệ lực đẩy tối đa / trọng lượng ≈ 2.78 (4·0.232/0.3335): hợp lý cho Crazyflie brushless.

Lệnh kiểm tra nhanh:

```bash
$PY -c "
from Drone_RL.uav import uav_cfg as U
print('hover throttle', U.DRONE_HOVER_THROTTLE)      # ~0.502
print('F max / motor', U.CF_F_MAX_N)                 # ~0.232 N
print('KM', U.DRONE_KM, 'drag', U.CF_DRAG_COEF)      # 2.08e-3, 0.0292
print('inertia', U.DRONE_INERTIA_DIAG)               # (1.81e-5, 1.81e-5, 2.44e-5)
print('T/W max', 4*U.CF_F_MAX_N/U.DRONE_HOVER_THRUST_N)   # ~2.78
"
```

Bẫy:
- `9.81` dùng khắp nơi nhưng `_fit_thrust_curve` dùng `9.80665`: sai lệch 0.03 %, không quan trọng nhưng nên biết.
- Cuối file có docstring lạc `"""Physics rate and motor lag constants shared by the action terms."""` sau `UAV_CFG` (dòng 195): chỉ là chuỗi thừa, vô hại.

### 1.2 `src/Drone_RL/uav/mdp/actions/constants.py` (16 dòng)

- `FW_TICK_HZ = 500.0`: tần số physics (ATTITUDE_RATE của firmware).
- `MOTOR_TAU_INC_RANGE = (0.05, 0.08)`, `MOTOR_TAU_DEC_RANGE = (0.005, 0.005)`: hằng số trễ tăng/giảm tốc.

Checklist:
- [ ] Hiểu đây là **giả định** (lấy từ robot ARL của Isaac Lab), chưa có số đo cho Crazyflie brushless.
- [ ] Trễ giảm tốc (5 ms) nhanh hơn nhiều so với tăng tốc (50–80 ms): động cơ "thả" nhanh, "đẩy" chậm.

### 1.3 `src/Drone_RL/uav/mdp/actions/mixer.py` (34 dòng)

| Hàm | Làm gì |
|---|---|
| `wrench_matrix()` (dòng 20) | ma trận 4×4: [F1..F4] → [T, τx, τy, τz]; hàng: 1, y_i, −x_i, KM·s_i |
| `force_allocation_inverse()` (dòng 32) | nghịch đảo (float64 rồi `.float()`) |

Checklist:
- [ ] τx = Σ F_i·y_i: động cơ bên trái (y > 0) đẩy bên trái lên → roll dương. Đúng quy tắc tay phải quanh x.
- [ ] τy = −Σ F_i·x_i: động cơ phía trước (x > 0) đẩy mũi lên → pitch âm (quanh y, mũi lên là pitch âm).
- [ ] τz = KM·Σ s_i·F_i.
- [ ] Cùng quy ước với `Propulsion.step` bước 5 (sẽ kiểm ở cấp 2).

Lệnh: `$PY -m pytest tests/test_mixer.py -q`. Ma trận mong đợi:

```text
[[ 1.0000,  1.0000,  1.0000,  1.0000],
 [-0.0354, -0.0354,  0.0354,  0.0354],     y_i
 [-0.0354,  0.0354,  0.0354, -0.0354],     -x_i
 [-0.0021,  0.0021, -0.0021,  0.0021]]     KM * s_i
```

### 1.4 `src/Drone_RL/uav/pid_control/pid.py` (45 dòng) — class `PID`

- `__init__(kp, ki, kd, out_limit, int_limit)`: gain có thể là tensor broadcast theo trục.
- `reset(env_ids)`: `None` → xóa hẳn trạng thái (lần `update` sau khởi tạo lại); có `env_ids` → đặt 0 các hàng đó.
- `update(error, dt)`: I tích lũy rồi kẹp (anti-windup), D = (e − e_prev)/dt, lần gọi đầu không có "cú đá" D,
  kẹp đầu ra.

Checklist:
- [ ] Anti-windup kẹp **tích phân**, không phải `ki·tích phân`: `int_limit` có đơn vị của ∫e dt.
- [ ] Sau `reset(env_ids)` từng phần, `prev_error` các hàng đó = 0 → lần sau có cú đá D nhỏ = e/dt·kd (chỉ rate PID
  có kd). Chấp nhận được nhưng nên biết.
- [ ] Không có kẹp khi `out_limit=None`.

Lệnh: `$PY -m pytest tests/test_pid.py -q`.

---

## Cấp 2 — Mô hình lực đẩy (★★)

Mục tiêu: một lệnh động cơ trong [0, 1] biến thành đúng lực và mô-men trên thân. Đây là **con đường duy nhất** để lực
tới drone: PID, RL cascade và landing đều kết thúc ở `Propulsion.step`.

Guide: `guide/02_propulsion.md` (mục 2.3 là công thức từng bước).

### 2.1 `src/Drone_RL/uav/mdp/actions/propulsion.py` — các hàm thuần

| Dòng | Hàm | Kiểm tra |
|---|---|---|
| 15-22 | `pwm_to_thrust(pwm, a, b, c, g, f_max)` | gram tổng → /4 → /1000 → ·g; kẹp [0, f_max] |
| 25-30 | `thrust_to_pwm(force, a, b, c, g, pwm_max)` | nghịch đảo: ·4·1000/g rồi giải bậc 2, lấy nghiệm (−b + √Δ)/2a; Δ kẹp ≥ 0; kẹp [0, pwm_max] |
| 33-55 | `motor_lag_step(thrust, thrust_cmd, tau_inc, tau_dec, dt)` | trễ trên **tốc độ rotor** η ~ √T; chọn τ_dec khi đang giảm; hệ số 1/(dt+τ); RK4; trả η² |
| 58 | `SPIN_VISUAL_SCALE = 0.06` | 1435 rad/s thật → 86 rad/s trên hình |
| 62-74 | `spin_propellers(robot, force, spin, scale)` | chỉ ghi vận tốc joint (ghi cả góc làm cánh đứng hình) |

Checklist:
- [ ] `thrust_to_pwm(pwm_to_thrust(p)) == p` trong vùng [DZ, PWM_MAX].
- [ ] Lực mỗi động cơ âm (khi PID đòi) → Δ kẹp → PWM tối thiểu, không NaN.
- [ ] `motor_lag_step`: điều kiện `slowing = (thrust > 0) & (thrust_cmd < thrust)`; khi thrust = 0 luôn dùng τ_inc.
- [ ] Công thức 1/(dt+τ) (không phải 1/τ) giữ ổn định khi τ → 0.
- [ ] RK4 trên phương trình dη/dt = (η_cmd − η)/(dt+τ): k2..k4 giảm sai số dần như trong comment.

Lệnh:

```bash
$PY -c "
import torch
from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.propulsion import pwm_to_thrust, thrust_to_pwm, motor_lag_step
a,b,c = U.CF_THRUST_COEF_G
f = torch.tensor([[0.02, 0.05, 0.0834, 0.2]])
p = thrust_to_pwm(f, a, b, c, 9.81, U.CF_PWM_MAX); print(p)                 # ~[13960, 23596, 32916, 59037]
print(pwm_to_thrust(p, a, b, c, 9.81, U.CF_F_MAX_N))                         # lại đúng f
F = torch.zeros(1,4); cmd = torch.full((1,4), 0.0834); tau = torch.full((1,4), 0.05)
for k in range(100): F = motor_lag_step(F, cmd, tau, tau*0.1, 0.002)
print(F)                                                                     # gần 0.0834 sau 0.2 s
"
```

### 2.2 `Propulsion` (dòng 77-176)

| Phương thức | Kiểm tra |
|---|---|
| `__init__` | `force` (N,4) bắt đầu 0; `tau_inc`, `tau_dec`, `gain`, `drag` rút ngẫu nhiên qua `resample` |
| `resample(env_ids)` (113) | τ theo **từng động cơ, từng episode**; `gain` ∈ [0.9, 1.1] nếu `use_motor_asymmetry`; `drag` = CF_DRAG_COEF·U(0.5, 1.5) |
| `reset(env_ids)` (130) | lực về 0 + rút lại tham số (các action term ghi đè lực = hover sau đó) |
| `step(pwm, robot, body_id, dt)` (135) | 5 bước bên dưới |

`step` từng bước:
1. `f_cmd = pwm_to_thrust(pwm) * gain`.
2. Trễ động cơ (hoặc bỏ qua nếu `use_motor_lag=False`).
3. Nhiễu (tắt mặc định). Chú ý: nhiễu áp vào biến tạm `f`, **không** vào `self.force` → trạng thái trễ không bị nhiễu.
4. Lực thân: [0, 0, ΣF] − c_d·v_body (lực cản tuyến tính, cả 3 trục).
5. Mô-men: (ΣF·y, −ΣF·x, KM·Σs·F) — **cùng quy ước với `mixer.wrench_matrix`**.
6. `permanent_wrench_composer.set_forces_and_torques` (khung thân, tại tâm khối lượng; giữ đến khi bị ghi đè).
7. Quay cánh nếu `cfg.spin_propellers`.

Checklist:
- [ ] Bước 5 khớp từng hàng của `wrench_matrix` (so dấu).
- [ ] Lực cản dùng `root_lin_vel_b` (khung thân) vì lực được áp trong khung thân. Đúng.
- [ ] `_force_buf` có shape (N, 1, 3) (một body); `force = self._force_buf[:, 0]` là **view** nên ghi vào `force` là ghi vào buffer.
- [ ] Mô-men phản lực z chỉ phụ thuộc lực (KM·F), không có trễ riêng.
- [ ] Quay cánh: tắt trong cascade task (`CascadeActionCfg.spin_propellers=False`), bật trong landing và script PID.

### 2.3 `src/Drone_RL/uav/mdp/actions/motor.py` (119 dòng)

Script vẽ đường cong lực đẩy (`guide/media/motor_thrust.png`), **không được package import**. Chỉ cần kiểm:
- [ ] `simulated_thrust_n` dùng đúng `pwm_to_thrust` như mô phỏng (đường fit) để so với mô hình Folk nhiều điện áp.

### 2.4 `src/Drone_RL/uav/mdp/actions/motor_action.py` — `MotorAction`, `MotorActionCfg`

Dùng bởi: landing (action term), và `MotorActionCfg` được dùng làm cfg cho `Propulsion` trong script PID
(`flight_test.py`, `pid_hover_test.py`) và làm lớp cha của `CascadeActionCfg`.

| Phần | Kiểm tra |
|---|---|
| `process_actions` (71) | `raw = clip(offset + scale·a, 0, 1)`, `pwm = raw·65535` |
| `apply_actions` (75) | gọi `Propulsion.step` **mỗi tick physics** (Isaac Lab gọi `apply_action` trong vòng decimation) |
| `reset` (64-69) | raw, pwm = 0; `Propulsion.reset`; **lực = hover/4** (mới sửa) |
| `MotorActionCfg` (80-107) | mặc định: trễ bật, drag bật, bất đối xứng động cơ bật, nhiễu tắt, quay cánh bật |

Checklist:
- [ ] Landing: `offset = 0.502`, `scale = 0.25` → lệnh ∈ [0.25, 0.75] khi a ∈ [−1, 1] (không clip action ở rsl_rl landing vì `clip_actions` không đặt; phần clip [0,1] nằm ở đây).
- [ ] `reset`: `raw_actions = 0` nhưng lực = hover. `raw_actions` chỉ ảnh hưởng `ActionHistory` (lịch sử bắt đầu bằng 0). Ổn.
- [ ] Sau reset, bước đầu tiên `process_actions` ghi PWM mới trước khi `apply_actions` chạy, nên PWM = 0 không bao giờ được áp.

---

## Cấp 3 — PID cascade và bay động học (★★)

Guide: `guide/03_kinematic_flight.md`, `guide/04_pid_cascade.md` (bảng 4.2 = gain và giới hạn).

Quan trọng: PID là **khuôn mẫu** của RL. Mỗi lớp RL có cùng đầu vào/đầu ra với lớp PID cùng tên, và khi train một lớp
RL, các lớp PID phía trên sinh lệnh cho nó (`LayerCommand`). Nên PID sai → lệnh train sai.

### 3.1 `pid_control/rate.py` — `RateController`

- `update(wanted_rates, body_rates, dt)` → `inertia · PID(ω* − ω)` = mô-men [N·m].
- Gain: kp (28.4, 28.4, 9.7), kd (0.30, 0.30, 0.14); **không** có giới hạn đầu ra.

Checklist:
- [ ] Đơn vị: PID ra gia tốc góc [rad/s²], nhân quán tính → mô-men. Đúng.
- [ ] Bỏ qua ω × Iω (gyroscopic) trong luật điều khiển: chấp nhận được ở tốc độ góc nhỏ.
- [ ] `_control` của test: giữ độ cao bằng position+velocity PID trên trục z, thrust = m(g + a_z) không chia cos(tilt).

### 3.2 `pid_control/attitude.py` — `AttitudeController`

`update(wanted_acceleration, quat, dt, wanted_yaw)`:
1. `R = quat_to_rotmat(quat)` (quat x,y,z,w), `up = R[:, :, 2]`.
2. f = a* + g·e_z; hướng muốn = f/|f|.
3. **thrust = m·(f · z_body)**, kẹp ≥ 0 (chiếu lên trục thân hiện tại).
4. Trục nghiêng: `z_body × wanted_up` (world) → sang thân bằng Rᵀ (einsum) → lấy x, y.
5. yaw error = wrap(ψ* − ψ), ψ = atan2(R10, R00).
6. PID P thuần: kp (17.3, 17.3, 4.0), giới hạn (6, 6, 3) rad/s.

Checklist:
- [ ] Bước 3 là công thức mà `AttitudeLayer.output` dùng làm feedforward. So từng ký hiệu.
- [ ] Bước 4: |z × w| = sin(góc), nên sai số ≈ góc khi nhỏ; bão hòa khi góc > 90°.
- [ ] Yaw lấy từ R không phụ thuộc nghiêng lớn: gần 90° pitch thì atan2 suy biến (chấp nhận).
- [ ] `angles_to_acceleration`: gia tốc ngang giữ một góc nghiêng, thành phần z = 0.

### 3.3 `pid_control/velocity.py`, `position.py`

| Lớp | Gain | Giới hạn |
|---|---|---|
| Velocity | kp (9.3, 9.3, 20), ki (0.9, 0.9, 1.9), int_limit (2, 2, 1) | (7, 7, 6) m/s² |
| Position | kp (3.9, 3.9, 6.8) | 1.6 m/s mỗi trục |

Checklist:
- [ ] Giới hạn này = hệ số scale của lớp RL tương ứng: `VelocityLayer.ACCELERATION_SCALE = (7, 7, 6)`,
  `AttitudeLayer.RATE_SCALE = (6, 6, 3)`. `PositionLayer.VELOCITY_SCALE = 1.5` < 1.6 (+ offset 0.5 khi train velocity).
- [ ] **Vấn đề mở**: các gain này tune ở 500 Hz (script test), nhưng trong `LayerCommand` chúng chạy ở 50 Hz (hoặc 100 Hz
  với rate task). Vòng z velocity kp = 20 ở dt = 20 ms: kp·dt = 0.4, cộng trễ động cơ 50–80 ms → có thể kém tắt dần.
  Cần kiểm runtime (xem đồ thị lệnh trong TensorBoard hoặc chạy thử một env).

### 3.4 `pid_control/cascade.py` — `CascadePID`

- Bốn controller + `force_allocation_inverse`.
- `to_pwm(thrust, torque)`: wrench → lực từng động cơ (`wrench @ inv.T`) → `thrust_to_pwm`.
- `step(...)`: position → velocity → attitude (yaw mặc định 0) → rate → PWM.

Checklist:
- [ ] `wrench @ inv.T` đúng chiều nhân (hàng = env).
- [ ] Lực âm từ phân bổ → PWM kẹp về 0 hoặc DZ: mô-men bị bão hòa, không báo.
- [ ] `step` không truyền `wanted_yaw` → luôn giữ yaw 0. `LayerCommand` gọi từng lớp riêng và truyền yaw.

Lệnh: `$PY -m pytest tests/test_cascade.py -q`.

### 3.5 Script mô phỏng (đọc logic, **không cần chạy**)

| File | Vai trò | Kiểm tra |
|---|---|---|
| `pid_control/flight_test.py` | runner chung cho 4 test lớp (`LayerTest`, `run`, `save_plot`, `report`, `euler_from_quat`, `step_table`) | trễ động cơ cố định τ_inc min, không drag, không bất đối xứng; `propulsion.step` → `sim.step` → `robot.update(dt)` |
| `pid_control/pid_hover_test.py` | bay hover/vuông với cả cascade | cùng `Propulsion` |
| `pid_control/kinematics.py` | quỹ đạo + differential flatness | `trajectory`, `_attitude`, `_body_rates_at` (sai phân hữu hạn h = 2 ms), `state` (lực/mô-men cần), `quat_to_rotmat`/`rotmat_to_quat` (x,y,z,w) |
| `pid_control/run_kinematic.py` | ghi tư thế theo quỹ đạo, báo có bay được không | `spin_propellers` dùng `SPIN_VISUAL_SCALE` (giờ 0.06) |

Checklist:
- [ ] `quat_to_rotmat` đúng thứ tự (x, y, z, w) và ra ma trận body→world.
- [ ] `euler_from_quat` thứ tự Rz·Ry·Rx.
- [ ] Đồ thị `guide/media/pid_*.png` hợp lý (bám setpoint, không dao động).

---

## Cấp 4 — Bốn lớp RL thuần torch: `src/Drone_RL/uav/mdp/layers.py` (★★★)

**Câu hỏi của cấp này:** mỗi lớp RL *nhìn thấy gì* và *xuất ra cái gì*, và hai thứ đó khớp với PID cùng tên ra sao.

Bốn lớp xếp chồng giống hệt cascade PID. Mỗi lớp là một mạng nhỏ: đọc **lệnh của lớp trên** cộng **trạng thái bay**, rồi ghi
**lệnh cho lớp dưới**. Riêng lớp rate ghi thẳng 4 lệnh động cơ.

```text
position (50 Hz)  --v*, yaw*-->  velocity (100 Hz)  --a*, yaw*-->  attitude (250 Hz)  --ω*, T*-->  rate (500 Hz)  --> 4 động cơ
```

| Lớp | Tần số | Cứ bao nhiêu tick physics (2 ms) | Lệnh nhận vào | Lệnh xuất ra | obs | action | history |
|---|---|---|---|---|---|---|---|
| rate | 500 Hz | 1 | ω* (3, rad/s), T* (1, N) | 4 động cơ trong [0, 1] | 23 | 4 | 4 |
| attitude | 250 Hz | 2 | a* (3, m/s², khung world), ψ* | ω* (3), T* (1) | 18 | 4 | 2 |
| velocity | 100 Hz | 5 | v* (3, m/s, world), ψ* | a* (3), ψ* | 15 | 3 | 2 |
| position | 50 Hz | 10 | p* (3, m), ψ* | v* (3), ψ* | 12 | 3 | 2 |

(Chạy `$PY -c "from Drone_RL.uav.mdp.layers import LAYERS; print({k:(l.obs_dim,l.action_dim,l.hz) for k,l in LAYERS.items()})"` để
tự kiểm các con số trong bảng.)

Điều quan trọng nhất của file này: **cùng một đoạn code** định nghĩa quan sát và cách đổi đầu ra mạng thành đơn vị vật lý.
Nó được dùng ở hai nơi:

1. Lúc train lớp đó: `observations.layer_observation` và `CascadeAction.process_actions`.
2. Lúc lớp đó bị đông cứng nằm dưới một lớp khác: `cascade_action.FrozenLayer.step`.

Nhờ vậy mạng đã đông cứng thấy đúng loại dữ liệu nó đã thấy khi train. Nếu hai nơi dùng hai đoạn code khác nhau, lớp trên
sẽ chạy một lớp dưới "lạ" mà không có lỗi nào báo ra.

Guide đọc cùng: `guide/05_rl_cascade.md` mục 5.1–5.2.

### 4.1 Tiện ích: `FlightState`, `tilt_error` và các khung toạ độ

`FlightState` là ảnh chụp trạng thái của drone tại một thời điểm, lấy từ simulator. Điều dễ nhầm nhất là mỗi đại lượng nằm
ở **một khung khác nhau**:

| Trường | Kích thước | Khung | Ghi chú |
|---|---|---|---|
| `position` | (N, 3) | world, **trừ gốc của env** | Mỗi env có gốc riêng; nhờ trừ nên mọi env đều nghĩ "gốc = (0, 0, 0)" |
| `rotation` | (N, 3, 3) | ma trận `R`, body → world | Từ quaternion (x, y, z, w) qua `matrix_from_quat` |
| `velocity` | (N, 3) | world | |
| `body_rates` | (N, 3) | **body** (`root_ang_vel_b`) | Tốc độ góc đo trong khung thân, giống con quay trên drone thật |

Cột thứ 3 của `R` (`R[:, :, 2]`) là trục z của thân nhìn từ world, tức **hướng lực đẩy đang chỉ**. Hàng thứ 3 (`R[:, 2, :]`)
là trục z của world nhìn từ thân. Hai thứ này khác nhau và bị nhầm rất dễ (xem checklist).

Các hàm nhỏ:

- `wrap_angle(x)` = `atan2(sin x, cos x)`: đưa một góc về (−π, π]. Ví dụ ψ* = 3,0 rad và ψ = −3,0 rad: hiệu thô là 6,0 rad nhưng
  sau `wrap` là −0,28 rad (xoay ngắn hơn đi đường kia).
- `yaw_of(R)` = `atan2(R[1,0], R[0,0])`: góc yaw của thân.
- `tilt_error(R, a*)`: góc giữa trục lực đẩy hiện tại và hướng lực đẩy mà gia tốc mong muốn a* đòi hỏi. Hướng cần có là
  `(a* + g·e_z)` chuẩn hoá. **Ví dụ:** drone nằm ngang, a* = (1,73; 0; 0) m/s² → lực cần có = (1,73; 0; 9,81), nghiêng
  atan(1,73/9,81) = 10° so với đứng thẳng → `tilt_error` = 0,175 rad. Khi a* = 0 và drone nằm ngang thì bằng 0.

Checklist:
- [ ] `tilt_error` dùng `R[:, :, 2]` (trục lực đẩy trong world), **không** phải `R[:, 2, :]`.
- [ ] `clamp(min=1e-6)` ở bước chuẩn hoá: khi a* = −g (rơi tự do) vector lực bằng 0 và hướng không xác định; không clamp thì chia cho 0.
- [ ] `matrix_from_quat` của Isaac Lab 3.0 nhận quaternion thứ tự (x, y, z, w). (Đúng với bản đang dùng.)
- [ ] `body_rates` là khung **thân**, không phải world. Nếu ai đó đổi sang `root_ang_vel_w` thì mọi quan sát rate sẽ sai mà không có lỗi báo.

### 4.2 Lớp cơ sở `Layer`

Mỗi lớp khai báo 6 thuộc tính: `name`, `hz`, `command_dim`, `action_dim`, `history`, `below` (tên lớp mà nó ghi lệnh xuống;
`None` nghĩa là động cơ), và hai hàm:

- `observe(state, command, history)` → quan sát đưa vào mạng.
- `output(action, command, state)` → đầu ra mạng (trong [−1, 1]) đổi thành lệnh vật lý cho lớp dưới.

`obs_dim` **không được gõ tay**: nó được tính bằng cách chạy thử `observe` trên đầu vào toàn số 0 rồi đo kích thước. Vì vậy
sửa `observe` thì `obs_dim` tự đổi, không bao giờ lệch với code.

### 4.3 Quan sát của từng lớp, từng chỉ số

Cách đọc bảng: cột "Chỉ số" là vị trí trong vector đưa vào mạng; "Vì sao có" giải thích lý do thiết kế để bạn đánh giá được
nó đúng hay thừa.

**RateLayer** (obs = 23): lệnh = [ω* (3), T* (1)]

| Chỉ số | Nội dung | Vì sao có |
|---|---|---|
| 0:3 | ω* − ω (rad/s, khung thân) | Sai số tốc độ góc: thứ mạng phải triệt tiêu |
| 3:6 | ω | Cho mạng biết bản thân đang quay nhanh cỡ nào (để giảm tốc) |
| 6 | T*/W (W = trọng lượng 0,441 N) | Lực đẩy mong muốn, chia cho trọng lượng nên ≈ 1 khi hover |
| 7:23 | 4 đầu ra gần nhất × 4 động cơ, mới nhất trước | Mạng biết mình vừa ra gì; reward `output_change` dùng lại chính nó |

*Ví dụ:* ω* = (0,5; 0; 0), ω = (0,3; 0; 0), T* = W → `obs[0:3]` = (0,2; 0; 0), `obs[3:6]` = (0,3; 0; 0), `obs[6]` = 1,0.

**AttitudeLayer** (obs = 18): lệnh = [a* (3, world), ψ*]

| Chỉ số | Nội dung | Vì sao có |
|---|---|---|
| 0:3 | Rᵀ(a* + g·e_z) / g | Lực/khối lượng mong muốn **trong khung thân**, chia g. Drone nằm ngang, a* = 0 cho (0, 0, 1) |
| 3 | wrap(ψ* − ψ) | Sai số yaw đã gói về (−π, π] |
| 4:7 | `R[:, 2, :]` | Hướng "lên" của world nhìn từ thân: mạng biết mình đang nghiêng bao nhiêu |
| 7:10 | ω | Tốc độ góc hiện tại |
| 10:18 | 2 đầu ra gần nhất × 4 | Lịch sử |

**VelocityLayer** (obs = 15): lệnh = [v* (3), ψ*]

| Chỉ số | Nội dung |
|---|---|
| 0:3 | v* − v (world) |
| 3:6 | v |
| 6:9 | `R[:, :, 2]`: trục lực đẩy trong world, tức drone đang đẩy về hướng nào |
| 9:15 | 2 đầu ra gần nhất × 3 |

**PositionLayer** (obs = 12): lệnh = [p*, ψ*]

| Chỉ số | Nội dung |
|---|---|
| 0:3 | p* − p (cả hai tương đối gốc env) |
| 3:6 | v |
| 6:12 | 2 đầu ra gần nhất × 3 |

Lưu ý thiết kế:
- **Velocity và position không nhìn thấy yaw.** ψ* chạy xuyên qua chúng: `output()` chỉ ghép nguyên `command[:, 3:4]` vào đầu ra.
  Attitude là lớp đầu tiên quyết định yaw.
- **Position không nhìn thấy tư thế.** Nó chỉ ra v*, còn nghiêng người thế nào là việc của velocity và attitude bên dưới.
- Tất cả quan sát nằm trong cùng khung với lúc train vì cùng một code, nên không cần kiểm riêng.

### 4.4 Đầu ra: mạng ra số trong [−1, 1], `output()` đổi thành đơn vị vật lý

| Lớp | Công thức | Hằng số |
|---|---|---|
| Rate | `u = clamp(ff(T*/4) / 65535 + 0,2 · a, 0, 1)` | `MOTOR_SCALE = 0,2`; `ff` = nghịch đảo đường cong lực đẩy (`thrust_to_pwm`) |
| Attitude | `ω* = (6, 6, 3) · a[0:3]`; `T* = max(0, m(a* + g)·z_thân) + 0,5·W·a[3]` | `RATE_SCALE`, `THRUST_SCALE = 0,5` |
| Velocity | `a* = (7, 7, 6) · a`; ψ* chuyển tiếp nguyên | `ACCELERATION_SCALE` |
| Position | `v* = 1,5 · a`; ψ* chuyển tiếp nguyên | `VELOCITY_SCALE` |

**Hai "feedforward" cần hiểu cho kỹ** (đây là chỗ khó nhất của file):

*Rate.* Nếu cho mạng tự ra thẳng lệnh động cơ từ 0, nó phải tự học ra "hover = 0,6055". Lần chạy đầu làm vậy không học được.
Nên code tính trước phần đã biết: `ff(T*/4)` là lệnh động cơ để **mỗi** động cơ ra đúng T*/4 newton theo đường cong lực đẩy. Mạng chỉ học phần
hiệu chỉnh ±0,2 để tạo mô-men (động cơ này mạnh hơn, động cơ kia yếu hơn) và sửa lực đẩy.
*Ví dụ:* T* = W → ff = 0,6055 = hover throttle. a = +1 → 0,8055; a = −1 → 0,4055. T* = 1,5·W → ff = 0,794.

*Attitude.* Tương tự cho lực đẩy: `m(a* + g)·z_thân` chính là công thức của PID attitude (chiếu lực mong muốn lên trục lực đẩy hiện tại).
Mạng chỉ cộng thêm hiệu chỉnh tối đa nửa trọng lượng. *Ví dụ:* drone nằm ngang, a* = 0 → ff = m·g = W = 0,441 N. Mạng ra a[3] = +1 → T* = 0,662 N.
Lần chạy đầu không có phần feedforward này (T* = mg(1 + a)) học ra lực đẩy sai và drone rơi khi nằm dưới lớp velocity.

Kiểm tra mức bão hoà: attitude có thể đòi tối đa `m(g + 6)·1 + 0,5·W` ≈ 1,61 W + 0,5 W = **2,11 W**, trong khi 4 động cơ ở PWM tối đa cho
4 × 0,232 N = 0,929 N = **2,10 W**. Hai con số xấp xỉ bằng nhau: lệnh lực đẩy tối đa nằm đúng ở mép giới hạn của động cơ, không vượt xa.

Checklist:
- [ ] Rate: a = 0 và T* = W cho đúng hover throttle 0,6055 (test `test_zero_rate_action_at_hover_thrust_is_hover_throttle`).
- [ ] Attitude: feedforward dùng **trạng thái hiện tại** (`z_thân`), đúng như PID attitude; a = 0 cho đúng lực mà PID sẽ ra.
- [ ] Giới hạn đầu ra của mỗi lớp = out_limit của PID cùng tên (6/6/3 rad/s, 7/7/6 m/s², 1,5 m/s so với 1,6 m/s của PID position). Nhờ vậy
  lệnh gửi xuống nằm trong vùng lớp dưới đã học — **với điều kiện** offset khi train lớp dưới đủ rộng. Đối chiếu ở mục 6.2 và 7.2.
- [ ] Hai comment "lần đầu không có feedforward thì không học được" trong `RateLayer.output` và `AttitudeLayer.output` là lịch sử lý do thiết kế, không phải ghi chú tạm.

### 4.5 `LAYERS` và `layers_below`

- `LAYERS` là dict theo thứ tự train: rate trước, position sau cùng.
- `layers_below("position")` = [velocity, attitude, rate]: từ lớp ngay dưới xuống thấp nhất, cũng là **thứ tự chạy từ trên xuống**.
- `below`: rate → `None`, attitude → rate, velocity → attitude, position → velocity.

Lệnh kiểm tra nhanh:

```bash
$PY -c "
from Drone_RL.uav.mdp.layers import LAYERS, layers_below
print({k: (l.obs_dim, l.action_dim, l.hz, l.history, l.below) for k, l in LAYERS.items()})
print([l.name for l in layers_below('position')])
"
# {'rate': (23, 4, 500.0, 4, None), 'attitude': (18, 4, 250.0, 2, 'rate'),
#  'velocity': (15, 3, 100.0, 2, 'attitude'), 'position': (12, 3, 50.0, 2, 'velocity')}
```

---

## Cấp 5 — Đông cứng một lớp (★★★)

**Câu hỏi của cấp này:** làm sao một lớp đã train xong được "đóng băng" thành một file nhỏ, để lớp bên trên dùng nó như một
hàm cố định mà không train lại nó.

Quy trình: train lớp X → `freeze` chép actor của checkpoint cuối vào `rl_control/frozen/X.pt` → lớp bên trên đọc file đó.
Lớp trên **không bao giờ đọc thư mục `logs/`**, nên train lại lớp X không đổi gì cho lớp trên cho đến khi bạn freeze lại.

### 5.1 `src/Drone_RL/uav/mdp/actions/frozen_policy.py` — dựng lại actor từ file

File `.pt` chỉ chứa 3 thứ: `layer` (tên), `actor_state_dict` (đã lọc, chỉ giữ khoá `obs_normalizer.*` và `mlp.*`), `source` (đường
dẫn checkpoint gốc, để biết nó từ lần train nào).

`FrozenPolicy` dựng lại đúng actor mà rsl_rl dùng:

1. Chuẩn hoá quan sát: `x = (obs − mean) / (std + 0,01)`. `0,01` là `eps` của `EmpiricalNormalization` của rsl_rl 5.5.1. Con số này lấy cứng từ bản rsl_rl đó;
   nếu nâng cấp rsl_rl thì phải đối chiếu lại.
2. Chạy các lớp Linear, xen ELU **sau mọi Linear trừ cái cuối**.
3. Trả về **trung bình** (mean) của phân phối Gaussian, không lấy mẫu ngẫu nhiên: lúc chạy đông cứng không có nhiễu thăm dò.

Cách tìm các lớp Linear: lấy mọi khoá dạng `mlp.<i>.weight` rồi sắp theo **số** `i` (không theo chuỗi, vì "10" < "2" nếu so chuỗi). Mạng 64×64 có `i` = 0, 2, 4 (xen với ELU).

`load_frozen(layer, device, frozen_dir)`:
- Thiếu file → `FileNotFoundError` kèm hướng dẫn lệnh train + freeze.
- Kích thước obs/action trong file khác `layers.py` → `ValueError` "layer changed, retrain".

Checklist:
- [ ] Không có normaliser trong file → mean 0 và std = 1 − eps, để `(std + eps) = 1` (tức không chuẩn hoá).
- [ ] `requires_grad_(False)` và `eval()`: không có gradient chảy về lớp đã đông cứng.
- [ ] **Giới hạn lớn nhất của cơ chế này:** `load_frozen` chỉ kiểm **kích thước**. Nếu bạn đổi hằng số scale, feedforward, offset lệnh, tần số, hay mô hình
  động cơ nhưng giữ nguyên kích thước obs/action, file cũ vẫn load được và chạy "sai ngữ nghĩa" mà không báo gì. Đây là bẫy lớn nhất của hệ thống.
  Đề xuất (chưa làm): ghi một "chữ ký" (scale, feedforward, offset, tần số) vào `.pt` rồi so khi load.

### 5.2 `src/Drone_RL/uav/rl_control/freeze.py` — tạo file đông cứng

| Hàm | Làm gì |
|---|---|
| `FROZEN_DIR` | `rl_control/frozen/`, nơi mọi lớp trên đọc |
| `experiment_name(layer)` | `uav_<layer>`; cũng là tên experiment trong `agents/<layer>_ppo_cfg.py` nên `newest_checkpoint` tìm đúng thư mục |
| `newest_checkpoint(layer, log_root)` | Lấy run **mới nhất theo tên thư mục** (tên là timestamp) có ít nhất một `model_*.pt`, rồi lấy model có số lớn nhất (sắp theo số) |
| `freeze(layer, checkpoint, frozen_dir)` | Lọc actor, ghi `.pt`, rồi `load_frozen` lại ngay để lỗi lộ ra sớm |
| `main()` | `--layer`, `--checkpoint` (tuỳ chọn), `--log_root` (mặc định `logs/rsl_rl`, **tương đối với thư mục đang đứng**) |

Checklist:
- [ ] Phải chạy từ gốc repo, vì `log_root` là đường dẫn tương đối.
- [ ] Một run bị dừng giữa chừng vẫn có `model_<i>.pt`, nên `freeze` sẽ lấy nó nếu đó là run mới nhất. Xem `source` trong file để biết nó từ đâu:
  `$PY -c "import torch; print(torch.load('src/Drone_RL/uav/rl_control/frozen/rate.pt', weights_only=False)['source'])"`
- [ ] Task `Attitude-PIDRate` có experiment riêng `uav_attitude_pidrate` nên `freeze --layer attitude` **không bao giờ** lấy nhầm run của nó.

**Trạng thái hiện tại:** thư mục `rl_control/frozen/` **chưa tồn tại** (không có file đông cứng nào). Vì vậy các task attitude, velocity, position
(trừ `Attitude-PIDRate`, dùng PID rate nên không cần file) chưa chạy được cho đến khi train và freeze từ lớp rate. Hơn nữa, mọi policy cũ đã lỗi thời so
với code hiện tại (tần số mới, bỏ motor lag, khối lượng 45 g).

Lệnh: `$PY -m pytest tests/test_rl_layers.py -q -k "frozen or freeze"`.

---

## Cấp 6 — Các term MDP chạy trong Isaac Lab (★★★★)

**Câu hỏi của cấp này:** Isaac Lab gọi những hàm nào của mình, theo thứ tự nào, trong một bước của môi trường.

Một task RL của Isaac Lab ghép từ 5 "manager". Mỗi manager chứa các "term" do code ta viết:

| Manager | Term của ta | File |
|---|---|---|
| Action | `CascadeAction` | `mdp/actions/cascade_action.py` |
| Command | `LayerCommand` | `mdp/commands.py` |
| Observation | `layer_observation` | `mdp/observations.py` (+ `layers.py`) |
| Reward | `body_rate_tracking`, `tilt_tracking`... | `mdp/rewards.py` |
| Termination | `left_flight_envelope` | `mdp/terminations.py` |

Mở `guide/07_simulation_step.md` bên cạnh: nó cho biết **khi nào** mỗi hàm chạy trong một bước.

### 6.1 `mdp/actions/cascade_action.py` — chạy chuỗi lớp đông cứng và đẩy xuống động cơ

Mục đích: khi ta train lớp X, đầu ra của nó không đi thẳng vào vật lý mà phải đi qua **các lớp đã đông cứng bên dưới**, xuống tới 4 động cơ.
Ví dụ train lớp position: đầu ra v* → velocity (đông cứng) → attitude (đông cứng) → rate (đông cứng) → động cơ.

Các thành phần:

**`output_dim(layer)`**: kích thước đầu ra của một lớp = `command_dim` của lớp dưới nó, hoặc 4 nếu là rate.

**`History`**: tensor (N, độ dài, action_dim), phần tử mới nhất ở chỉ số 0. `push` = `torch.roll` một bước rồi ghi vào hàng 0.
Phần tử cũ nhất bị cuốn về chỉ số 0 rồi bị ghi đè ngay nên không rò rỉ.

**`FrozenLayer`**: một lớp đã đông cứng. `step(state, command)` làm 4 việc theo thứ tự: observe (dùng history *trước khi* đẩy) → policy → kẹp [−1, 1] → `push` vào history → `output()`.
`period` = `round(500 / hz)`: rate 1, attitude 2, velocity 5, position 10 (số tick physics giữa hai lần mạng chạy).

**`PIDRateLayer`**: thay lớp rate đông cứng bằng PID rate (khi `pid_rate = True`). Chuỗi: [ω*, T*] → `RateController` (ra mô-men τ, nhân thêm quán tính J) → nghịch đảo mixer
(ra lực mỗi động cơ) → nghịch đảo đường cong lực đẩy (ra PWM) → chia 65535 (ra [0, 1]). Cùng giao diện `step` / `reset` / `output` như `FrozenLayer`, nên `CascadeAction`
chạy nó giống một lớp đông cứng. `output` khởi tạo bằng hover throttle.

**`CascadeAction`** — phần chính:

| Phương thức | Khi nào chạy | Làm gì |
|---|---|---|
| `__init__` | lúc tạo env | Tạo `FrozenLayer` (hoặc `PIDRateLayer`) cho mọi lớp dưới. **Kiểm 500 Hz phải chia hết cho tần số mỗi lớp**, sai thì `ValueError` |
| `reset(env_ids)` | reset env | Xoá history, reset lớp đông cứng/PID rate, reset propulsion (lực 0, rút lại gain động cơ và drag), `_motor` = hover throttle |
| `process_actions(a)` | **1 lần mỗi bước env** | Kẹp [−1, 1] → push history → `layer.output(a, lệnh hiện tại, trạng thái)` → lưu `velocity_at_step_start` |
| `apply_actions()` | **mỗi tick physics** (decimation lần) | Chạy các lớp đông cứng từ trên xuống, ghi `_motor`, gọi `Propulsion.step(_motor · 65535)`, tăng `_tick` |
| `thrust` | reward `thrust_tracking` | Tổng lực 4 động cơ **ngay lúc này** (không còn độ trễ động cơ nên bằng lực vừa đặt) |

**Lịch chạy các lớp (điểm khó nhất).** `_tick` là bộ đếm tick physics **không bao giờ reset** (giống `RATE_DO_EXECUTE` trên firmware). Một lớp chạy khi `_tick % period == 0`.

Task position (decimation 10, một bước env = 10 tick):

```text
tick:       0 1 2 3 4 5 6 7 8 9
velocity:   ● . . . . ● . . . .     period 5  (100 Hz)
attitude:   ● . ● . ● . ● . ● .     period 2  (250 Hz)
rate:       ● ● ● ● ● ● ● ● ● ●     period 1  (500 Hz)
```

Trong một tick, thứ tự luôn là velocity → attitude → rate (trên xuống dưới). Lớp nào *không* chạy ở tick đó thì lớp dưới dùng **đầu ra cũ được giữ lại** của nó,
nhưng **trạng thái mới** của drone. Đây chính là hành vi của firmware: lớp trên cập nhật chậm, lớp dưới vẫn đo lại liên tục.

Vì sao `_tick` không được reset mỗi bước env? Xét task velocity (decimation 5) với lớp attitude đông cứng (period 2):

```text
bước env 0:   tick 0 1 2 3 4        attitude chạy ở 0, 2, 4         (3 lần)
bước env 1:   tick 5 6 7 8 9        attitude chạy ở 6, 8            (2 lần, vì 5, 7, 9 lẻ)
```

Trung bình 2,5 lần mỗi bước = đúng 250 Hz. Nếu reset `_tick` về 0 mỗi bước, attitude sẽ luôn chạy 3 lần mỗi bước = 300 Hz, lệch so với lúc train. Đó là lý do
bộ đếm toàn cục, và cũng là lý do điều kiện duy nhất cần đúng là "500 Hz chia hết cho tần số của từng lớp", **không** phải "decimation chia hết cho period".

Task rate: danh sách lớp đông cứng rỗng, `_motor = _output` (đầu ra của chính rate) ở mọi tick. Decimation là 1, nên mỗi bước env cũng là 1 tick.

Checklist:
- [ ] Task position, 10 tick: velocity chạy ở tick 0 và 5, attitude ở 0, 2, 4, 6, 8, rate ở mọi tick (test `test_cascade_runs_frozen_layers_top_down_at_their_rates`).
- [ ] `command()` trong `process_actions` là c_t, tức lệnh mà policy **đã thấy** lúc chọn a_t. Command manager chỉ cập nhật *sau* khi tính reward (xem cấp 9).
- [ ] Thứ tự observe-rồi-push của `FrozenLayer` giống lúc train: observe thấy history chứa a_k, ra a_{k+1}, rồi mới push. Test `test_frozen_layer_sees_what_the_layer_saw_in_training` bảo vệ điều này.
- [ ] Cấu hình `CascadeActionCfg` kế thừa `MotorActionCfg`: bật drag và bất đối xứng động cơ (rút lại mỗi episode); **tắt quay cánh** (chỉ để vẽ, chậm khi train).
- [ ] `frozen_dir` và `layer` là `MISSING`: env cfg bắt buộc điền (`cascade_env_cfg.py` đã điền; Pylance báo "MISSING not assignable to str" là cảnh báo giả quen thuộc của Isaac Lab).
- [ ] Không còn motor lag: `Propulsion.step` không nhận `dt`, lực = đường cong lực đẩy × gain, đặt ngay lập tức.

### 6.2 `mdp/commands.py` — `LayerCommand`: lệnh mà lớp đang train được thấy

Đây là chỗ phức tạp nhất về **ý nghĩa**, vì nó quyết định lớp đang train được tập luyện với kiểu lệnh nào.

Ý tưởng: lớp X đang train đáng lẽ nhận lệnh từ lớp phía trên. Nhưng các lớp phía trên chưa tồn tại (chưa train), nên ta **thay chúng bằng PID đã tune**,
chạy vòng kín trên chuyến bay thật, rồi cộng thêm một **offset ngẫu nhiên**.

```text
target ngẫu nhiên --PID position--> v* --PID velocity--> a* --PID attitude--> (ω*, T*)
                                    ↑ lớp position       ↑ lớp velocity     ↑ lớp attitude     ↑ lớp rate
PID chạy từ target xuống tới ngay trên lớp đang train; sau đó cộng offset.
```

Offset ngẫu nhiên là gì? Mỗi `offset_time` giây (0,3–1,5 s, riêng rate 0,2–0,8 s) env nào hết hạn sẽ rút offset mới ~ U(−scale, scale); với xác suất `p_zero` = 0,3 offset đó bằng 0.
Mục đích: để mạng gặp cả những lệnh mà PID sẽ không bao giờ đưa ra, tức là không chỉ học thuộc cách PID hành xử.

Lệnh theo từng lớp đang train:

| Lớp train | PID nào chạy | Lệnh (command) | Offset |
|---|---|---|---|
| position | không | target ngẫu nhiên [p*, ψ*] | không |
| velocity | position | [v*, ψ*] với v* ≤ 1,6 m/s từng trục | (0,5; 0,5; 0,5; 0) m/s |
| attitude | position → velocity | [a*, ψ*] với a* ≤ (7, 7, 6) m/s² | (2; 2; 1,5; 0) m/s² |
| rate | position (**chỉ trục z**) → velocity → attitude | [ω*, T*] | (6; 6; 3 rad/s; 0,22 N) |

Riêng **rate** dùng chế độ `level_only`: PID phía trên chỉ giữ độ cao 1–2 m và dừng drone (vận tốc ngang mong muốn = 0), nên lệnh gần như là 0 rồi cộng offset.
Lý do: với chuỗi PID đầy đủ, lệnh ω* luôn nằm ở giới hạn 6 rad/s và policy rate học cách đứng yên bỏ qua lệnh. Cũng vì muốn lệnh "nhẹ", yaw của target chỉ ±0,3 rad và drone xuất phát gần thăng bằng, đứng yên, quay gần đúng hướng target.

Target ngẫu nhiên: x, y ∈ [−1, 1] m, z ∈ [1, 2] m, yaw ∈ ±π, đổi mỗi 2–4 s (position: 3–5 s).

**`reset`**: reset PID của các env đó, xoá offset, gọi `super().reset` (ghi metric, rút target mới), xoá tổng metric. Nếu reset **mọi** env (chỉ xảy ra ở `env.reset()` đầu tiên) thì
gọi `_update_command()` ngay, vì không thì quan sát đầu tiên sẽ thấy lệnh bằng 0.

**`_update_command`**: mỗi bước env, rút offset nếu hết hạn, chạy các PID cần thiết, cộng offset, rồi `wrap` góc yaw (trừ lớp rate, vì cột 3 của lệnh rate là lực đẩy chứ không phải góc).

**Các PID ở đây được gọi với `dt = step_dt`**, không phải 2 ms: position task 20 ms, velocity 10 ms, attitude 4 ms, rate 2 ms. PID này được tune ở 500 Hz,
nên khi gọi chậm hơn nó không còn đúng hoàn toàn. Đây là vấn đề mở đã biết (phụ lục B, mục 2).

**`_update_metrics`**: sai số của lớp (rate: ‖ω*−ω‖; attitude: `tilt_error`; velocity: ‖v*−v‖; position: ‖p*−p‖) lấy **trung bình từ đầu episode**, không phải giá trị bước cuối.
Nó xuất hiện trên TensorBoard là `Metrics/layer/error`.

Checklist:
- [ ] Phủ vùng giữa các tầng (xem bảng "khớp phân bố" ở 7.2): lệnh mà tầng trên có thể gửi phải nằm trong vùng tầng dưới đã học.
- [ ] Nhánh rate dùng `root_quat_w` trực tiếp, các nhánh khác dùng `FlightState`: cùng một dữ liệu.
- [ ] Offset rate (6 rad/s) mà PID attitude (kp 8,6) kéo về: một offset 6 rad/s được giữ lâu sẽ cân bằng ở nghiêng ≈ 6/8,6 ≈ 0,7 rad. Kiểm tra lại bằng mắt trong video train.
- [ ] Reset toàn bộ gọi `_update_command` làm PID của mọi env cập nhật thêm một lần. Chỉ xảy ra ở `env.reset()` hoặc khi mọi env cùng hết episode: chấp nhận được.
- [ ] `metrics["error"]` chia cho số bước ≥ 1 (chỉ chia sau khi đã tăng bước).

### 6.3 `mdp/observations.py` — quan sát

| Hàm / class | Dùng bởi | Làm gì |
|---|---|---|
| `layer_observation` | 4 task cascade | = `term.layer.observe(term.state(), command, term.history.values)`: ghép `layers.py` vào manager |
| `ActionHistory` | landing | Giữ 4 lệnh động cơ gần nhất, **chỉ đẩy một lần mỗi bước env** (kiểm `common_step_counter`), reset xoá env đó |
| `position_relative_to_target`, `orientation_matrix`, `linear_velocity_world`, `angular_velocity_body` | landing | Các thành phần theo danh sách trạng thái của Eschmann et al. |
| `ArucoObservation` | landing | (u, v, kích thước, found) từ camera nhìn xuống. Mất marker thì giữ giá trị cũ, `found = 0` báo là cũ |

Checklist:
- [ ] `layer_observation` dùng **cùng** `History` mà `process_actions` đẩy vào: nhờ vậy khớp với `FrozenLayer`.
- [ ] `ActionHistory` đẩy một lần mỗi bước vì observation có thể được tính hai lần trong một bước (quan sát cuối, recorder).
- [ ] ArUco: ảnh (N, H, W, 3) đưa sang CPU numpy mỗi bước nên chậm; dùng ít env (32).

### 6.4 `mdp/rewards.py` — phần thưởng

Mọi reward "tracking" là **hàm kernel** `exp(−e²/σ²)`: bằng 1 khi sai số 0, giảm dần khi sai số tăng. *Ví dụ:* σ = 1, sai số 1 → 0,37; sai số 0,5 → 0,78; sai số 2 → 0,02.
Nhược điểm: khi sai số lớn hơn nhiều so với σ, kernel phẳng gần 0 và policy không nhận được hướng nào để cải thiện. Vì vậy mỗi task thêm một
reward `*_error_l2` (bình phương sai số, **trọng số âm**) có độ dốc ở cả sai số lớn; và có hai kernel (σ thô và σ tinh) để học thô trước, mịn sau.

Reward được tính **sau physics, trước khi cập nhật lệnh**, nên nó so trạng thái *sau* bước với lệnh c_t mà policy đã thấy khi chọn hành động. Reward manager nhân mọi term với
`step_dt` (xem 7.2).

| Hàm | Lớp | Công thức |
|---|---|---|
| `body_rate_tracking(σ)` | rate | exp(−‖ω* − ω‖² / σ²) |
| `body_rate_error_l2` | rate | ‖ω* − ω‖² (trọng số âm) |
| `thrust_tracking(σ)` | rate | exp(−((T − T*) / W)² / σ²), T = tổng lực 4 động cơ hiện tại |
| `tilt_tracking(σ)`, `tilt_error_l2` | attitude | theo `tilt_error` |
| `yaw_tracking(σ)` | attitude | theo wrap(ψ* − ψ) |
| `acceleration_tracking(σ)` | attitude | (v_hiện tại − v_đầu_bước) / step_dt so với a*: gia tốc **trung bình trong bước** |
| `velocity_tracking(σ)`, `velocity_error_l2` | velocity | |
| `position_tracking(σ)`, `position_error_l2` | position | |
| `output_change` | mọi lớp | ‖h[0] − h[1]‖²: phạt thay đổi đầu ra giữa hai bước |
| `body_rates_l2` | attitude | ‖ω‖²: phạt quay |
| `alignment`, `descent_over_pad`, `touchdown_speed_penalty`, `landed_bonus`, `crashed_penalty` | landing | xem cấp 8 |

Checklist:
- [ ] Reward tính trước khi cập nhật lệnh: so trạng thái sau bước với c_t. Nếu đổi thứ tự sẽ phạt policy vì một lệnh nó chưa thấy.
- [ ] Sau reset, `output_change` ở bước đầu = ‖a_0 − 0‖² (history toàn 0): phạt nhỏ, chấp nhận được.
- [ ] `acceleration_tracking`: reset xảy ra *sau* reward trong `step`, nên `velocity_at_step_start` luôn thuộc cùng episode.

### 6.5 `mdp/terminations.py` — khi nào kết thúc episode

`left_flight_envelope` kết thúc episode khi drone **ra khỏi hộp bay**: thấp hơn 0,2 m, cao hơn 4 m, cách gốc env xa hơn 3 m theo mặt phẳng ngang,
hoặc nghiêng hơn 1,57 rad (≈ 90°, tính từ `R[2,2]` = cos của góc nghiêng). Cộng với `time_out` khi hết `EPISODE_S`.

Drone khởi động ở 1,5 m (rate: 1,5 ± 0,1 m), nên còn khoảng 1,3 m để rơi trước khi chạm ngưỡng 0,2 m.

Các hàm `landed`, `crashed`, `upright`, `pad_distance_and_height`, `linear_speed` là của task hạ cánh (cấp 8).

---

## Cấp 7 — Cấu hình task và PPO (★★★)

**Câu hỏi của cấp này:** ghép các term ở cấp 6 thành một task, rồi cho PPO train nó. Cấp này toàn là con số cấu hình, nên phần khó là
kiểm chúng nhất quán với nhau.

### 7.1 `rl_control/cascade_env_cfg.py` — phần chung của 4 task

| Phần | Giá trị |
|---|---|
| `FlightSceneCfg` | đất 200 × 200 m, đèn dome, robot bắt đầu ở 1,5 m |
| `ActionsCfg.cascade` | `CascadeActionCfg(layer="rate", frozen_dir=...)`, `layer` bị ghi đè trong `__post_init__` |
| `CommandsCfg.layer` | `LayerCommandCfg`, cũng bị ghi đè |
| `ObservationsCfg` | 1 term `layer_observation`, **không thêm nhiễu** |
| `EventCfg.reset_drone` | xy ±0,5 m, z ±0,3 m, roll/pitch ±0,3 rad, yaw ±π; vận tốc ±0,3 m/s, ±0,5 rad/s |
| `TerminationsCfg` | `time_out`, `left_envelope` |
| `termination_penalty(w)` | `is_terminated` (không tính time_out) nhân `w` |

`LayerEnvCfg.__post_init__` đặt: tên lớp cho action và command; `sim.dt = 1/500`; **`decimation = round(500 / hz)`**; `render_interval = decimation`; viewer theo env 0.

| Lớp | decimation | Một bước env | Độ dài episode | Số bước / episode |
|---|---|---|---|---|
| rate | 1 | 2 ms | 5 s | 2500 |
| attitude | 2 | 4 ms | 6 s | 1500 |
| velocity | 5 | 10 ms | 6 s | 600 |
| position | 10 | 20 ms | 8 s | 400 |

Checklist:
- [ ] `LAYER` là thuộc tính lớp (không phải field configclass), nên mỗi file task con đổi được.
- [ ] 500 Hz chia hết cho tần số mỗi lớp (kiểm ở `CascadeAction.__init__`).
- [ ] `termination_penalty` có trọng số tính **theo giây**; reward manager nhân với `step_dt`, nên giá trị trừ mỗi lần ngã **phụ thuộc vào lớp** (xem bảng dưới).

### 7.2 Bốn file task

| Task | Reward (trọng số; σ) | Ghi đè | Phạt khi ngã (w × dt) |
|---|---|---|---|
| `rate_env_cfg.py` | rate thô (1; 4), rate tinh (1; 1), rate_error² (−0,02), thrust (1; 0,1), output_change (−0,2), ngã (−1000) | `level_only`, yaw ±0,3, `offset_time` 0,2–0,8 s, reset nhỏ hơn (xy ±0,2, z ±0,1, roll/pitch ±0,1, yaw ±0,3), không vận tốc đầu, **offset (6; 6; 3; 0,22 N)** | −1000 × 0,002 = **−2** |
| `attitude_env_cfg.py` | tilt (2; 0,2), tilt_error² (−1), yaw (1; 0,4), accel (1; 2), output_change (−0,05), spin (−0,01), ngã (−500) | offset (2; 2; 1,5; 0) m/s² | −500 × 0,004 = **−2** |
| `velocity_env_cfg.py` | vel thô (2; 1), vel_error² (−0,1), vel tinh (1; 0,15), output_change (−0,05), ngã (−500) | offset (0,5; 0,5; 0,5; 0) m/s | −500 × 0,01 = **−5** |
| `position_env_cfg.py` | pos thô (2; 1), pos_error² (−0,1), pos tinh (1; 0,1), output_change (−0,05), ngã (−500) | `resampling_time_range` 3–5 s | −500 × 0,02 = **−10** |

Task `Attitude-PIDRate` (`AttitudePIDRateEnvCfg`) giống hệt attitude, chỉ thêm `pid_rate = True`: lớp rate bên dưới là PID thay vì mạng đông cứng, nên **không cần file `rate.pt`**.

Điểm cần cân nhắc: phạt khi ngã **co lại** khi tần số tăng, vì trọng số nhân với `dt`. Tổng reward tối đa mỗi episode của rate là khoảng 3/s × 5 s = 15,
còn cú ngã chỉ trừ 2 (13%). Với position, tổng tối đa 3/s × 8 s = 24 và cú ngã trừ 10 (42%). Tức là drone rate "ngã rẻ" hơn nhiều so với position;
nếu muốn mức phạt tương đương thì phải tăng trọng số `termination_penalty` của rate và attitude lên (chưa làm).

**Khớp phân bố giữa các tầng** (điều quan trọng nhất của cấp này): lệnh mà một tầng có thể gửi xuống phải nằm trong vùng tầng dưới đã học.

| Tầng dưới (đã học) | Nó được train với lệnh | Tầng trên có thể gửi xuống tối đa | Phủ? |
|---|---|---|---|
| rate | PID (nhỏ vì `level_only`) + offset ≤ (6, 6, 3) rad/s, ±0,22 N | attitude: ω* ≤ (6, 6, 3), T* tới ≈ 2,1 W | ω* phủ. T*: PID (quanh W, tới ≈ 1,6 W khi chỉnh độ cao) + 0,5 W tới ≈ 2,1 W trong trường hợp xấu nhất, nhưng vùng T* > 1,5 W hiếm gặp lúc train |
| attitude | a* từ PID velocity ≤ (7, 7, 6) + offset (2, 2, 1,5) | velocity: a* ≤ (7, 7, 6) | phủ (rộng hơn) |
| velocity | v* từ PID position ≤ 1,6 + offset 0,5 | position: v* ≤ 1,5 | phủ |

- [ ] Rate được train với reset nhỏ hơn các task khác (xuất phát gần thăng bằng, đứng yên). Task trên reset với nghiêng ±0,3 rad và tốc độ ±0,5 rad/s, nên lớp rate sẽ gặp trạng thái lớn hơn lúc train trong vài bước đầu mỗi episode. Offset rộng bù một phần.
- [ ] Tên term reward chính là tên hiện trên TensorBoard `Episode_Reward/<tên>`.

### 7.3 `rl_control/agents/*_ppo_cfg.py` — PPO (rsl_rl)

Chung cho cả 4 lớp (`cascade_ppo_cfg.py`):

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| Actor | 64 × 64, ELU, **chuẩn hoá quan sát nằm trong actor** | Để `FrozenPolicy` mang chuẩn hoá theo. Đủ nhỏ cho STM32 của Crazyflie |
| Critic | 128 × 128, ELU | Chỉ dùng lúc train |
| `BoundedGaussianCfg.std_range` | (0,02; 0,5) | Kẹp độ lệch chuẩn nhiễu hành động. Không kẹp thì std của lần chạy rate đầu tăng từ 0,2 lên 8,5: hành động nằm ở hai đầu và động cơ chỉ bật/tắt |
| `entropy_coef` | 0 | Nhiễu hành động chỉ thu nhỏ dần khi policy học tốt. (0,002 từng giữ std ở mức trần) |
| `learning_rate` | 1e-3, lịch **adaptive** theo KL mong muốn 0,01 | |
| `gamma`, `lam` | 0,99; 0,95 | |
| `clip_actions` | 1,0 | Khớp với kẹp [−1, 1] ở `CascadeAction` |

Riêng từng lớp:

| Lớp | num_steps/env | Iteration | init_std | std_range | Số tham số actor |
|---|---|---|---|---|---|
| rate | 32 | 1500 | 0,1 | (0,02; 0,3) | 5956 |
| attitude | 24 | 1000 | 0,5 | (0,02; 0,5) | 5636 |
| velocity | 24 | 600 | 0,2 | (0,02; 0,5) | 5379 |
| position | 24 | 600 | 0,2 | (0,02; 0,5) | 5187 |

Số tham số tính từ 64 × 64 với đầu vào/ra của từng lớp (ví dụ rate: 23·64+64 + 64·64+64 + 64·4+4 = 5956).

**Điểm cần cân nhắc (chưa chỉnh sau khi đổi tần số).** `num_steps_per_env` và `gamma` chưa được chỉnh theo tần số mới:

| Lớp | Một rollout dài | Tầm nhìn của γ = 0,99 (≈ 100 bước) |
|---|---|---|
| rate | 32 × 2 ms = **64 ms** | **0,2 s** |
| attitude | 24 × 4 ms = 96 ms | 0,4 s |
| velocity | 24 × 10 ms = 0,24 s | 1 s |
| position | 24 × 20 ms = 0,48 s | 2 s |

Tầm nhìn 0,2 s của rate rất ngắn so với episode 5 s. Chưa rõ nó có thật sự là vấn đề hay không (rate chỉ cần phản ứng nhanh), nhưng khi train lại mà không học được thì
đây là chỗ đầu tiên nên nghi ngờ (thử tăng `gamma` lên 0,999 hoặc tăng `num_steps_per_env`).

Checklist:
- [ ] `experiment_name` = `uav_<layer>` nên `freeze.newest_checkpoint` tìm đúng thư mục. `Attitude-PIDRate` có tên `uav_attitude_pidrate` riêng.
- [ ] Số iteration trong các file này tính cho 1024 env (số dùng khi quay `--video`). Landing (cấp 8) dùng `landing_ppo_cfg.py`: actor 128-128-64, init_std 1,0, entropy 0,005, 1500 iteration.

### 7.4 Đăng ký task

| File | Làm gì |
|---|---|
| `rl_control/__init__.py` | `_TASKS` gồm **6 task**: Rate, Attitude, Attitude-PIDRate, Velocity, Position, Landing-ArUco. Entry point là chuỗi (lười), `ManagerBasedRLEnv` |
| `uav/__init__.py` | import `rl_control` (chạy `gym.register`) |
| `tasks/__init__.py` | `import_packages` + `import Drone_RL.uav` — chỗ Isaac Lab tìm task |

Lệnh: `$PY -m pytest tests/test_registration.py -q`.

---

## Cấp 8 — Hạ cánh ArUco (★★★)

Guide: `guide/06_aruco_landing.md`.

| File | Kiểm tra |
|---|---|
| `mdp/aruco.py` | `marker_image` (lề trắng cần cho detector), `save_marker_texture`, `ArucoDetector.detect` (tâm = trung bình 4 góc, cạnh = trung bình 4 cạnh, chỉ id 0) |
| `rl_control/marker_plate.py` | `spawn_aruco_plate`: mesh vuông mặt +z, UV (0,0)…(1,1), texture qua diffuse **và** emissive (không phụ thuộc ánh sáng), không collision |
| `rl_control/landing_env_cfg.py` | scene (pad 0.8 m, camera 192 px FOV 90° nhìn thẳng xuống, quat (0.707, −0.707, 0, 0) quy ước ros), action (offset 0.502, scale 0.25), obs (Eschmann + aruco), event (xy ±1, z 0.7–1.3, yaw ±π, không vận tốc), reward, termination, 500 Hz / decimation 10 / 10 s |

Checklist:
- [ ] Hướng ảnh: x ảnh = −y drone, lên trên ảnh = x drone (comment dòng 48). So với `guide/06`.
- [ ] Quaternion camera dạng (x, y, z, w) = (0.707, −0.707, 0, 0) với `convention="ros"`: xoay 180° quanh trục (1, −1, 0)/√2.
- [ ] `landed` thưởng 2500·dt = 50; `crashed` −1000·dt = −20; `descent_over_pad` chỉ trả khi thẳng trên pad.
- [ ] Episode bắt đầu ở lực hover (mới sửa).

Lệnh: `$PY -m pytest tests/test_aruco.py -q`.

---

## Cấp 9 — Toàn hệ thống theo thời gian (★★★★★)

**Câu hỏi của cấp này:** từ lúc PPO đưa ra một hành động đến lúc nhận lại quan sát và reward, **cái gì chạy, theo thứ tự nào**.

Đây là phần khó nhất vì không đọc từng file mà phải **đi theo một bước env** xuyên qua mọi file bạn đã đọc ở các cấp trước. Mở song song
`guide/07_simulation_step.md` và mã nguồn Isaac Lab `ManagerBasedRLEnv.step`
(`~/Documents/GitHub/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`).

### 9.1 Một bước env của task Position (decimation 10, mỗi bước 20 ms)

Hai khái niệm thời gian cần phân biệt:

- **Bước env** (20 ms): policy position chạy một lần. PPO gọi `env.step(a_t)` một lần.
- **Tick physics** (2 ms): vật lý chạy một lần. Một bước env gồm 10 tick.

```text
PPO (rsl_rl) ── a_t ──► ManagerBasedRLEnv.step(a_t)

 1. action_manager.process_action(a_t)                                    [1 lần cho cả bước]
      CascadeAction.process_actions: kẹp [−1,1] → history.push(a_t) → v* = PositionLayer.output(a_t, c_t, s)
                                     → lưu velocity_at_step_start

 2. lặp 10 tick (tick = 0..9):
      action_manager.apply_action()
        CascadeAction.apply_actions:
          s = FlightState.of(...)                       ← MỘT ảnh chụp trạng thái, dùng chung cho các lớp trong tick này
          tick 0:   velocity.step(s, v*)  → a*          (period 5)
          tick 0:   attitude.step(s, a*)  → (ω*, T*)    (period 2)
          tick 0:   rate.step(s, (ω*, T*)) → u          (period 1)
          tick 1:   chỉ rate chạy (velocity, attitude giữ đầu ra cũ)
          tick 2:   attitude chạy lại, rồi rate
          ...
          tick 5:   velocity chạy lại (period 5), attitude không (5 lẻ), rate chạy
          mọi tick: Propulsion.step(u · 65535): PWM → lực mỗi động cơ → wrench lên thân
      scene.write_data_to_sim();  sim.step();  scene.update(2 ms)

 3. episode_length_buf += 1;  common_step_counter += 1

 4. termination_manager.compute()          left_flight_envelope, time_out

 5. reward_manager.compute(20 ms)          so trạng thái s_{t+1} với lệnh c_t; mỗi term nhân 20 ms

 6. reset các env đã xong:
      scene.reset → event reset (tư thế mới) → reset obs/action/reward/command/event/termination
        CascadeAction.reset:  history 0, lớp đông cứng reset, propulsion reset, _motor = hover
        LayerCommand.reset:   PID 0, target mới, ghi metric

 7. command_manager.compute(20 ms)         LayerCommand: ghi metric, rút target mới khi hết hạn, _update_command → c_{t+1}

 8. event interval (không dùng)

 9. observation_manager.compute()          layer_observation(s_{t+1}, c_{t+1}, history)

PPO ◄── obs, reward, done
```

Lịch các lớp trong 10 tick (bảng ở mục 6.1): velocity chạy ở tick 0 và 5; attitude ở 0, 2, 4, 6, 8; rate ở mọi tick.
Trong cùng một tick thứ tự luôn là velocity → attitude → rate, và cả ba dùng **cùng một ảnh chụp trạng thái**.

Với task khác: attitude (decimation 2) có 2 tick mỗi bước, velocity (5) có 5 tick, rate (1) có 1 tick. Bước env càng ngắn thì policy của lớp đó
được hỏi càng thường, và số lớp đông cứng bên dưới càng ít.

Checklist theo thời gian:
- [ ] a_t được chấm điểm với c_t (bước 5 trước bước 7). Đổi thứ tự sẽ phạt policy vì một lệnh nó chưa thấy khi chọn hành động.
- [ ] Env vừa reset ở bước 6 nhận lệnh mới ngay ở bước 7 (tính từ trạng thái mới), và quan sát ở bước 9 thấy lệnh đó.
- [ ] Ở `env.reset()` đầu tiên (không có bước 7), lệnh được tính trong `LayerCommand.reset`.
- [ ] Các lớp đông cứng luôn chạy từ trên xuống, trên cùng một ảnh chụp trạng thái trong một tick.
- [ ] Lệnh động cơ giữ nguyên giữa các lần rate chạy (với rate đông cứng ở 500 Hz thì nó chạy mọi tick).
- [ ] Task rate: decimation 1, không có lớp đông cứng, `_motor = _output` ở mọi tick.
- [ ] `_tick` không bao giờ reset: nhờ đó lớp chậm hơn bước env (attitude 250 Hz dưới velocity decimation 5) vẫn chạy đúng 250 Hz. Xem mục 6.1.
- [ ] Không có chỗ nào gọi `sim.step` ngoài vòng decimation.
- [ ] Không còn trễ động cơ: lực đặt ở tick k tác động ngay trong `sim.step` của tick k.

### 9.2 Bất biến xuyên suốt (mỗi điều phải đúng ở MỌI nơi)

Đây là danh sách các chỗ mà **hai file khác nhau phải đồng ý với nhau**. Sửa một nơi mà quên nơi kia thì không có lỗi báo, chỉ có hành vi sai.

| Bất biến | Những chỗ phải đồng nhất |
|---|---|
| Một hàm observe cho train và frozen | `layers.py` ↔ `observations.layer_observation` ↔ `cascade_action.FrozenLayer.step` |
| Một hàm output cho train và frozen | `layers.py` ↔ `CascadeAction.process_actions` ↔ `FrozenLayer.step` |
| Lịch sử: mới nhất trước, đẩy sau khi kẹp | `History.push` ở cả hai đường |
| Giới hạn đầu ra ↔ offset của lớp dưới | `layers.py` (scale) ↔ `*_env_cfg.py` (offset) ↔ `pid_control` (out_limit) |
| Kích thước obs 23 / 18 / 15 / 12, action 4 / 4 / 3 / 3 | `layers.py` ↔ file `.pt` ↔ `load_frozen` |
| Quy ước mô-men | `mixer.wrench_matrix` ↔ `Propulsion.step` bước 4 |
| Đường cong lực đẩy (tổng 4 động cơ, gram) | `uav_cfg.CF_THRUST_COEF_G` ↔ `pwm_to_thrust` ↔ `thrust_to_pwm` ↔ `RateLayer.output` ↔ `CascadePID.to_pwm` ↔ `PIDRateLayer.step` |
| Bắt đầu episode ở hover | `CascadeAction.reset` ↔ `MotorAction.reset` (lệnh động cơ = hover throttle) |
| Tần số | `constants.py` (`RATE_HZ`...) ↔ `Layer.hz` ↔ decimation ↔ `FrozenLayer.period` ↔ `PIDRateLayer.period` |
| Tên experiment | `freeze.experiment_name` ↔ `agents/*_ppo_cfg.py` |

### 9.3 Quy trình train → freeze → lớp trên (logic, không chạy)

- [ ] Train rate → `freeze --layer rate` → attitude đọc `frozen/rate.pt` → … → position đọc 3 file.
- [ ] Task `Attitude-PIDRate` bỏ qua bước đầu: rate là PID, không cần `rate.pt`. Dùng để train attitude mà không phụ thuộc vào rate đã train.
- [ ] Sửa bất cứ thứ gì ở `uav_cfg.py`, `propulsion.py`, `layers.py`, hằng số tần số, hoặc phân bố lệnh của một task → train lại **từ lớp thấp nhất bị ảnh hưởng** và freeze lại mọi lớp phía trên.
- [ ] `logs/console/pipeline.sh` làm chuỗi này tự động (đọc để kiểm điều kiện dừng: độ dài episode cuối < 80 % tối đa).

---

## Cấp 10 — Test (★★)

| File | Bảo vệ | Không bảo vệ |
|---|---|---|
| `test_registration.py` | hằng số nhất quán; task landing đăng ký | 4 task cascade chỉ gián tiếp |
| `test_mixer.py` | bố cục ma trận, nghịch đảo, tái tạo wrench | dấu so với `Propulsion.step` (chỉ cùng quy ước bằng mắt) |
| `test_pid.py` | P, I + reset, không đá D, giới hạn | |
| `test_cascade.py` | hover → hover throttle; target phía trước → chúi mũi; gọi từng lớp | PID ở 50 Hz |
| `test_aruco.py` | tìm đúng vị trí/kích thước; sàn trống không tìm thấy | ảnh render thật |
| `test_rl_layers.py` | kích thước mọi lớp; hover; FrozenPolicy = actor rsl_rl; freeze ghi được; **khung nghiêng attitude; khớp train/frozen; lịch tick** | ý nghĩa file `.pt` (stale); phân bố lệnh giữa các tầng |

Checklist:
- [ ] Chạy `$PY -m pytest tests -q` → 21 passed.
- [ ] Đọc 3 test mới (`test_rl_layers.py:64-148`), đặc biệt test lịch tick dựng `CascadeAction` bằng `__new__` và stub.

Những gì chỉ kiểm được khi chạy mô phỏng (bạn tự quyết khi nào chạy):
- Video train của từng lớp: drone có bám lệnh không, có lật khi offset rate lớn không.
- TensorBoard: `Metrics/layer/error` (giờ là trung bình episode), `Episode_Reward/*`, `Episode_Termination/*`.
- Phản hồi PID ở 50 Hz trong `LayerCommand`.

---

## Phụ lục A — File ngoài luồng chính

| File | Tình trạng |
|---|---|
| `src/Drone_RL/uav/tools/summarize_training_run.py` | script tóm tắt TensorBoard cho đồ án; docstring còn đường dẫn cũ `sim/uav/tools/...` và tên experiment cũ (`uav_visual_track`) |
| `src/Drone_RL/ui_extension_example.py` | template extension của Isaac Lab, không dùng |
| `src/Drone_RL/assets/__init__.py` | `DRONE_RL_ASSETS_DIR` |
| `logs/console/pipeline.sh`, `pipeline_status.txt`, `HANDOFF.md` | chuỗi train tự động và nhật ký |
| `config/extension.toml`, `pyproject.toml` | đóng gói |

## Phụ lục B — Vấn đề mở (không phải lỗi code, cần quyết định hoặc đo)

1. Cả 4 file frozen stale → train lại từ rate (xem 0.4).
2. PID sinh lệnh chạy ở 50/100 Hz thay vì 500 Hz lúc tune (3.3).
3. Hằng số trễ động cơ, khối lượng cánh, nhiễu IMU là giả định (`guide/01`, mục 1.3).
4. Trạng thái là trạng thái thật của simulator, không có ước lượng / nhiễu cảm biến (`guide/05` mục 5.8).
5. `load_frozen` chỉ kiểm kích thước: có thể thêm một "chữ ký" (scale, feedforward, offset) vào file `.pt` để phát hiện
   file cũ. Chưa làm.

## Phụ lục C — Bảng theo dõi

| Cấp | Mục | Xong | Ghi chú |
|---|---|---|---|
| 0 | quy ước, bản đồ, trạng thái | [ ] | |
| 1.1 | `uav_cfg.py` | [ ] | |
| 1.2 | `constants.py` | [ ] | |
| 1.3 | `mixer.py` | [ ] | |
| 1.4 | `pid.py` | [ ] | |
| 2.1 | hàm propulsion | [ ] | |
| 2.2 | `Propulsion` | [ ] | |
| 2.3 | `motor.py` | [ ] | |
| 2.4 | `MotorAction` | [ ] | |
| 3.1 | rate PID | [ ] | |
| 3.2 | attitude PID | [ ] | |
| 3.3 | velocity, position PID | [ ] | |
| 3.4 | `CascadePID` | [ ] | |
| 3.5 | script mô phỏng | [ ] | |
| 4.1–4.5 | `layers.py` | [ ] | |
| 5.1 | `frozen_policy.py` | [ ] | |
| 5.2 | `freeze.py` | [ ] | |
| 6.1 | `cascade_action.py` | [ ] | |
| 6.2 | `commands.py` | [ ] | |
| 6.3 | `observations.py` | [ ] | |
| 6.4 | `rewards.py` | [ ] | |
| 6.5 | `terminations.py` | [ ] | |
| 7.1 | `cascade_env_cfg.py` | [ ] | |
| 7.2 | 4 task | [ ] | |
| 7.3 | PPO | [ ] | |
| 7.4 | đăng ký | [ ] | |
| 8 | landing | [ ] | |
| 9.1 | một bước env | [ ] | |
| 9.2 | bất biến | [ ] | |
| 9.3 | quy trình train/freeze | [ ] | |
| 10 | test | [ ] | |

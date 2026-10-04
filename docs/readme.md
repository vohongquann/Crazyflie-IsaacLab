# docs: sources (local only)

This folder holds the documents the numbers of the simulation come from. It is listed in `.gitignore` (the PDFs are
not ours to publish). The documentation itself is [guide/](../guide/readme.md); every value taken from these files is
listed with its source in [guide/01_drone.md](../guide/01_drone.md).

| File | Used for |
|---|---|
| `hardware/crazyflie-brushless/bitcraze_crazyflie_2.1_brushless_datasheet_rev3.pdf` | frame size, mass, motors (guide 1.2) |
| `hardware/crazyflie-brushless/folk_2026_real_time_local_wind_inference_for_robust_autonomous_navigation.pdf` | motor thrust curve, air drag (guide 1.2, 2.1) |
| `hardware/crazyflie-brushless/busetto_2025_nonlinear_system_identification_nano-drone_benchmark.pdf` | inertia, yaw torque ratio $k_M/k_F$ (guide 1.2, 2.2) |
| `hardware/crazyflie-brushless/eschmann_2024_data-driven_system_identification_of_quadrotors_subject_to_motor_delays.pdf` | state list of the landing observation (guide 6.4) |
| `hardware/crazyflie-brushless/ullah_2026_nanobench_multi-task_benchmark_dataset_for_nano-quadrotors.pdf` | thrust-to-weight cross-check |
| `hardware/crazyflie2.1_revb.pdf` | Crazyflie 2.1 Rev B schematic |
| `hardware/papers-crazyflie-modelling.pdf` | Crazyflie modelling thesis (background) |

---

# Ghi chú nghiên cứu: flow tầng rate, thực tế và mô phỏng

## 0. Ba loại đại lượng

Mọi ô trong các sơ đồ dưới đây thuộc một trong ba loại (màu trong sơ đồ: xanh dương, xanh lá, cam):

| Nhãn | Nghĩa | Ví dụ |
|---|---|---|
| **[ĐO]** (xanh dương) | cảm biến đo được | $\boldsymbol\omega$ (gyro) |
| **[TÍNH]** (xanh lá) | bộ điều khiển tự tính ra, là *lệnh* (có dấu $^*$) | $\boldsymbol\omega^*$, $\boldsymbol\tau^*$, $F_i^*$, PWM |
| **[VẬT LÝ]** (cam) | xảy ra thật, không ai tính (trong mô phỏng thì code tính thay) | lực $F_i$, mô-men $\boldsymbol\tau$, chuyển động |

- $\boldsymbol\tau^*$ là mô-men bộ điều khiển **muốn**. $\boldsymbol\tau$ là mô-men **thật** tác dụng lên thân. Không có cảm biến đo $\boldsymbol\tau$.
- Nếu mô hình đúng thì $\boldsymbol\tau = \boldsymbol\tau^*$. Nếu sai thì gyro thấy $\boldsymbol\omega$ chưa đạt và PID bù ở vòng sau.

## 1. Bức tranh chung: hai nửa, hai chiều

Màu: xanh dương = [ĐO], xanh lá = [TÍNH], cam = [VẬT LÝ].

```mermaid
flowchart LR
    subgraph CTRL["BỘ ĐIỀU KHIỂN: giống hệt nhau ở thực tế và mô phỏng"]
        direction LR
        CMD["ω*, T*<br/>từ tầng attitude"] --> TAU["τ*<br/>mô-men MUỐN"]
        TAU --> MIX["mixer NGHỊCH ĐẢO<br/>[T*, τ*] → F1*..F4*"]
        MIX --> PWM["PWM 1..4"]
    end
    subgraph PHYS["VẬT LÝ: thực tế tự xảy ra / mô phỏng do Propulsion + PhysX tính"]
        direction LR
        F["lực thật F1..F4"] --> FWD["cộng THUẬN<br/>F_i → [T, τ]"]
        FWD --> MOVE["thân drone quay"]
    end
    PWM --> F
    MOVE --> GYRO["gyro đo ω"]
    GYRO -->|"so với ω*"| TAU

    classDef measure fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    classDef compute fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef physics fill:#ffedd5,stroke:#c2410c,color:#431407
    class GYRO measure
    class CMD,TAU,MIX,PWM compute
    class F,FWD,MOVE physics
```

- **Nghịch đảo** ($[T^*, \boldsymbol\tau^*] \to F_i^*$, `force_allocation_inverse`): bộ điều khiển nghĩ theo "thân cần mô-men bao
  nhiêu", motor chỉ hiểu lực/PWM của riêng nó, nên phải đổi ngược.
- **Thuận** ($F_i \to [T, \boldsymbol\tau]$, công thức trong `Propulsion.step`): 4 lực cộng lại trên khung. Ngoài đời không ai tính
  bước này, vật lý tự làm. Chỉ mô phỏng mới phải viết nó thành code.

## 2. Trường hợp 1: rate do PID

### 2a. Thực tế (Crazyflie bay thật)

```mermaid
flowchart TD
    ATT["tầng attitude<br/>ω*, T*"] --> ERR
    GYRO["gyro BMI088<br/>đo ω"] --> ERR["e = ω* − ω"]
    subgraph CHIP["CHIP STM32: bộ điều khiển"]
        ERR --> PID["PID rate<br/>α* = Kp·e + Kd·de/dt<br/>gia tốc góc muốn"]
        PID --> TAU["τ* = J·α*<br/>J = quán tính"]
        TAU --> MIX["mixer nghịch đảo<br/>[T*, τ*] → F1*..F4*"]
        MIX --> PWM["thrust_to_pwm<br/>F_i* → PWM_i"]
    end
    subgraph HW["PHẦN CỨNG + THẾ GIỚI: không ai tính"]
        PWM --> MOTOR["ESC + motor + cánh quạt<br/>→ lực thật F_i<br/>trễ motor, pin yếu, cánh lệch → F_i ≠ F_i*"]
        MOTOR --> SUM["4 lực trên khung → T, τ thật"]
        SUM --> BODY["thân quay<br/>J·dω/dt + ω×Jω = τ"]
    end
    BODY -->|"ω mới"| GYRO

    classDef measure fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    classDef compute fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef physics fill:#ffedd5,stroke:#c2410c,color:#431407
    class GYRO measure
    class ATT,ERR,PID,TAU,MIX,PWM compute
    class MOTOR,SUM,BODY physics
```

Firmware gốc của Crazyflie làm gọn hơn: PID rate ra thẳng "lệnh roll/pitch/yaw" theo đơn vị PWM, mixer chỉ cộng trừ
(m1 = thrust − roll + pitch + yaw ...). $J$ và hệ số lực đẩy đã nằm sẵn trong gain. Project này tách $J$ và mixer ra theo
đơn vị vật lý (N, N·m), nhưng ý tưởng giống hệt.

### 2b. Mô phỏng (Isaac Lab)

```mermaid
flowchart TD
    ATT["tầng attitude (PID hoặc policy RL)<br/>ω*, T*"] --> ERR
    READ["robot.data.root_ang_vel_b<br/>ω đọc thẳng từ PhysX, không nhiễu"] --> ERR["e = ω* − ω"]
    subgraph CTRL["BỘ ĐIỀU KHIỂN: code y như sẽ chạy trên chip"]
        ERR --> TAU["RateController.update<br/>τ* = J·PID(e)<br/>pid_control/rate.py"]
        TAU --> MIX["[T*, τ*] @ force_allocation_inverse → F_i*<br/>mdp/actions/mixer.py"]
        MIX --> PWM["thrust_to_pwm → PWM_i<br/>mdp/actions/propulsion.py"]
    end
    subgraph SIM["VẬT LÝ GIẢ: code tính thay thế giới, mỗi 2 ms"]
        PWM --> P1["Propulsion.step 1<br/>pwm_to_thrust × độ khỏe motor → F_i"]
        P1 --> P3["Propulsion.step 2<br/>nhiễu lực đẩy (tùy chọn)"]
        P3 --> P4["Propulsion.step 3–4: THUẬN<br/>T = ΣF_i, τx = ΣF_i·y_i<br/>τy = −ΣF_i·x_i, τz = km·Σs_i·F_i"]
        P4 --> PHYSX["permanent_wrench_composer → PhysX<br/>sim.step: tích phân 2 ms"]
    end
    PHYSX -->|"pose, ω mới"| READ

    classDef measure fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    classDef compute fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef physics fill:#ffedd5,stroke:#c2410c,color:#431407
    class READ measure
    class ATT,ERR,TAU,MIX,PWM compute
    class P1,P3,P4,PHYSX physics
```

Trong code: cascade PID (`pid_control/cascade.py`, `pid_hover_test.py`) và task `Isaac-UAV-Attitude-PIDRate-RL-v0`
(`PIDRateLayer` trong `mdp/actions/cascade_action.py`).

## 3. Trường hợp 2: rate do model (policy RL)

Mạng neural thay **cả cụm** "PID rate → τ* → mixer → thrust_to_pwm". Nó ra thẳng 4 lệnh motor, không có $\boldsymbol\tau^*$, không có mixer.

### 3a. Thực tế

```mermaid
flowchart TD
    ATT["tầng attitude<br/>ω*, T*"] --> OBS
    GYRO["gyro BMI088<br/>đo ω"] --> OBS["observation<br/>ω* − ω, ω, T*/mg, 4 output trước"]
    subgraph CHIP["CHIP STM32: bộ điều khiển"]
        OBS --> NET["mạng 64×64<br/>→ a1..a4 ∈ [−1, 1]"]
        NET --> U["u_i = clip(u_hover(T*/4) + 0.2·a_i, 0, 1)<br/>PWM_i = u_i × 65535"]
    end
    subgraph HW["PHẦN CỨNG + THẾ GIỚI"]
        U --> MOTOR["motor thật → F_i"]
        MOTOR --> BODY["T, τ thật → thân quay"]
    end
    BODY -->|"ω mới"| GYRO

    classDef measure fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    classDef compute fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef physics fill:#ffedd5,stroke:#c2410c,color:#431407
    class GYRO measure
    class ATT,OBS,NET,U compute
    class MOTOR,BODY physics
```

### 3b. Mô phỏng

```mermaid
flowchart TD
    ATT["tầng attitude<br/>ω*, T*"] --> OBS
    READ["robot.data.root_ang_vel_b<br/>ω từ PhysX"] --> OBS["RateLayer.observe<br/>mdp/layers.py"]
    subgraph CTRL["BỘ ĐIỀU KHIỂN"]
        OBS --> NET["FrozenPolicy (frozen/rate.pt)<br/>→ a1..a4"]
        NET --> U["RateLayer.output → u_i → PWM_i"]
    end
    subgraph SIM["VẬT LÝ GIẢ, mỗi 2 ms"]
        U --> PROP["Propulsion.step<br/>y hệt 2b: thrust, nhiễu, THUẬN"]
        PROP --> PHYSX["PhysX sim.step"]
    end
    PHYSX -->|"ω mới"| READ

    classDef measure fill:#dbeafe,stroke:#1d4ed8,color:#0b1b3f
    classDef compute fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef physics fill:#ffedd5,stroke:#c2410c,color:#431407
    class READ measure
    class ATT,OBS,NET,U compute
    class PROP,PHYSX physics
```

Mạng đã **tự học luôn phần mixer**: trong lúc train nó thử 4 lệnh motor, Propulsion + PhysX cho thấy drone quay ra sao,
và reward dạy nó tổ hợp nào cho ra đúng $\boldsymbol\omega^*$.

## 4. Bảng đối chiếu từng bước

| Bước | Thực tế | Mô phỏng | Rate PID | Rate RL |
|---|---|---|---|---|
| Đo $\boldsymbol\omega$ | gyro BMI088 (nhiễu, rung) | `robot.data.root_ang_vel_b` (PhysX, không nhiễu) | có | có |
| Ra $\boldsymbol\omega^*, T^*$ | tầng attitude trên chip | tầng attitude (PID hoặc RL) | có | có |
| $\boldsymbol\tau^* = J\cdot\mathrm{PID}(e)$ | firmware | `RateController.update` | có | **không** |
| Mixer nghịch đảo $[T^*,\boldsymbol\tau^*] \to F_i^*$ | firmware (power distribution) | `force_allocation_inverse` | có | **không** (mạng tự học) |
| Ra PWM | firmware | `thrust_to_pwm` / `RateLayer.output` | có | có |
| PWM → lực $F_i$ | motor, cánh quạt thật | `Propulsion.step` bước 1–2 | có | có |
| Thuận $F_i \to T, \boldsymbol\tau$ | vật lý tự làm | `Propulsion.step` bước 3–4 | có | có |
| $T, \boldsymbol\tau$ → chuyển động | định luật Newton | PhysX (`sim.step`, 2 ms) | có | có |

**Tóm lại:**
- Khung **bộ điều khiển / chip** (ô xanh lá) giống nhau giữa thực tế và mô phỏng. Đó chính là thứ đem lên drone.
- Khung **vật lý** (ô cam) là thế giới. Ngoài đời thì tự xảy ra. Trong mô phỏng thì `Propulsion` + PhysX bắt chước nó, và mô phỏng
  càng giống thật thì policy/gain mang sang drone thật càng ít phải chỉnh.
- Mixer **nghịch đảo** thuộc bộ điều khiển (chỉ có khi rate là PID). Công thức **thuận** thuộc vật lý (chỉ phải viết ra
  trong mô phỏng).



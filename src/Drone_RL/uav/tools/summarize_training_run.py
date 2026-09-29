#!/usr/bin/env python3
"""Tóm tắt một run huấn luyện thành số liệu + bản nháp nhận xét cho đồ án.

Script này làm đúng phần việc đã phải làm tay mỗi lần viết một mục "kết quả huấn
luyện" trong ``report_tools/report/thesis_vi``: mở file TensorBoard của run, đọc
thêm ``params/env.yaml`` và ``params/agent.yaml`` để biết cấu hình, rồi in ra ba
thứ dùng được ngay:

  1. Bảng mọi scalar đã ghi (giá trị đầu, cuối, cực đại, cực tiểu) để tra nhanh.
  2. Phân tích từng thành phần reward theo TRẦN của nó — xem giải thích ở
     ``ceiling_of_reward_term`` bên dưới, đây là phần cho biết nhiệm vụ nào policy
     học tốt và nhiệm vụ nào chưa.
  3. Một bảng LaTeX và một bản nháp nhận xét tiếng Việt, số liệu điền sẵn theo
     đúng quy ước dấu phẩy thập phân của đồ án.

Bản nháp nhận xét CHỈ là điểm khởi đầu: nó nêu đúng con số và những suy luận máy
móc chắc chắn đúng (hội tụ ở vòng lặp nào, thành phần nào chạm trần, episode kết
thúc vì lý do gì), còn phần diễn giải theo thiết kế bài toán thì vẫn phải tự viết.

Ví dụ dùng:

    # Run mới nhất của một experiment
    python3 sim/uav/tools/summarize_training_run.py --experiment uav_visual_track

    # Một run cụ thể, kèm xuất bảng LaTeX ra file
    python3 sim/uav/tools/summarize_training_run.py \\
        --run logs/rsl_rl/uav_attitude_tracking/2026-07-31_17-50-18 \\
        --latex-out /tmp/bang.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Tiền tố của ba nhóm scalar mà rsl_rl luôn ghi, tách riêng vì mỗi nhóm được
# diễn giải theo một cách khác nhau ở phần nhận xét bên dưới.
REWARD_PREFIX = "Episode_Reward/"
TERMINATION_PREFIX = "Episode_Termination/"
METRIC_PREFIX = "Metrics/"


class _ConfigLoader(yaml.SafeLoader):
    """Loader đọc được ``params/*.yaml`` do Isaac Lab ghi ra.

    Isaac Lab serialize config bằng tag ``!!python/tuple`` và một số tag
    ``!!python/...`` khác trỏ tới class thật; ``yaml.safe_load`` gặp chúng là ném
    ``ConstructorError`` và không đọc được gì. Ở đây tuple được dựng lại đúng kiểu,
    còn mọi tag python khác trả về ``None`` vì phần script cần (trọng số reward,
    độ dài episode, số môi trường) đều là số thuần.
    """


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)
_ConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def latest_run(log_root: Path, experiment: str) -> Path:
    """Trả run mới nhất của một experiment, so theo mtime của file tfevents."""
    experiment_dir = log_root / experiment
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Không có thư mục experiment: {experiment_dir}")
    runs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    runs = [path for path in runs if list(path.glob("events.out.tfevents.*"))]
    if not runs:
        raise FileNotFoundError(f"Không có run nào trong {experiment_dir}")
    return max(
        runs,
        key=lambda path: max(f.stat().st_mtime for f in path.glob("events.out.tfevents.*")),
    )


def load_config(run: Path) -> tuple[dict, dict]:
    """Đọc ``env.yaml`` và ``agent.yaml`` của run; thiếu file thì trả dict rỗng."""
    def _read(name: str) -> dict:
        path = run / "params" / name
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_ConfigLoader) or {}

    return _read("env.yaml"), _read("agent.yaml")


def vn(value: float, digits: int = 3) -> str:
    """Định dạng số theo quy ước đồ án tiếng Việt: dấu phẩy làm dấu thập phân."""
    return f"{value:.{digits}f}".replace(".", ",")


def reward_weights(env_cfg: dict) -> dict[str, float]:
    """Trọng số của từng term reward, lấy từ ``env.yaml``."""
    terms = env_cfg.get("rewards") or {}
    return {
        name: float(spec["weight"])
        for name, spec in terms.items()
        if isinstance(spec, dict) and spec.get("weight") is not None
    }


def ceiling_of_reward_term(weight: float) -> float | None:
    """Giá trị lớn nhất mà một term reward có thể đạt, hoặc ``None`` nếu không suy ra được.

    Mọi term thưởng trong repo này đều dạng nhân của một hàm mũ ``exp(-e²/std²)``
    với trọng số, mà hàm mũ cực đại bằng 1 khi sai số bằng 0 — nên trần của term
    chính là trọng số của nó. Nhờ vậy có thể quy tất cả về phần trăm và so trực
    tiếp mức học giữa các nhiệm vụ, thay vì nhìn con số tuyệt đối vốn không nói
    lên điều gì. Các term phạt (trọng số âm) không có trần theo nghĩa này: giá trị
    tốt nhất của chúng là 0, nên hàm trả ``None`` và phần nhận xét xử lý riêng.
    """
    return weight if weight > 0 else None


def max_episode_steps(env_cfg: dict) -> int | None:
    """Trần số bước của một episode, suy từ độ dài episode và nhịp policy.

    Cần con số này để biết ``Train/mean_episode_length`` đang ở mức bao nhiêu phần
    trăm — một episode ngắn hơn trần nghĩa là phần lớn episode bị kết thúc sớm.
    """
    episode_s = env_cfg.get("episode_length_s")
    decimation = env_cfg.get("decimation")
    sim_dt = (env_cfg.get("sim") or {}).get("dt")
    if not (episode_s and decimation and sim_dt):
        return None
    return int(round(episode_s / (sim_dt * decimation)))


def collect_scalars(run: Path) -> dict[str, list[float]]:
    """Đọc toàn bộ scalar của run thành ``{tag: [giá trị theo vòng lặp]}``."""
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [point.value for point in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def converged_iteration(values: list[float], tolerance: float = 0.02) -> int | None:
    """Vòng lặp đầu tiên mà giá trị đã vào trong ``tolerance`` quanh mức cuối.

    Dùng để nói "hội tụ từ khoảng vòng lặp thứ N" mà không phải ước lượng bằng mắt
    trên đồ thị. Ngưỡng mặc định 2% là đủ chặt để không báo sớm ở đoạn đang lên
    dốc, nhưng vẫn bỏ qua dao động nhỏ ở đoạn đi ngang.
    """
    if not values:
        return None
    final = values[-1]
    span = max(values) - min(values)
    if span == 0:
        return 0
    for index, value in enumerate(values):
        window = values[index:]
        if all(abs(v - final) <= tolerance * span for v in window):
            return index
    return None


def print_overview(scalars: dict[str, list[float]]) -> None:
    print("=" * 96)
    print("TOAN BO SCALAR")
    print("=" * 96)
    print(f"{'tag':<48}{'n':>6}{'dau':>12}{'cuoi':>12}{'max':>12}{'min':>12}")
    for tag in sorted(scalars):
        values = scalars[tag]
        if not values:
            continue
        print(
            f"{tag:<48}{len(values):>6}{values[0]:>12.4f}"
            f"{values[-1]:>12.4f}{max(values):>12.4f}{min(values):>12.4f}"
        )


def print_reward_analysis(scalars: dict[str, list[float]], weights: dict[str, float]) -> None:
    """In từng thành phần reward kèm phần trăm so với trần của chính nó."""
    print()
    print("=" * 96)
    print("PHAN TICH REWARD (gia tri cuoi so voi tran = trong so)")
    print("=" * 96)
    if not weights:
        print("Khong doc duoc trong so tu env.yaml — bo qua phan nay.")
        return

    print(f"{'term':<34}{'trong so':>10}{'cuoi':>12}{'tran':>10}{'%':>9}{'dat max o vong':>16}")
    positive_sum = 0.0
    positive_ceiling = 0.0
    for name, weight in sorted(weights.items()):
        tag = REWARD_PREFIX + name
        if tag not in scalars or not scalars[tag]:
            continue
        values = scalars[tag]
        final = values[-1]
        ceiling = ceiling_of_reward_term(weight)
        peak_at = values.index(max(values))
        if ceiling:
            positive_sum += final
            positive_ceiling += ceiling
            pct = f"{100.0 * final / ceiling:>8.1f}"
            ceiling_text = f"{ceiling:>10.3f}"
        else:
            pct = f"{'--':>8}"
            ceiling_text = f"{'phat':>10}"
        print(f"{name:<34}{weight:>10.3f}{final:>12.3f}{ceiling_text}{pct}{peak_at:>16}")

    if positive_ceiling:
        print("-" * 96)
        print(
            f"{'TONG cac term thuong':<34}{'':>10}{positive_sum:>12.3f}"
            f"{positive_ceiling:>10.3f}{100.0 * positive_sum / positive_ceiling:>9.1f}"
        )
        print(
            "Luu y: neu hai term duoc tach bang mask loai tru nhau (vd below/above hover) thi"
        )
        print(
            "chung CHIA CHUNG mot tran, nen phai cong lai roi so voi mot trong so, khong so rieng."
        )


def print_termination_analysis(scalars: dict[str, list[float]]) -> None:
    print()
    print("=" * 96)
    print("NGUYEN NHAN KET THUC EPISODE (ty le cuoi)")
    print("=" * 96)
    rows = [
        (tag[len(TERMINATION_PREFIX):], values[-1])
        for tag, values in scalars.items()
        if tag.startswith(TERMINATION_PREFIX) and values
    ]
    for name, final in sorted(rows, key=lambda item: -item[1]):
        print(f"  {name:<24}{final:>8.4f}   ({100.0 * final:.1f}%)")
    if rows:
        healthy = dict(rows).get("time_out", 0.0)
        if healthy < 0.9:
            print()
            print(
                f"  CANH BAO: chi {100.0 * healthy:.1f}% episode ket thuc do het thoi gian —"
            )
            print("  phan con lai la ket thuc som, phai giai thich nguyen nhan trong bao cao.")


def build_latex_table(
    scalars: dict[str, list[float]],
    weights: dict[str, float],
    steps_cap: int | None,
    label: str,
) -> str:
    """Sinh bảng LaTeX dán thẳng vào chương kết quả."""
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Giá trị cuối của các đại lượng huấn luyện.}",
        rf"  \label{{{label}}}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Đại lượng} & \textbf{Giá trị cuối} \\",
        r"    \midrule",
    ]

    def row(name: str, value: str) -> str:
        return f"    {name} & {value} \\\\"

    if "Train/mean_reward" in scalars:
        lines.append(row(r"Train/mean\_reward", vn(scalars["Train/mean_reward"][-1])))
    if "Train/mean_episode_length" in scalars:
        final = scalars["Train/mean_episode_length"][-1]
        text = vn(final, 2) + (f" / {steps_cap}" if steps_cap else "")
        lines.append(row(r"Train/mean\_episode\_length", text))

    for name in sorted(weights):
        tag = REWARD_PREFIX + name
        if tag in scalars and scalars[tag]:
            escaped = name.replace("_", r"\_")
            value = scalars[tag][-1]
            text = f"$-{vn(abs(value))}$" if value < 0 else vn(value)
            lines.append(row(rf"Episode\_Reward/{escaped}", text))

    for tag in sorted(scalars):
        if tag.startswith(METRIC_PREFIX) and scalars[tag]:
            escaped = tag.replace("_", r"\_")
            lines.append(row(escaped, vn(scalars[tag][-1])))

    for tag in sorted(scalars):
        if tag.startswith(TERMINATION_PREFIX) and scalars[tag]:
            escaped = tag.replace("_", r"\_")
            lines.append(row(escaped, vn(scalars[tag][-1])))

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_draft_comment(
    scalars: dict[str, list[float]],
    weights: dict[str, float],
    env_cfg: dict,
    steps_cap: int | None,
) -> str:
    """Sinh bản nháp nhận xét tiếng Việt với số liệu đã điền sẵn."""
    parts: list[str] = []

    num_envs = (env_cfg.get("scene") or {}).get("num_envs")
    episode_s = env_cfg.get("episode_length_s")
    reward = scalars.get("Train/mean_reward", [])
    length = scalars.get("Train/mean_episode_length", [])

    setup = []
    if num_envs:
        setup.append(f"{num_envs} môi trường song song")
    if episode_s:
        setup.append(f"mỗi episode dài {vn(float(episode_s), 0)[:-2] or episode_s}~s")
    if steps_cap:
        setup.append(f"tương ứng trần {steps_cap} bước điều khiển")
    if setup and reward:
        parts.append(
            f"Policy được huấn luyện trên {', '.join(setup)}. "
            f"Hình~\\ref{{fig:TODO}} tổng hợp diễn biến của {len(reward)} vòng lặp đã chạy."
        )

    if reward:
        converged = converged_iteration(reward)
        text = (
            f"Reward tổng tăng từ {vn(reward[0], 2)} lên {vn(reward[-1], 2)} "
            f"và đạt cực đại {vn(max(reward), 2)}"
        )
        if converged is not None:
            text += f", phần lớn mức tăng diễn ra trong khoảng {converged} vòng lặp đầu rồi đi ngang"
        parts.append(text + ".")

    if length:
        text = f"Độ dài episode trung bình đạt {vn(length[-1], 2)} bước"
        if steps_cap:
            text += f" trên trần {steps_cap} bước, tức {100.0 * length[-1] / steps_cap:.1f}\\%"
        parts.append(text + ".")

    bonus = [(n, w) for n, w in weights.items() if ceiling_of_reward_term(w)]
    for name, weight in sorted(bonus, key=lambda item: -item[1]):
        tag = REWARD_PREFIX + name
        if tag not in scalars or not scalars[tag]:
            continue
        final = scalars[tag][-1]
        pct = 100.0 * final / weight
        escaped = name.replace("_", r"\_")
        parts.append(
            f"Thành phần \\texttt{{{escaped}}} đạt {vn(final)} trên trần {vn(weight)}, "
            f"tức khoảng {pct:.0f}\\% mức tối đa."
        )

    terminations = {
        tag[len(TERMINATION_PREFIX):]: values[-1]
        for tag, values in scalars.items()
        if tag.startswith(TERMINATION_PREFIX) and values
    }
    if terminations:
        ranked = sorted(terminations.items(), key=lambda item: -item[1])
        listed = ", ".join(f"{name} {100.0 * value:.1f}\\%" for name, value in ranked if value > 0.001)
        parts.append(f"Về nguyên nhân kết thúc episode: {listed}.")
        if terminations.get("time_out", 0.0) < 0.9:
            parts.append(
                "Tỷ lệ kết thúc do hết thời gian chưa chiếm đa số, nghĩa là còn một phần "
                "đáng kể episode dừng sớm; cần giải thích nguyên nhân theo thiết kế bài toán "
                "thay vì chỉ nêu con số."
            )

    for tag, values in sorted(scalars.items()):
        if tag.startswith(METRIC_PREFIX) and values:
            escaped = tag.replace("_", r"\_")
            parts.append(
                f"Metric \\texttt{{{escaped}}}, được tính độc lập với reward, đi từ "
                f"{vn(values[0])} qua đỉnh {vn(max(values))} rồi về {vn(values[-1])} ở cuối."
            )

    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--experiment", help="Tên experiment; lấy run mới nhất của nó")
    parser.add_argument("--run", type=Path, help="Thư mục run cụ thể, ưu tiên hơn --experiment")
    parser.add_argument("--label", default="tab:training-scalars", help="Nhãn cho bảng LaTeX")
    parser.add_argument("--latex-out", type=Path, help="Ghi bảng LaTeX ra file thay vì chỉ in")
    args = parser.parse_args()

    if args.run:
        run = args.run.resolve()
    elif args.experiment:
        run = latest_run(args.log_root.resolve(), args.experiment)
    else:
        parser.error("Cần --run hoặc --experiment")

    scalars = collect_scalars(run)
    env_cfg, _ = load_config(run)
    weights = reward_weights(env_cfg)
    steps_cap = max_episode_steps(env_cfg)

    print(f"Run: {run}")
    iterations = len(scalars.get("Train/mean_reward", []))
    print(f"So vong lap da ghi: {iterations}")
    if steps_cap:
        print(f"Tran do dai episode: {steps_cap} buoc")
    print()

    print_overview(scalars)
    print_reward_analysis(scalars, weights)
    print_termination_analysis(scalars)

    table = build_latex_table(scalars, weights, steps_cap, args.label)
    print()
    print("=" * 96)
    print("BANG LATEX")
    print("=" * 96)
    print(table)
    if args.latex_out:
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        args.latex_out.write_text(table + "\n", encoding="utf-8")
        print(f"\nDa ghi bang ra: {args.latex_out}")

    print()
    print("=" * 96)
    print("BAN NHAP NHAN XET (can bien tap lai truoc khi dan vao do an)")
    print("=" * 96)
    print(build_draft_comment(scalars, weights, env_cfg, steps_cap))


if __name__ == "__main__":
    main()

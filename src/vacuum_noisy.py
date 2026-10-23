"""
توسعه تمرین جاروبرقی: چالش «حسگر نویزی»

تغییرات نسبت به vacuum_baseline.py (کد کلاس):
  1) حسگر وضعیت کثیفی اتاق فعلی را با احتمال P_ERROR اشتباه گزارش می‌کند.
     عامل‌ها دیگر dirt[pos] واقعی را نمی‌بینند، فقط خوانش حسگر را می‌بینند.
  2) نویز حسگر از یک مولد تصادفی جدا می‌آید تا محیط اولیه و رفتار تصادفی عامل‌ها
     با کد اصلی یکسان بماند (با P_ERROR=0 نتایج دقیقاً برابر کد کلاس است).
  3) عامل جدید "model_vote": مثل عامل مدل‌دار است ولی هر بار K_READS بار حسگر را
     می‌خواند و رأی اکثریت می‌گیرد. (همان ایده کاهش نویز با میانگین‌گیری در تصویر میوه)
  4) امتیاز همچنان با وضعیت واقعی محیط حساب می‌شود (معیار عملکرد = تمیزی واقعی).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_ROOMS = 4     # number of rooms
STEPS = 50      # steps in one run
RUNS = 100      # number of experiments per agent
K_READS = 5     # تعداد خوانش حسگر در عامل رأی‌گیر (فرد باشد)


def simulate(agent, seed, p_error=0.0):
    rng = np.random.default_rng(seed)
    sensor_rng = np.random.default_rng(seed + 10_000)   # مولد جدا برای نویز حسگر
    dirt = rng.random(N_ROOMS) < 0.5               # True = dirty room
    known_clean = np.zeros(N_ROOMS, dtype=bool)    # memory of the "model" agent
    pos, direction, score = 0, 1, 0

    def read_sensor():
        """خوانش نویزی: با احتمال p_error جواب برعکس می‌شود."""
        flip = sensor_rng.random() < p_error
        return bool(dirt[pos]) ^ flip

    for step in range(STEPS):
        if agent == "model_vote":
            reads = [read_sensor() for _ in range(K_READS)]
            sensed_dirty = sum(reads) > K_READS / 2     # رأی اکثریت
        else:
            sensed_dirty = read_sensor()
        known_clean[pos] = not sensed_dirty

        # 1) choose an action (فقط بر اساس ادراک نویزی)
        if agent == "random":
            action = rng.choice(["Left", "Right", "Suck"])
        elif sensed_dirty:
            action = "Suck"
        elif agent == "reflex":
            action = rng.choice(["Left", "Right"])
        elif known_clean.all():
            action = "NoOp"
        else:
            if pos == N_ROOMS - 1:
                direction = -1
            if pos == 0:
                direction = 1
            action = "Right" if direction == 1 else "Left"

        # 2) apply the action
        if action == "Suck":
            dirt[pos] = False
        elif action == "Right":
            pos = min(pos + 1, N_ROOMS - 1)
        elif action == "Left":
            pos = max(pos - 1, 0)

        score += (~dirt).sum()                     # +1 برای هر اتاق واقعاً تمیز
    return int(score)


AGENTS = ["random", "reflex", "model", "model_vote"]
P_ERRORS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

if __name__ == "__main__":
    # بررسی سازگاری با کد کلاس (p=0)
    for agent in ["random", "reflex", "model"]:
        print(agent, simulate(agent, seed=1, p_error=0.0))

    rows = []
    for p in P_ERRORS:
        for agent in AGENTS:
            for seed in range(RUNS):
                rows.append({"p_error": p, "agent": agent,
                             "score": simulate(agent, seed, p)})
    df = pd.DataFrame(rows)

    summary = df.groupby(["p_error", "agent"])["score"].agg(["mean", "std"]).round(1)
    print(summary.unstack("agent")["mean"])
    summary.to_csv("results/vacuum_noise_summary.csv")

    plt.figure(figsize=(6.5, 4.2))
    for agent in AGENTS:
        s = df[df.agent == agent].groupby("p_error")["score"].agg(["mean", "std"])
        plt.errorbar(s.index, s["mean"], s["std"] / np.sqrt(RUNS), marker="o", label=agent)
    plt.xlabel("Sensor error probability p_error")
    plt.ylabel("Mean score (over 100 runs)")
    plt.title("Vacuum agents with noisy sensor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/vacuum_noise.png", dpi=150)

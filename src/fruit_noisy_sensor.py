"""
تشخیص میوه با «حسگر نویزی» (توسعه ایده تمرین جاروبرقی)

در جاروبرقی: حسگر با احتمال p وضعیت کثیفی را اشتباه می‌گزارد.
در پروژه ما: دوربین (حسگر) تصویر میوه را با نویز می‌گیرد و مدل باید تشخیص دهد.

اگر پوشه دیتاست واقعی (مثلاً Fruits-360) موجود باشد از آن استفاده می‌شود:
    python src/fruit_noisy_sensor.py --data path/to/fruits   (هر کلاس = یک زیرپوشه)
در غیر این صورت تصاویر مصنوعی چهار میوه تولید می‌شود تا کد همیشه اجرا شود.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

SIZE = 32
CLASSES = ["apple", "banana", "orange", "grape"]


# ---------- داده ----------
def make_fruit(label, rng):
    """تصویر مصنوعی ۳۲x۳۲ رنگی با مقادیر ۰ تا ۱."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    img = np.ones((SIZE, SIZE, 3)) * 0.9
    cx, cy = SIZE / 2 + rng.integers(-3, 4, 2)
    j = rng.normal(0, 0.08, 3) + rng.normal(0, 0.05)  # تغییر رنگ و روشنایی
    if label == "apple":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < (rng.integers(9, 12)) ** 2
        color = np.array([0.85, 0.30, 0.12])
    elif label == "banana":
        mask = ((xx - cx) / 13) ** 2 + ((yy - cy) / 4) ** 2 < 1
        color = np.array([0.95, 0.85, 0.15])
    elif label == "orange":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < (rng.integers(9, 12)) ** 2
        color = np.array([0.92, 0.50, 0.10])
    else:  # grape: خوشه دایره‌های کوچک
        mask = np.zeros((SIZE, SIZE), bool)
        for _ in range(7):
            gx, gy = cx + rng.integers(-7, 8), cy + rng.integers(-7, 8)
            mask |= (xx - gx) ** 2 + (yy - gy) ** 2 < 16
        color = np.array([0.45, 0.1, 0.55])
    img[mask] = np.clip(color + j, 0, 1)
    return img


def load_synthetic(n_per_class=150, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(make_fruit(c, rng))
            y.append(k)
    return np.array(X), np.array(y)


def load_folder(path, per_class=150):
    from PIL import Image
    classes = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    X, y = [], []
    for k, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(path, c)))[:per_class]
        for f in files:
            im = Image.open(os.path.join(path, c, f)).convert("RGB").resize((SIZE, SIZE))
            X.append(np.asarray(im) / 255.0)
            y.append(k)
    return np.array(X), np.array(y), classes


# ---------- حسگر نویزی ----------
def noisy_sensor(X, sigma, rng):
    """نویز گاوسی با انحراف معیار sigma؛ معادل p_error در جاروبرقی."""
    return np.clip(X + rng.normal(0, sigma, X.shape), 0, 1)


def features(X):
    """ویژگی خام: پیکسل‌های فلت‌شده."""
    return X.reshape(len(X), -1)


def features_pooled(X, k=4):
    """بهبود: میانگین‌گیری بلوکی k×k (فیلتر کاهش نویز) و سپس فلت کردن."""
    n, h, w, c = X.shape
    return X.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4)).reshape(n, -1)


# ---------- آزمایش ----------
def experiment(X, y, sigmas, repeats=5, seed=0):
    """مقایسه مدل پایه (پیکسل خام) با مدل بهبودیافته (میانگین‌گیری بلوکی)."""
    rng = np.random.default_rng(seed)
    acc_raw = np.zeros((repeats, len(sigmas)))
    acc_pool = np.zeros((repeats, len(sigmas)))
    for r in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=r)
        m_raw = KNeighborsClassifier(3).fit(features(Xtr), ytr)
        m_pool = KNeighborsClassifier(3).fit(features_pooled(Xtr), ytr)
        for i, s in enumerate(sigmas):
            Xn = noisy_sensor(Xte, s, rng)   # فقط داده تست نویزی می‌شود
            acc_raw[r, i] = m_raw.score(features(Xn), yte)
            acc_pool[r, i] = m_pool.score(features_pooled(Xn), yte)
    return acc_raw, acc_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="پوشه دیتاست واقعی (اختیاری)")
    args = ap.parse_args()

    if args.data and os.path.isdir(args.data):
        X, y, names = load_folder(args.data)
        print("Real dataset:", len(X), "images,", len(names), "classes")
    else:
        X, y = load_synthetic()
        print("Synthetic dataset:", len(X), "images,", len(CLASSES), "classes")

    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    a_raw, a_pool = experiment(X, y, sigmas)

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.errorbar(sigmas, a_raw.mean(0), a_raw.std(0), marker="o", label="Baseline (raw pixels)")
    plt.errorbar(sigmas, a_pool.mean(0), a_pool.std(0), marker="s", label="With 4x4 block averaging")
    plt.xlabel("Noise level (sigma)")
    plt.ylabel("Accuracy")
    plt.title("Fruit recognition under noisy sensor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_noise.png", dpi=150)

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 4, figsize=(8, 2.4))
    for a, s in zip(ax, [0.0, 0.5, 1.0, 2.0]):
        a.imshow(noisy_sensor(X[:1], s, rng)[0]); a.set_title(f"sigma={s}"); a.axis("off")
    plt.tight_layout()
    plt.savefig("results/noise_examples.png", dpi=150)

    print("sigma | baseline | improved")
    for s, c, n in zip(sigmas, a_raw.mean(0), a_pool.mean(0)):
        print(f"{s:5.2f} | {c:8.3f} | {n:8.3f}")


if __name__ == "__main__":
    main()

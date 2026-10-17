#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课题 08 · 多姿态步态加速度特征采集与对比 —— 数据分析脚本

功能
----
1. 读取 Phyphox 导出的 4 份 CSV（静止 / 平地慢走 / 快跑 / 上下楼梯）
2. 分窗计算时域 + 频域特征
3. 输出：特征汇总表、时域波形图、特征箱线图、频谱对比图
4. 自动推导「区分四种动作」的阈值规则，生成 threshold_rules.md

用法
----
    # 用真实数据（把 CSV 放进 raw_data/，文件名带上状态关键字）
    python analyze_gait.py

    # 没有数据时先跑一遍演示，验证脚本能正常工作
    python analyze_gait.py --demo

文件名识别关键字（大小写不敏感，中英文都认）
    静止  : still / static / idle / 静止 / 站立
    慢走  : walk / slow / 慢走 / 步行
    快跑  : run / fast / 跑 / 快跑
    楼梯  : stair / updown / 楼梯 / 上下楼

输出目录
    ../images/     图片
    ../raw_data/   特征表 CSV
    ../images/threshold_rules.md  阈值规则
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------------
# 0. 中文字体
# ----------------------------------------------------------------------------
def setup_cn_font() -> None:
    """让 matplotlib 能显示中文。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "WenQuanYi Micro Hei", "PingFang SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[提示] 未找到中文字体，图中中文可能显示为方框。")
    plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------
# 1. 读取 Phyphox CSV
# ----------------------------------------------------------------------------
STATE_KEYWORDS = {
    "静止": ["still", "static", "idle", "静止", "站立", "停"],
    "慢走": ["walk", "slow", "慢走", "步行", "走路"],
    "快跑": ["run", "fast", "jog", "跑", "快跑"],
    "楼梯": ["stair", "updown", "up_down", "楼梯", "上下楼", "上楼"],
}
STATE_COLOR = {"静止": "#7f8c8d", "慢走": "#2980b9", "快跑": "#c0392b", "楼梯": "#e67e22"}
STATE_ORDER = ["静止", "慢走", "快跑", "楼梯"]
GRAVITY = 9.81


def guess_state(filename: str) -> str | None:
    low = filename.lower()
    for state, keys in STATE_KEYWORDS.items():
        if any(k.lower() in low for k in keys):
            return state
    return None


def load_phyphox_csv(path: Path) -> pd.DataFrame:
    """
    读取 Phyphox 导出的 CSV，返回统一列名：time, ax, ay, az

    Phyphox 的 CSV 有两种情况：
      - 新版本：前几行是 # 开头的元数据注释，然后是表头
      - 旧版本：直接是表头
    表头可能是中文（加速度 x (m/s²)）或英文（Acceleration x (m/s^2)）
    """
    # 跳过 # 注释行
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path.name}: 文件为空")

    header = [c.strip().strip('"') for c in lines[0].split(",")]
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]
    df = pd.DataFrame(rows, columns=header)
    df = df.apply(pd.to_numeric, errors="coerce")

    low = [c.lower() for c in header]

    def find(*keys: str) -> int | None:
        for i, col in enumerate(low):
            if all(k in col for k in keys):
                return i
        return None

    i_time = find("time") if find("time") is not None else find("时间")
    i_x = find("x (m/s")
    i_y = find("y (m/s")
    i_z = find("z (m/s")

    # 兜底：按位置推断（Phyphox 默认列序 Time, |a|, ax, ay, az）
    if i_time is None:
        i_time = 0
    if i_x is None or i_y is None or i_z is None:
        if len(header) >= 5:
            i_x, i_y, i_z = 2, 3, 4
        elif len(header) == 4:
            i_x, i_y, i_z = 1, 2, 3
        else:
            raise ValueError(f"{path.name}: 无法识别加速度列，表头为 {header}")

    out = pd.DataFrame({
        "time": df.iloc[:, i_time].to_numpy(dtype=float),
        "ax": df.iloc[:, i_x].to_numpy(dtype=float),
        "ay": df.iloc[:, i_y].to_numpy(dtype=float),
        "az": df.iloc[:, i_z].to_numpy(dtype=float),
    }).dropna()

    out = out.sort_values("time").reset_index(drop=True)
    if len(out) < 20:
        raise ValueError(f"{path.name}: 有效数据点太少（{len(out)} 行）")
    return out


# ----------------------------------------------------------------------------
# 2. 特征提取
# ----------------------------------------------------------------------------
def detrend(x: np.ndarray) -> np.ndarray:
    """去掉直流分量（重力），只留动态部分。"""
    return x - np.mean(x)


def dominant_frequency(dyn: np.ndarray, fs: float) -> float:
    """动态信号的频谱主峰频率（Hz），搜索范围 0.3–10 Hz。"""
    n = len(dyn)
    if n < 8:
        return float("nan")
    spec = np.abs(np.fft.rfft(dyn * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= 0.3) & (freqs <= 10.0)
    if not band.any():
        return float("nan")
    return float(freqs[band][np.argmax(spec[band])])


def zero_crossing_rate(dyn: np.ndarray) -> float:
    """过零率（次/秒）。"""
    if len(dyn) < 2:
        return 0.0
    signs = np.sign(dyn)
    signs[signs == 0] = 1
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    return float(crossings)


def windowed_features(df: pd.DataFrame, win_sec: float = 2.0, overlap: float = 0.5) -> pd.DataFrame:
    """按滑动窗切分，逐窗计算特征。"""
    t = df["time"].to_numpy()
    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()

    dt = np.median(np.diff(t))
    fs = 1.0 / dt if dt > 0 else 50.0

    win = max(int(win_sec * fs), 16)
    step = max(int(win * (1 - overlap)), 1)

    records = []
    for start in range(0, len(t) - win + 1, step):
        sl = slice(start, start + win)
        _ax, _ay, _az = ax[sl], ay[sl], az[sl]
        svm = np.sqrt(_ax**2 + _ay**2 + _az**2)
        dyn = detrend(svm)

        records.append({
            "svm_mean": float(np.mean(svm)),
            "svm_std": float(np.std(dyn, ddof=0)),
            "svm_max": float(np.max(svm)),
            "svm_p2p": float(np.max(svm) - np.min(svm)),
            "ax_std": float(np.std(_ax)),
            "ay_std": float(np.std(_ay)),
            "az_std": float(np.std(_az)),
            "ax_mean": float(np.mean(_ax)),
            "ay_mean": float(np.mean(_ay)),
            "az_mean": float(np.mean(_az)),
            "zcr": zero_crossing_rate(dyn) / win_sec,
            "dom_freq": dominant_frequency(dyn, fs),
            "cv": float(np.std(dyn) / np.mean(svm)) if np.mean(svm) else float("nan"),
            "energy": float(np.mean(dyn**2)),
        })

    out = pd.DataFrame(records)
    out["fs"] = fs
    return out


# ----------------------------------------------------------------------------
# 3. 演示数据（没有真实 CSV 时用，验证脚本可用）
# ----------------------------------------------------------------------------
def make_demo_data(out_dir: Path, dur: float = 60.0, fs: float = 100.0) -> list[Path]:
    """合成 4 种状态的加速度信号，写入 out_dir，返回文件路径列表。"""
    rng = np.random.default_rng(42)
    t = np.arange(0, dur, 1.0 / fs)
    paths = []

    specs = {
        "静止": dict(amp=0.05, freq=0.6, z_bias=0.0, noise=0.04, label="still"),
        "慢走": dict(amp=2.2, freq=1.8, z_bias=0.3, noise=0.9, label="walk"),
        "快跑": dict(amp=7.5, freq=3.0, z_bias=0.8, noise=2.2, label="run"),
        "楼梯": dict(amp=4.0, freq=1.2, z_bias=1.8, noise=1.4, label="stair"),
    }

    for state, sp in specs.items():
        w = 2 * np.pi * sp["freq"]
        # 竖直方向以 2 倍频为主（每步两次冲击），横向以基频为主
        az = GRAVITY + sp["amp"] * np.sin(2 * w * t) + rng.normal(0, sp["noise"], len(t))
        ax = sp["amp"] * 0.55 * np.sin(w * t) + rng.normal(0, sp["noise"] * 0.6, len(t))
        ay = sp["amp"] * 0.35 * np.sin(w * t + 0.7) + sp["z_bias"] + rng.normal(0, sp["noise"] * 0.6, len(t))

        svm = np.sqrt(ax**2 + ay**2 + az**2)
        f = out_dir / f"demo_{sp['label']}.csv"
        pd.DataFrame({
            "Time (s)": t,
            "Absolute acceleration (m/s^2)": svm,
            "Acceleration x (m/s^2)": ax,
            "Acceleration y (m/s^2)": ay,
            "Acceleration z (m/s^2)": az,
        }).to_csv(f, index=False, float_format="%.4f")
        paths.append(f)

    print(f"[演示] 已生成 4 份合成 CSV 到 {out_dir}")
    return paths


# ----------------------------------------------------------------------------
# 4. 出图
# ----------------------------------------------------------------------------
def plot_waveforms(data: dict[str, pd.DataFrame], out: Path, sec: float = 8.0) -> None:
    """四状态时域波形对比（各截取前 sec 秒）。"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, state in zip(axes, STATE_ORDER):
        df = data[state]
        t = df["time"].to_numpy()
        m = t <= t[0] + sec
        ax.plot(t[m] - t[0], df["ax"].to_numpy()[m], lw=0.8, label="ax", alpha=0.85)
        ax.plot(t[m] - t[0], df["ay"].to_numpy()[m], lw=0.8, label="ay", alpha=0.85)
        ax.plot(t[m] - t[0], df["az"].to_numpy()[m], lw=0.8, label="az", alpha=0.85)
        ax.set_ylabel("加速度 (m/s²)", fontsize=9)
        ax.set_title(f"{state}　（前 {sec:.0f} 秒）", fontsize=11, loc="left",
                     color=STATE_COLOR[state], fontweight="bold")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("四种运动状态的原始三轴加速度波形", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out.name}")


def plot_feature_boxes(feat: pd.DataFrame, out: Path) -> None:
    """关键特征的分布对比。"""
    keys = [("svm_std", "合加速度标准差\n(动态强度)"),
            ("dom_freq", "主频 (Hz)\n(步频)"),
            ("svm_max", "合加速度峰值\n(m/s²)"),
            ("cv", "变异系数\n(波动占比)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (col, title) in zip(axes, keys):
        groups = [feat.loc[feat["state"] == s, col].dropna().to_numpy() for s in STATE_ORDER]
        bp = ax.boxplot(groups, tick_labels=STATE_ORDER, patch_artist=True, widths=0.6)
        for patch, s in zip(bp["boxes"], STATE_ORDER):
            patch.set_facecolor(STATE_COLOR[s])
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.4)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=9)
    fig.suptitle("四种状态的特征分布对比（每箱 = 一个 2 秒窗口）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out.name}")


def plot_spectrum(data: dict[str, pd.DataFrame], out: Path) -> None:
    """四状态频谱对比，标注主频。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, state in zip(axes.ravel(), STATE_ORDER):
        df = data[state]
        svm = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2).to_numpy()
        dyn = detrend(svm)
        dt = np.median(np.diff(df["time"].to_numpy()))
        fs = 1.0 / dt
        spec = np.abs(np.fft.rfft(dyn * np.hanning(len(dyn)))) / len(dyn)
        freqs = np.fft.rfftfreq(len(dyn), d=1.0 / fs)
        m = freqs <= 10
        ax.plot(freqs[m], spec[m], color=STATE_COLOR[state], lw=1.1)
        pk = dominant_frequency(dyn, fs)
        if np.isfinite(pk):
            ax.axvline(pk, ls="--", lw=1, color="black", alpha=0.6)
            ax.annotate(f"主频 {pk:.2f} Hz", xy=(pk, ax.get_ylim()[1] * 0.82),
                        fontsize=9, ha="left", xytext=(pk + 0.25, ax.get_ylim()[1] * 0.82))
        ax.set_title(state, fontsize=11, color=STATE_COLOR[state], fontweight="bold")
        ax.set_xlabel("频率 (Hz)", fontsize=9)
        ax.set_ylabel("幅值", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("合加速度动态分量的频谱对比", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out.name}")


# ----------------------------------------------------------------------------
# 5. 阈值规则推导
# ----------------------------------------------------------------------------
def build_threshold_rules(feat: pd.DataFrame, out: Path) -> str:
    """
    对每个特征计算各状态的 [p10, p90] 区间，
    挑选「区间互不重叠」的特征作为阈值依据，生成规则文档。
    """
    feat_cols = ["svm_std", "dom_freq", "svm_max", "svm_p2p", "cv", "zcr", "energy"]

    lines = ["# 阈值规则（由数据自动推导）", "",
             "> 生成方式：对每份 CSV 按 2 秒窗、50% 重叠切分，逐窗提特征，",
             "> 再对每个状态取 [p10, p90] 区间作为该状态的典型取值范围。", "",
             "## 一、各状态特征取值范围", ""]

    header = "| 特征 | " + " | ".join(STATE_ORDER) + " | 可分离 |"
    lines += [header, "|" + "---|" * (len(STATE_ORDER) + 2)]

    separability: dict[str, bool] = {}
    for col in feat_cols:
        cells, ranges = [], []
        for s in STATE_ORDER:
            v = feat.loc[feat["state"] == s, col].dropna()
            if len(v) == 0:
                cells.append("—")
                ranges.append(None)
                continue
            lo, hi = np.percentile(v, [10, 90])
            cells.append(f"{lo:.2f} ~ {hi:.2f}")
            ranges.append((lo, hi))
        # 判断是否两两不重叠
        ok = all(r is not None for r in ranges)
        if ok:
            srt = sorted(ranges)
            for a, b in zip(srt, srt[1:]):
                if a[1] >= b[0]:
                    ok = False
                    break
        separability[col] = ok
        lines.append(f"| `{col}` | " + " | ".join(cells) + f" | {'✅' if ok else '—'} |")

    good = [c for c, v in separability.items() if v]
    lines += ["", "## 二、推荐判据", ""]

    if not good:
        lines += ["> 本次数据里没有特征能做到四类完全分离。",
                  "> 常见原因：采样时长太短、状态区分度不够、或手机放置位置不一致。",
                  "> 建议：延长每段采集到 60 秒以上，固定手机位置（如裤兜同一侧），再跑一次。"]
    else:
        lines.append(f"以下特征在各状态间的取值范围**互不重叠**，可直接作为阈值判据："
                     f" {'、'.join('`%s`' % c for c in good)}")
        lines += ["", "```text"]
        # 以区分度最好的两个特征构建决策链
        primary = good[0]
        lines.append(f"# 判据一：{primary}（{len(good)} 个可用特征中区分度最强）")
        order = sorted(STATE_ORDER, key=lambda s: feat.loc[feat["state"] == s, primary].mean())
        prev = None
        for s in order:
            v = feat.loc[feat["state"] == s, primary].dropna()
            lo, hi = np.percentile(v, [10, 90]) if len(v) else (np.nan, np.nan)
            if prev is not None:
                thr = (prev + lo) / 2
                lines.append(f"if {primary} > {thr:.2f}:  →  {s}")
            else:
                lines.append(f"if {primary} <= {(lo + hi) / 2:.2f}:  →  {s}")
            prev = hi
        lines += ["```", ""]

    lines += ["## 三、说明", "",
              "- 上表区间来自**自动统计**，不是教科书常数。换手机、换采集姿势、换人，数值都会变。",
              "- 论文/报告里写规则时，务必注明「基于本人 XX 型号手机、裤兜固定放置、N 段样本统计」。",
              "- 阈值取两段区间的**中点**是为了留最大安全边界；若某两类挨得很近，可改用其他特征交叉判断。",
              ""]

    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"[规则] {out.name}")
    return text


# ----------------------------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="课题08 步态加速度特征分析")
    ap.add_argument("--data-dir", default=None, help="CSV 所在目录，默认 ../raw_data")
    ap.add_argument("--out-images", default=None, help="图片输出目录，默认 ../images")
    ap.add_argument("--out-data", default=None, help="特征表输出目录，默认 ../raw_data")
    ap.add_argument("--window", type=float, default=2.0, help="分窗长度（秒），默认 2.0")
    ap.add_argument("--demo", action="store_true", help="用合成数据跑一遍，验证脚本")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    base = here.parent                       # 01-Field-Discovery
    data_dir = Path(args.data_dir) if args.data_dir else base / "raw_data"
    img_dir = Path(args.out_images) if args.out_images else base / "images"
    dat_dir = Path(args.out_data) if args.out_data else base / "raw_data"

    img_dir.mkdir(parents=True, exist_ok=True)
    dat_dir.mkdir(parents=True, exist_ok=True)
    setup_cn_font()

    # --- 找数据 ---
    if args.demo:
        # 演示模式：所有产物放进 tools/_demo，不污染 raw_data/
        demo_dir = here / "_demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        csv_files = make_demo_data(demo_dir)
        img_dir = dat_dir = demo_dir
    else:
        csv_files = sorted(data_dir.glob("*.csv"))
        csv_files = [f for f in csv_files
                     if not f.name.startswith("features") and "demo_" not in f.name]

    if not csv_files:
        print(f"[错误] {data_dir} 里没有找到 CSV 文件。")
        print("       提示：先用 python analyze_gait.py --demo 验证脚本；")
        print("       真实数据请从 Phyphox 导出后放进 raw_data/，文件名带状态关键字。")
        return 1

    # --- 读数据 ---
    data: dict[str, pd.DataFrame] = {}
    for f in csv_files:
        state = guess_state(f.name)
        if state is None:
            print(f"[跳过] {f.name}：文件名没有状态关键字（still/walk/run/stair 或 静止/慢走/快跑/楼梯）")
            continue
        if state in data:
            print(f"[跳过] {f.name}：{state} 已有数据")
            continue
        try:
            data[state] = load_phyphox_csv(f)
            print(f"[读取] {f.name} → {state}，{len(data[state])} 点")
        except Exception as e:                       # noqa: BLE001
            print(f"[错误] {f.name}: {e}")

    missing = [s for s in STATE_ORDER if s not in data]
    if missing:
        print(f"[警告] 缺少状态：{'、'.join(missing)}（对比分析会不完整）")
    if not data:
        return 1

    # --- 特征 ---
    frames = []
    for state, df in data.items():
        feat = windowed_features(df, win_sec=args.window)
        feat["state"] = state
        feat["source"] = state
        frames.append(feat)
    features = pd.concat(frames, ignore_index=True)

    feat_path = dat_dir / "features_windows.csv"
    features.to_csv(feat_path, index=False, float_format="%.4f")
    print(f"[表] {feat_path.name}（{len(features)} 个窗口）")

    cols = ["svm_mean", "svm_std", "svm_max", "svm_p2p", "cv",
            "dom_freq", "zcr", "energy", "ax_std", "ay_std", "az_std"]
    summary = features.groupby("state")[cols].agg(["mean", "std"]).round(3)
    sum_path = dat_dir / "features_summary.csv"
    summary.to_csv(sum_path)
    print(f"[表] {sum_path.name}")

    # --- 控制台摘要 ---
    print("\n" + "=" * 78)
    print("特征均值汇总（每行 = 一个状态）")
    print("=" * 78)
    show = ["svm_std", "svm_max", "dom_freq", "cv", "zcr"]
    head = f"{'状态':<6}" + "".join(f"{c:>12}" for c in show)
    print(head)
    print("-" * len(head))
    for s in STATE_ORDER:
        if s not in data:
            continue
        row = features[features["state"] == s]
        print(f"{s:<6}" + "".join(f"{row[c].mean():>12.3f}" for c in show))

    # --- 图 ---
    ordered = {s: data[s] for s in STATE_ORDER if s in data}
    plot_waveforms(ordered, img_dir / "fig1_waveforms.png")
    plot_feature_boxes(features, img_dir / "fig2_feature_boxes.png")
    plot_spectrum(ordered, img_dir / "fig3_spectrum.png")

    # --- 规则 ---
    build_threshold_rules(features, img_dir / "threshold_rules.md")

    print("\n完成。产物：")
    for p in [feat_path, sum_path,
              img_dir / "fig1_waveforms.png",
              img_dir / "fig2_feature_boxes.png",
              img_dir / "fig3_spectrum.png",
              img_dir / "threshold_rules.md"]:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

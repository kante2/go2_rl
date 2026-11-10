#!/usr/bin/env python3
"""Descriptive force/velocity-noise associations from precomputed contact events.

Uses train only by default. Does not fit a model, tune timing, select loud peaks,
or change the source event tables. Repeated/overlapping events are not treated
as independent observations for significance tests (no p-values are reported).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/low_noise_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "noise_prediction_model/results/analysis/contact_events"
PRIMARY = {
    "force_post_peak_sensor_units": "Contact force: post-contact maximum (sensor units)",
    "pre_downward_speed_mps": "Impact velocity: pre-contact downward speed (m/s)",
}
FEATURES = {
    **PRIMARY,
    "pre_speed_world_mps": "Pre-contact 3D foot speed (m/s)",
    "force_post_max_rise_sensor_units_per_s": "Maximum force rise rate (sensor units/s)",
}
TARGETS = {
    "noise_post_rms_dbfs": "Post-contact RMS noise (dBFS)",
    "noise_post_peak_dbfs": "Post-contact peak noise (dBFS)",
}
COLORS = {"FL": "tab:blue", "FR": "tab:orange", "RL": "tab:green", "RR": "tab:purple"}


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pair_statistics(frame, feature, target):
    pair = frame[["trial", feature, target]].copy()
    pair = pair[np.isfinite(pair[[feature, target]]).all(axis=1)]
    x, y = pair[feature], pair[target]
    # Within-trial ranks remove trial-level offsets before pooling. This is
    # event-weighted, descriptive, and not a causal adjustment for walking speed.
    ranks = pair.groupby("trial")[[feature, target]].rank(method="average")
    ranks = ranks - ranks.groupby(pair.trial).transform("mean")
    centered = pair[[feature, target]] - pair.groupby("trial")[[feature, target]].transform("mean")
    local = []
    for _, trial in pair.groupby("trial"):
        local.append(correlation(trial[feature].rank(), trial[target].rank()))
    local = np.asarray(local, float)
    local = local[np.isfinite(local)]
    return dict(
        n_events=len(pair), n_trials=pair.trial.nunique(),
        pearson_r=correlation(x, y), spearman_rho=correlation(x.rank(), y.rank()),
        within_trial_pearson_r=correlation(centered[feature], centered[target]),
        within_trial_rank_rho=correlation(ranks[feature], ranks[target]),
        n_trials_with_defined_rho=len(local),
        mean_trial_spearman_rho=float(np.mean(local)) if len(local) else np.nan,
        median_trial_spearman_rho=float(np.median(local)) if len(local) else np.nan,
    )


def subsets(frame):
    return {
        "all_valid": frame[frame.feature_valid],
        "sync_candidate": frame[frame.quality_ok],
        "isolated_valid": frame[frame.feature_valid & frame.isolated_foot_event],
        "isolated_sync_candidate": frame[frame.quality_ok & frame.isolated_foot_event],
    }


def tables(frame):
    summary, per_trial = [], []
    for subset, data in subsets(frame).items():
        for leg in ["ALL", *COLORS]:
            group = data if leg == "ALL" else data[data.leg == leg]
            for feature in FEATURES:
                for target in TARGETS:
                    summary.append(dict(subset=subset, leg=leg, feature=feature, target=target,
                                        **pair_statistics(group, feature, target)))
                    for trial, local in group.groupby("trial"):
                        per_trial.append(dict(subset=subset, leg=leg, trial=trial,
                                              feature=feature, target=target,
                                              **pair_statistics(local, feature, target)))
    return pd.DataFrame(summary), pd.DataFrame(per_trial)


def number(value):
    return f"{value:+.3f}" if np.isfinite(value) else "undefined"


def scatter(ax, frame, feature, target):
    finite = np.isfinite(frame[[feature, target]]).all(axis=1)
    data = frame[finite]
    for leg, color in COLORS.items():
        local = data[data.leg == leg]
        ax.scatter(local[feature], local[target], s=9, alpha=.22, color=color,
                   edgecolors="none", label=leg, rasterized=True)
    # Show medians across quantile bins; do not imply a fitted linear mechanism.
    if len(data) >= 30 and data[feature].nunique() >= 3:
        bins = pd.qcut(data[feature], q=10, duplicates="drop")
        medians = data.groupby(bins, observed=True)[[feature, target]].median()
        ax.plot(medians[feature], medians[target], "ko-", lw=1.5, ms=4,
                label="Bin medians")
    stats = pair_statistics(data, feature, target)
    annotation = (f"n={stats['n_events']:,}; trials={stats['n_trials']}\n"
                  f"Pearson r={number(stats['pearson_r'])}\n"
                  f"Spearman rho={number(stats['spearman_rho'])}\n"
                  f"Within-trial rank rho={number(stats['within_trial_rank_rho'])}")
    ax.text(.025, .975, annotation, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=.88, edgecolor=".8"))
    ax.set(xlabel=FEATURES[feature], ylabel=TARGETS[target])
    ax.grid(alpha=.2)
    return stats


def figures(frame, output):
    groups = subsets(frame)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, feature in enumerate(PRIMARY):
        for col, target in enumerate(TARGETS):
            scatter(axes[row, col], groups["all_valid"], feature, target)
    axes[0, 0].legend(loc="lower right", fontsize=8, ncol=3)
    fig.suptitle("Contact-event force / impact velocity vs noise | train, all valid events\n"
                 "Force and audio: 0 to +40 ms; velocity: last sample before contact", fontsize=14)
    fig.savefig(output / "contact_force_impact_velocity_correlations.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="row", sharey="row", constrained_layout=True)
    for row, feature in enumerate(PRIMARY):
        for col, subset in enumerate(("all_valid", "sync_candidate")):
            scatter(axes[row, col], groups[subset], feature, "noise_post_rms_dbfs")
            axes[row, col].set_title("All valid events" if col == 0 else "Sync-candidate events only")
    axes[0, 0].legend(loc="lower right", fontsize=8, ncol=3)
    fig.suptitle("Sensitivity to synchronization quality | train\n"
                 "Sync candidates cover fewer trials; this is also a change of sampled conditions", fontsize=13)
    fig.savefig(output / "contact_force_impact_velocity_sync_comparison.png", dpi=170)
    plt.close(fig)


def report(frame, summary, output, source):
    valid = frame[frame.feature_valid]
    quality = frame[frame.quality_ok]
    isolated = valid[valid.isolated_foot_event]
    lines = [
        "# 접촉 이벤트의 힘·착지 속도와 소음 상관관계", "",
        f"입력: `{source}`. train만 사용하며 test와 기존 예측 모델·선정 feature는 변경하지 않았다.",
        f"전체 {len(frame):,}건 중 유효 {len(valid):,}건 / {valid.trial.nunique()} trial. "
        f"동기화 후보 통과 {len(quality):,}건 / {quality.trial.nunique()} trial. "
        f"다른 발의 착지가 분석 창에 없는 유효 이벤트 {len(isolated):,}건.", "",
        "## 변수와 구간", "",
        "- Contact force: 해당 발의 접촉 시작부터 +40ms까지 최대 센서값. N으로 보정되지 않은 sensor units다.",
        "- Impact velocity: 접촉 직전 마지막 유효 샘플의 지면 좌표계 하강 속도 `max(-v_z, 0)` (m/s). "
        "단단한 평지의 법선 속도 대리지표이며, 상승 속도는 0이다. 몸체 선속도·자이로·관절 기구학으로 계산된 추정값이다.",
        "- Noise RMS / peak: 접촉 시작부터 +40ms 미만까지 원본 PCM에서 DC를 제거해 계산한 RMS / 절대 최대 진폭의 dBFS. "
        "음압 교정 dB SPL이 아니다. dBFS가 커질수록(음수의 절댓값이 작을수록) 더 큰 녹음 진폭이다.",
        "- 접촉 기준은 force on=10, off=5, 6ms 지속 확인 후 최초 교차 시점으로 소급한 이벤트다. "
        "속도는 보통 접촉 -2ms 값이다. 소음 peak를 찾아 이벤트를 다시 정렬하거나 큰 소리만 선택하지 않았다.",
        "- CSV에는 보조 지표인 접촉 전 3D 속력과 접촉 후 최대 힘 증가율도 포함한다.", "",
        "## 전체 유효 이벤트", "",
        "|변수|소음|n|Pearson r|Spearman rho|trial 내 rank rho|",
        "|---|---|---:|---:|---:|---:|",
    ]
    main = summary[(summary.subset == "all_valid") & (summary.leg == "ALL") & summary.feature.isin(PRIMARY)]
    for row in main.itertuples():
        feature = "접촉 후 최대 힘" if row.feature == "force_post_peak_sensor_units" else "접촉 전 하강 속도"
        target = "RMS" if row.target == "noise_post_rms_dbfs" else "Peak"
        lines.append(f"|{feature}|{target}|{row.n_events:,}|{number(row.pearson_r)}|{number(row.spearman_rho)}|{number(row.within_trial_rank_rho)}|")
    lines += [
        "", "## 계산과 해석", "",
        "Pearson은 선형 관계, Spearman은 순위 관계다. 부호를 유지한다. "
        "trial 내 rank rho는 각 trial 안에서 x·y를 각각 평균 순위로 바꾸고 trial 평균 순위를 뺀 뒤 합쳐 계산한다. "
        "이벤트 수로 가중되며, 전체 상관에 섞인 trial 간 차이를 구분하기 위한 기술 통계다. "
        "trial별 Spearman의 동일 가중 평균·중앙값도 요약 CSV에 저장했다. "
        "상수 또는 3건 미만이면 상관은 정의하지 않고 빈 값으로 남긴다.", "",
        "`all_valid`, `sync_candidate`, `isolated_valid`, `isolated_sync_candidate`를 각각 ALL 및 FL/FR/RL/RR로 나눴다. "
        "통과 조건은 기존 feature_valid / quality_ok를 그대로 사용한다. command_vx=0 trial도 실제 착지 이벤트가 있어 포함한다.", "",
        f"유효 이벤트 중 {len(valid)-len(isolated):,}건 ({100*(1-len(isolated)/len(valid)):.1f}%)은 "
        "[-20,+40]ms에 다른 발의 착지가 있어 마이크 소음을 해당 발 하나에 배정할 수 없다. "
        "isolated는 다른 발의 착지 시작이 없는 조건이며 다른 발의 지지력·모터 소음까지 제거하지는 않는다.", "",
        "같은 trial의 반복 착지와 겹치는 오디오 창은 독립 표본이 아니다. 독립성 가정의 p-value는 보고하지 않는다. "
        "검은 선은 x의 분위수 구간별 x·y 중앙값으로, 회귀선이 아니다. "
        "동기화 후보 역시 완벽한 센서 동기화를 보증하지 않으며 subset 간 차이에는 trial 구성 차이도 반영된다.", "",
        "이 결과는 개별 상관관계이며 다른 feature 대비 중요도·인과관계·충격음 단독 발생량을 판정하지 않는다. "
        "속도·힘·소음 모두 보행 속도, 다리, 접촉 조합 등의 영향을 받을 수 있다.", "",
        "## 파일", "",
        "- `contact_force_impact_velocity_correlations.png`: 최대 힘 / 하강 속도 × 소음 RMS / peak.",
        "- `contact_force_impact_velocity_sync_comparison.png`: 전체 유효 / 동기화 후보의 RMS 관계 비교.",
        "- `contact_event_correlation_summary.csv`: subset·발·변수·소음별 전체/내부 상관과 trial별 요약.",
        "- `contact_event_correlations_by_trial.csv`: 개별 trial의 동일 계산.",
        "- `contact_event_correlation_events.csv`: 실제 사용한 유효 이벤트의 원래 53개 컬럼.",
        "- `contact_event_correlation_coverage.csv`: trial·발별 전체/유효/동기화/고립 이벤트 수.",
        "- `contact_event_correlation_metadata.json`: 입력 해시와 정의·분석 설정.", "",
        "기존 전체 보행 기반 `contact_feature_correlations.png`와 `contact_feature_correlation_summary.csv`는 제거했다. "
        "기존 모델 분석의 report.md·selected_features 및 기타 결과는 별도 분석으로 유지한다. "
        "기존 model_evaluation.py를 실행하면 이전 파일이 다시 생성될 수 있다.", "",
        "재실행: `.venv/bin/python noise_prediction_model/analysis/contact_event_correlations.py`", "",
    ]
    (output / "contact_event_correlation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "preprocessed/1007/train/contact_event_features.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"event_id", "trial", "split", "leg", "feature_valid", "quality_ok", "isolated_foot_event",
                "window_after_ms", "window_before_ms", *FEATURES, *TARGETS}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or not frame.split.eq("train").all():
        raise ValueError("Expected a nonempty train-only event table")
    if frame.event_id.duplicated().any() or not frame.leg.isin(COLORS).all():
        raise ValueError("Invalid event IDs or leg labels")
    for column in ("feature_valid", "quality_ok", "isolated_foot_event"):
        if frame[column].isna().any() or frame[column].dtype != bool:
            raise ValueError(f"Expected boolean flag: {column}")
    if (frame.quality_ok & ~frame.feature_valid).any():
        raise ValueError("quality_ok must be a subset of feature_valid")
    if not frame.window_before_ms.eq(20).all() or not frame.window_after_ms.eq(40).all():
        raise ValueError("This report requires the documented [-20,+40]ms event windows")
    if not frame.feature_valid.any():
        raise ValueError("No valid events to analyze")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, per_trial = tables(frame)
    summary.to_csv(args.output / "contact_event_correlation_summary.csv", index=False)
    per_trial.to_csv(args.output / "contact_event_correlations_by_trial.csv", index=False)
    frame[frame.feature_valid].to_csv(args.output / "contact_event_correlation_events.csv", index=False)
    coverage = frame.assign(
        isolated_valid=frame.feature_valid & frame.isolated_foot_event,
        isolated_sync_candidate=frame.quality_ok & frame.isolated_foot_event,
    ).groupby(["trial", "leg"]).agg(
        n_events=("event_id", "size"), n_valid=("feature_valid", "sum"),
        n_sync_candidate=("quality_ok", "sum"), n_isolated_valid=("isolated_valid", "sum"),
        n_isolated_sync_candidate=("isolated_sync_candidate", "sum"),
    ).reset_index()
    coverage.to_csv(args.output / "contact_event_correlation_coverage.csv", index=False)
    figures(frame, args.output)
    report(frame, summary, args.output, args.input.resolve())
    metadata = dict(
        input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), split="train",
        primary_features=PRIMARY, supplementary_features={k: v for k, v in FEATURES.items() if k not in PRIMARY},
        targets=TARGETS, source_event_window_ms=[-20, 40], force_window_ms=[0, 40],
        pcm_noise_window_ms="[0,40)", velocity="last valid sample strictly before threshold onset",
        n_source_events=len(frame), n_valid_events=int(frame.feature_valid.sum()),
        no_p_values=True, no_time_shift_fitted=True,
        within_trial_rank_method="Per-trial average ranks, demean within trial, pool and correlate (event weighted)",
    )
    (args.output / "contact_event_correlation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    # These legacy plots use different, whole-walk derived features. The event
    # outputs use distinct names so rerunning model_evaluation cannot overwrite them.
    for name in ("contact_feature_correlations.png", "contact_feature_correlation_summary.csv"):
        (args.output / name).unlink(missing_ok=True)
    main_rows = summary[(summary.subset == "all_valid") & (summary.leg == "ALL") & summary.feature.isin(PRIMARY)]
    print(main_rows.to_string(index=False))
    print(f"Saved contact-event correlations -> {args.output}")


if __name__ == "__main__":
    main()

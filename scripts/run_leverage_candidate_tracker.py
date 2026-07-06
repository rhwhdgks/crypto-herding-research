"""Leverage-state 후보 forward tracker.

후보 규칙 (experiments/informed_trading/ step5~9, 2026-07-06):
  DOGE down micro-herding event (15m, rolling 5d 15%ile threshold — ex-ante)
  + funding_pre > 0.0001 (crowded long, 이벤트 전 공표값)
  + d_oi_event > flush_cut (이벤트 버킷 OI 미붕괴, 버킷 종료 시점 관측)
  → 알트 4종(+DOGE) 동일가중 long basket, +30min 보유

이 스크립트는 신규 OOS 구간의 이벤트를 로그에 누적하고 레짐 상태를 보고한다.
판정 기준: outputs/tracker_decision_criteria_2026-04-11.md

실행:
  .venv/bin/python scripts/run_leverage_candidate_tracker.py \
      --config configs/tick/leverage_candidate/tracker.yaml
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tick_short_horizon import build_tick_short_horizon_dataset, prepare_micro_herding_frame  # noqa: E402
from utils import load_config, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leverage-state candidate forward tracker")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


# ---------- futures state (funding / OI) ----------

def _download(url: str, dest: Path) -> str:
    if dest.exists():
        return "cache"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp, dest.open("wb") as fh:
            fh.write(resp.read())
        return "download"
    except HTTPError as exc:
        dest.unlink(missing_ok=True)
        if exc.code == 404:
            return "missing"
        raise


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        member = [n for n in zf.namelist() if n.endswith(".csv")][0]
        return pd.read_csv(io.BytesIO(zf.read(member)))


def build_futures_state(symbol: str, start: str, end: str, fs_cfg: dict) -> pd.DataFrame:
    """15분 그리드 funding_pre / d_oi_event. 워밍업으로 start 이전 1개월 funding 포함."""
    base = str(fs_cfg["base_url"])
    raw_dir = Path(fs_cfg["local_data_dir"]) / symbol

    # funding: start 직전 달부터 (직전 공표값 ffill 용)
    f_start = (pd.Timestamp(start) - pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d")
    frames = []
    for ts in pd.date_range(f_start, end, freq="MS"):
        name = f"{symbol}-fundingRate-{ts.strftime('%Y-%m')}.zip"
        dest = raw_dir / "fundingRate" / name
        if _download(f"{base}/monthly/fundingRate/{symbol}/{name}", dest) == "missing":
            continue
        frames.append(_read_zip_csv(dest))
    funding = pd.concat(frames, ignore_index=True)
    funding["calc_time"] = pd.to_datetime(pd.to_numeric(funding["calc_time"]), unit="ms", utc=True).dt.as_unit("ns")
    funding = funding[["calc_time", "last_funding_rate"]].sort_values("calc_time")

    # metrics(OI): 일 단위
    frames = []
    for ts in pd.date_range(start, end, freq="D"):
        name = f"{symbol}-metrics-{ts.strftime('%Y-%m-%d')}.zip"
        dest = raw_dir / "metrics" / name
        if _download(f"{base}/daily/metrics/{symbol}/{name}", dest) == "missing":
            continue
        frames.append(_read_zip_csv(dest))
    if not frames:
        raise SystemExit("metrics 데이터 없음 — 날짜 범위 확인 필요")
    metrics = pd.concat(frames, ignore_index=True)
    metrics["create_time"] = pd.to_datetime(metrics["create_time"], utc=True, format="mixed")
    metrics = metrics[["create_time", "sum_open_interest"]].sort_values("create_time")

    metrics["bucket_start"] = metrics["create_time"].dt.floor("15min")
    grp = metrics.groupby("bucket_start")
    state = pd.DataFrame({"oi_end": grp["sum_open_interest"].last()}).reset_index()
    full_idx = pd.date_range(state["bucket_start"].min(), state["bucket_start"].max(), freq="15min", tz="UTC")
    state = state.set_index("bucket_start").reindex(full_idx)
    state.index.name = "bucket_start"
    state["oi_start"] = state["oi_end"].shift(1)
    state["d_oi_event"] = state["oi_end"] / state["oi_start"] - 1.0
    state = state.reset_index()
    state["bucket_start"] = state["bucket_start"].dt.as_unit("ns")

    state = pd.merge_asof(
        state.sort_values("bucket_start"),
        funding.rename(columns={"calc_time": "bucket_start", "last_funding_rate": "funding_pre"}),
        on="bucket_start", direction="backward",
    )
    return state


# ---------- tracker ----------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    cand = config["candidate"]
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = int(config["analysis"]["interval_minutes"][0])
    horizon = int(config["analysis"]["forward_horizons_minutes"][0])
    leader = str(cand["leader"])
    oos_start = pd.Timestamp(cand["oos_start"], tz="UTC")

    print("1) tick bucket frames 빌드 (다운로드 포함)...")
    bucket_frames, load_summary = build_tick_short_horizon_dataset(config)
    bucket_frame = bucket_frames[interval]
    micro = prepare_micro_herding_frame(bucket_frame, config)
    print(f"   micro frame: {len(micro):,} rows, "
          f"{micro['bucket_start'].min()} ~ {micro['bucket_start'].max()}")

    print("2) futures state (funding/OI) 빌드...")
    data_cfg = config["data"]
    state = build_futures_state(leader, str(data_cfg["start"]),
                                micro["bucket_start"].max().strftime("%Y-%m-%d"),
                                config["futures_state"])
    n_state = int(state["oi_end"].notna().sum())
    print(f"   state: {n_state:,} OI buckets")

    # 3) leader 이벤트 + 진입 조건
    lead = micro.loc[
        (micro["symbol"] == leader)
        & (micro["event_label"] == f"micro_herding_{cand['direction']}")
        & (micro["bucket_start"] >= oos_start),
        ["bucket_start", "herding_score", "bucket_return"],
    ].copy()
    lead["bucket_start"] = lead["bucket_start"].dt.as_unit("ns")
    lead = lead.merge(state[["bucket_start", "funding_pre", "d_oi_event"]], on="bucket_start", how="left")

    crowded = lead["funding_pre"] > float(cand["funding_threshold"])
    no_flush = lead["d_oi_event"] > float(cand["flush_cut"])
    lead["signal"] = crowded & no_flush & lead["funding_pre"].notna() & lead["d_oi_event"].notna()

    # 4) basket forward return
    fwd_col = f"forward_return_{horizon}m"
    targets = list(cand["basket_targets"])
    basket_syms = targets + ([leader] if cand.get("include_leader_in_basket") else [])
    fwd = micro.loc[micro["symbol"].isin(basket_syms), ["symbol", "bucket_start", fwd_col]].copy()
    fwd["bucket_start"] = fwd["bucket_start"].dt.as_unit("ns")
    fwd_wide = fwd.pivot_table(index="bucket_start", columns="symbol", values=fwd_col)
    fwd_wide["basket_fwd"] = fwd_wide[basket_syms].mean(axis=1)
    lead = lead.merge(fwd_wide.reset_index()[["bucket_start", "basket_fwd"]], on="bucket_start", how="left")

    signals = lead.loc[lead["signal"]].copy()

    # 5) 레짐 상태 (state 전체 기준)
    recent = state.loc[state["bucket_start"] >= oos_start]
    crowded_share = float((recent["funding_pre"] > float(cand["funding_threshold"])).mean())
    regime = "ACTIVE" if crowded_share > 0.05 else "DORMANT"

    # 6) 로그 append (bucket_start 기준 dedupe)
    log_path = out_dir / "leverage_tracker_log.csv"
    new_rows = signals[["bucket_start", "funding_pre", "d_oi_event", "herding_score",
                        "bucket_return", "basket_fwd"]].copy()
    new_rows["logged_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    if log_path.exists():
        old = pd.read_csv(log_path)
        old["bucket_start"] = pd.to_datetime(old["bucket_start"], utc=True).dt.as_unit("ns")
        combined = pd.concat([old, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["bucket_start"], keep="first")
    else:
        combined = new_rows
    combined = combined.sort_values("bucket_start")
    combined.to_csv(log_path, index=False)

    # 7) 리포트
    n_events = len(lead)
    n_signals = len(signals)
    fees = cand.get("fee_scenarios", {})
    lines = [
        "# Leverage-state 후보 forward tracker",
        "",
        f"실행: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        f"OOS 구간: {oos_start.date()} ~ {micro['bucket_start'].max().date()}",
        "",
        "## 규칙",
        f"- {leader} down micro-herding event (15m, rolling 5d 15%ile)",
        f"- funding_pre > {cand['funding_threshold']} (crowded long)",
        f"- d_oi_event > {cand['flush_cut']} (OI no-flush, in-sample tercile 고정컷)",
        f"- basket: {', '.join(s.replace('USDT','') for s in basket_syms)} 동일가중 +{horizon}min",
        "",
        "## 레짐 상태",
        f"- OOS 구간 crowded funding 버킷 비중: **{crowded_share:.1%}**",
        f"- 판정: **{regime}** (기준: 5% 초과 시 ACTIVE)",
        "",
        "## 신규 관측",
        f"- {leader} down 이벤트: {n_events}",
        f"- 진입 신호 (전체 조건 충족): **{n_signals}**",
        "",
    ]
    total = pd.read_csv(log_path)
    done = total.loc[total["basket_fwd"].notna()]
    if len(done):
        gross = done["basket_fwd"].astype(float)
        lines += [
            "## 누적 성과 (로그 전체, 이벤트당)",
            f"- n = {len(done)}",
            f"- gross: mean {gross.mean()*100:+.4f}% / win {(gross>0).mean():.3f}",
        ]
        for fname, fee in fees.items():
            net = gross - float(fee)
            lines.append(f"- {fname} ({fee*100:.2f}%): mean {net.mean()*100:+.4f}% / 누적 {net.sum()*100:+.2f}%")
    else:
        lines += ["## 누적 성과", "- 아직 기록된 신호 없음 (레짐 휴면 지속 중이면 정상)"]
    lines += [
        "",
        "## 참고",
        "- 근거 분석: `experiments/informed_trading/informed_trading_report.md`",
        "- 판정 기준: `outputs/tracker_decision_criteria_2026-04-11.md`",
        "- 이 후보는 crowded-long 레짐 조건부 — DORMANT 기간의 신호 부재는 후보 결함이 아님",
    ]
    report_path = out_dir / "leverage_tracker_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== Tracker 결과 ===")
    print(f"레짐: {regime} (crowded 비중 {crowded_share:.1%})")
    print(f"{leader} down 이벤트 {n_events} / 진입 신호 {n_signals}")
    print(f"log   → {log_path}")
    print(f"report→ {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.framework.strategy_base import RunContext, RunResult, Strategy
from strategies.framework.performance import build_equity_from_trades


def _compute_pnl_ratio_by_asset(trades_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    盈亏比（平均盈利 / 平均亏损）按资产分组。
    若无亏损交易则返回 inf；无交易则跳过该资产。
    """
    if trades_df is None or trades_df.empty:
        return {}
    if "asset" not in trades_df.columns or "return_pct_portfolio" not in trades_df.columns:
        return {}

    ratios: Dict[str, float] = {}
    for asset, df in trades_df.groupby("asset"):
        wins = df[df["return_pct_portfolio"] > 0]["return_pct_portfolio"]
        losses = df[df["return_pct_portfolio"] < 0]["return_pct_portfolio"]
        if wins.empty and losses.empty:
            continue
        if losses.empty:
            ratio = float("inf")
        else:
            ratio = wins.mean() / abs(losses.mean()) if not wins.empty else 0.0
        safe_key = asset.replace("/", "_").replace(" ", "_")
        ratios[f"pnl_ratio_{safe_key}"] = float(ratio)
    return ratios

class MomentumRotationWeeklyStrategy(Strategy):
    """
    Simplified momentum rotation on weekly bars.
    Expects `prices` as dict[asset_id -> DataFrame] with daily OHLCV in native currency.
    Uses RunContext.fx_prices to convert为基准货币用于绩效，但交易明细保留原币种价格。
    """

    # Canonical id (legacy configs may still use "momentum_rotation_4w" as an alias)
    name = "momentum_rotation_weekly"

    def run(
        self,
        prices: Any,
        asset_pool: List[str],
        params: Dict[str, Any],
        context: RunContext,
    ) -> RunResult:
        lookback_weeks = int(params.get("lookback_weeks", 4))
        rebalance_weeks = int(params.get("rebalance_weeks", 1))
        warmup_weeks = int(params.get("warmup_weeks", lookback_weeks))

        # 控制是否丢弃“整周完全无价格”的锚点周。
        # 默认保持历史行为：丢弃这类周，从而 lookback_weeks 按“有效锚点周”计数。
        # 若 preserve_empty_weeks=True，则保留这些周，仅用于时间轴/动量窗口计数，
        # 但当周不做换仓决策（因为没有价格可以交易）。
        preserve_empty_weeks_raw = params.get("preserve_empty_weeks", False)
        if isinstance(preserve_empty_weeks_raw, str):
            preserve_empty_weeks = preserve_empty_weeks_raw.strip().lower() not in (
                "false",
                "0",
                "no",
                "off",
            )
        else:
            preserve_empty_weeks = bool(preserve_empty_weeks_raw)

        # 绝对动量阈值（可选）：若设置为数值，则当最强资产在 lookback 窗的收益
        # 低于该阈值时，本周持仓将被平掉并进入空仓（持有现金，收益记为 0）。
        abs_mom_raw = params.get("abs_mom_threshold", None)
        if abs_mom_raw in (None, "", "null"):
            abs_mom_threshold: Optional[float] = None
        else:
            try:
                abs_mom_threshold = float(abs_mom_raw)
            except (TypeError, ValueError):
                abs_mom_threshold = None

        execution_mode = params.get("execution", "close")  # "close" or "next_open"
        if execution_mode not in ("close", "next_open"):
            raise ValueError("execution must be 'close' or 'next_open'")
        price_precision = getattr(context, "price_precision", None)
        def _round_px(x):
            try:
                return round(x, price_precision) if price_precision is not None and not pd.isna(x) else x
            except Exception:
                return x

        guard_filters = params.get("_ma_guard_filters") or []
        guard_event_scope = params.get("_ma_guard_event_scope")
        guard_events: List[Dict[str, Any]] = []
        asset_price_meta: Dict[str, Dict[str, Any]] = getattr(context, "asset_price_meta", {}) or {}

        holiday_catchup_raw = params.get("holiday_catchup_days", 3)
        try:
            holiday_catchup_days = int(holiday_catchup_raw)
        except (TypeError, ValueError):
            holiday_catchup_days = None
        if holiday_catchup_days is not None and holiday_catchup_days < 0:
            holiday_catchup_days = None

        def _to_naive_timestamp(val: Any) -> Optional[pd.Timestamp]:
            if val is None:
                return None
            try:
                ts = pd.to_datetime(val)
            except Exception:
                return None
            tzinfo = getattr(ts, "tzinfo", None)
            if tzinfo is not None:
                ts = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
            return ts

        def _covered_by_known_gap(asset_id: str, target_ts: Optional[pd.Timestamp]) -> bool:
            if target_ts is None or not asset_price_meta:
                return False
            meta = asset_price_meta.get(asset_id)
            if not isinstance(meta, dict):
                return False
            ranges = meta.get("known_missing_ranges")
            if not isinstance(ranges, list):
                return False
            for rng in ranges:
                if not isinstance(rng, dict):
                    continue
                start_val = rng.get("start")
                end_val = rng.get("end")
                try:
                    start_ts = _to_naive_timestamp(start_val) if start_val else None
                    end_ts = _to_naive_timestamp(end_val) if end_val else None
                except Exception:
                    continue
                start_ok = True if start_ts is None else target_ts >= start_ts
                end_ok = True if end_ts is None else target_ts <= end_ts
                if start_ok and end_ok:
                    return True
            return False

        def _emit_guard_event(reason: str, payload: Dict[str, Any], scope: Optional[Any] = None) -> None:
            record = dict(payload)
            record["reason"] = reason
            guard_events.append(record)
            if scope is not None and hasattr(scope, "emit"):
                try:
                    scope.emit(event=reason, payload=record)
                except Exception:
                    pass

        def _count_workdays(start_dt: Optional[pd.Timestamp], end_dt: Optional[pd.Timestamp]) -> int:
            if start_dt is None or end_dt is None or pd.isna(start_dt) or pd.isna(end_dt):
                return 0
            start_day = pd.Timestamp(start_dt).normalize()
            end_day = pd.Timestamp(end_dt).normalize()
            if end_day <= start_day:
                return 0
            day_range = pd.date_range(start=start_day + pd.Timedelta(days=1), end=end_day, freq="D")
            return int(sum(ts.weekday() < 5 for ts in day_range))

        weekly_close = {}
        weekly_close_date = {}
        weekly_fx_close = {}
        weekly_close_ccy = {}
        weekly_next_open = {}
        weekly_next_open_ccy = {}
        weekly_next_open_date = {}
        weekly_last_ts = {}
        weekly_first_open = {}
        weekly_first_open_ccy = {}
        weekly_first_open_date = {}
        daily_price_history: Dict[str, pd.DataFrame] = {}
        # allow warmup data earlier than start_date to build indicators
        start_for_data = context.start_date
        if context.start_date is not None and warmup_weeks > 0:
            start_for_data = context.start_date - pd.to_timedelta(7 * warmup_weeks, unit="D")

        for aid in asset_pool:
            df = prices[aid].copy()
            if start_for_data is not None:
                df = df[df.index >= start_for_data]
            if context.end_date is not None:
                df = df[df.index <= context.end_date]

            ccy = context.asset_currencies.get(aid, context.base_currency)
            if ccy == context.base_currency:
                df["fx_close"] = 1.0
            else:
                pair = f"{ccy}/{context.base_currency}"
                fx_df = context.fx_prices.get(pair)
                if fx_df is None:
                    raise KeyError(f"FX prices for pair {pair} not provided.")
                fx_df = fx_df.rename(columns={"close": "fx_close"})
                df = df.join(fx_df[["fx_close"]], how="inner")

            # base-currency prices for performance/selection
            df["open_base"] = df["open"] * df["fx_close"]
            df["high_base"] = df["high"] * df["fx_close"]
            df["low_base"] = df["low"] * df["fx_close"]
            df["close_base"] = df["close"] * df["fx_close"]

            # group by calendar week (anchor configurable per run or param override)
            week_anchor = params.get("week_anchor") or getattr(context, "week_anchor", "SUN")
            anchor_key = str(week_anchor).upper()
            # Allow flexible weekly anchors: SUN/MON/TUE/WED/THU/FRI/SAT
            anchor_map = {
                "SUN": "W-SUN",
                "MON": "W-MON",
                "TUE": "W-TUE",
                "WED": "W-WED",
                "THU": "W-THU",
                "FRI": "W-FRI",
                "SAT": "W-SAT",
            }
            freq = anchor_map.get(anchor_key)
            if freq is None:
                raise ValueError(
                    f"Unsupported week_anchor '{week_anchor}'. "
                    "Use one of: SUN, MON, TUE, WED, THU, FRI, SAT."
                )
            week_period = df.index.to_period(freq)
            daily_price_history[aid] = df
            weekly_close_ccy[aid] = df["close"].groupby(week_period).last()
            weekly_close[aid] = df["close_base"].groupby(week_period).last()
            weekly_close_date[aid] = df.groupby(week_period).apply(lambda x: x.index.max())
            weekly_fx_close[aid] = df["fx_close"].groupby(week_period).last()
            weekly_last_ts[aid] = df.groupby(week_period).apply(lambda x: x.index.max().to_period("h"))
            weekly_first_open_ccy[aid] = df["open"].groupby(week_period).first()
            weekly_first_open[aid] = df["open_base"].groupby(week_period).first()
            weekly_first_open_date[aid] = df.groupby(week_period).apply(lambda x: x.index.min())

        all_weeks = None
        for series in weekly_close.values():
            all_weeks = series.index if all_weeks is None else all_weeks.union(series.index)
        if all_weeks is None:
            raise ValueError("No weekly data available.")
        weeks = all_weeks.sort_values()
        if preserve_empty_weeks:
            # 构造连续的自然周序列：即便整周无任何成交价也占据一个锚点周，
            # 这样在计算动量时 lookback_weeks 按自然周计数，而不是按“有数据的周”计数。
            weeks = pd.period_range(start=weeks.min(), end=weeks.max(), freq=freq)
        close_df = pd.concat(weekly_close, axis=1).reindex(weeks)
        close_date_df = pd.concat(weekly_close_date, axis=1).reindex(weeks)
        close_ccy_df = pd.concat(weekly_close_ccy, axis=1).reindex(weeks)
        # 标记整周完全无价格的锚点周（所有资产均为 NaN）
        empty_week_mask = close_df.isna().all(axis=1)
        holiday_week_mask = empty_week_mask.copy()
        if preserve_empty_weeks and empty_week_mask.any():
            # 对整周无价格的 period，用“向前最近一周的价格/日期”作为占位，从而保持动量窗口连续。
            ffilled_close = close_df.ffill()
            close_df = close_df.where(~empty_week_mask, ffilled_close)
            ffilled_close_ccy = close_ccy_df.ffill()
            close_ccy_df = close_ccy_df.where(~empty_week_mask, ffilled_close_ccy)
            ffilled_close_date = close_date_df.ffill()
            close_date_df = close_date_df.where(~empty_week_mask, ffilled_close_date)
        if not preserve_empty_weeks:
            # 历史行为：丢弃这些周，lookback_weeks 按“有效周”计数
            close_df = close_df.loc[~empty_week_mask]
            close_date_df = close_date_df.loc[close_df.index]
            close_ccy_df = close_ccy_df.loc[close_df.index]
            # 重新计算 mask，保证与后续 index 对齐（这里应全为 False，仅占位）
            empty_week_mask = close_df.isna().all(axis=1)
        last_ts_df = pd.concat(weekly_last_ts, axis=1).reindex(weeks)
        open_df = pd.concat(weekly_first_open, axis=1).reindex(weeks)
        open_ccy_df = pd.concat(weekly_first_open_ccy, axis=1).reindex(weeks)
        open_date_df = pd.concat(weekly_first_open_date, axis=1).reindex(weeks)
        if execution_mode == "next_open":
            # next available week's first open; if下一周停盘，则取再下一周的开盘
            next_open_df = open_df.shift(-1).bfill()
            next_open_ccy_df = open_ccy_df.shift(-1).bfill()
            next_open_date_df = open_date_df.shift(-1).bfill()
        # use week period end (Sunday) as timestamp; trading dates inside week captured in values
        close_df.index = close_df.index.to_timestamp(how="end")
        if preserve_empty_weeks:
            # 与 close_df 对齐：在时间轴上标记哪些锚点周“无价格”，用于后续跳过交易决策
            empty_week_mask.index = close_df.index
            holiday_week_mask.index = close_df.index
        close_date_df.index = close_date_df.index.to_timestamp(how="end")
        close_ccy_df.index = close_ccy_df.index.to_timestamp(how="end")
        last_ts_df.index = last_ts_df.index.to_timestamp(how="end")
        open_df.index = open_df.index.to_timestamp(how="end")
        open_ccy_df.index = open_ccy_df.index.to_timestamp(how="end")
        open_date_df.index = open_date_df.index.to_timestamp(how="end")
        if execution_mode == "next_open":
            next_open_df.index = next_open_df.index.to_timestamp(how="end")
            next_open_ccy_df.index = next_open_ccy_df.index.to_timestamp(how="end")
            next_open_date_df.index = next_open_date_df.index.to_timestamp(how="end")

        # Momentum on weekly close
        momentum = close_df / close_df.shift(lookback_weeks) - 1

        current = None
        entry_price = None  # trade entry price (base)
        entry_price_ccy = None  # trade entry price (original ccy)
        prev_entry_date: Optional[pd.Timestamp] = None
        trade_records: List[Dict[str, Any]] = []
        hold_weeks = 0
        rebalance_event_dates: List[pd.Timestamp] = []
        consecutive_holiday_weeks = 0
        prev_anchor_date: Optional[pd.Timestamp] = None
        def _lookup_force_exit_price(asset: str, cross_dt: pd.Timestamp, timing: str):
            price_df = daily_price_history.get(asset)
            if price_df is None or price_df.empty:
                return None
            timing = (timing or execution_mode).lower()
            cross_dt = pd.to_datetime(cross_dt)
            before_or_on = price_df.loc[:cross_dt]
            if before_or_on.empty:
                return None
            if timing == "close":
                row = before_or_on.iloc[-1]
                exit_date = row.name
                exit_price_portfolio = row["close_base"]
                exit_price_ccy = row["close"]
                return exit_price_portfolio, exit_price_ccy, exit_date
            future_rows = price_df.loc[price_df.index > cross_dt]
            if future_rows.empty:
                row = before_or_on.iloc[-1]
                exit_date = row.name
                exit_price_portfolio = row["close_base"]
                exit_price_ccy = row["close"]
                return exit_price_portfolio, exit_price_ccy, exit_date
            row = future_rows.iloc[0]
            exit_date = row.name
            exit_price_portfolio = row.get("open_base", row.get("close_base"))
            exit_price_ccy = row.get("open", row.get("close"))
            return exit_price_portfolio, exit_price_ccy, exit_date

        def _detect_force_exit(asset: Optional[str], entry_dt: Optional[pd.Timestamp], week_end_dt: pd.Timestamp):
            if not guard_filters or asset is None or entry_dt is None:
                return None
            entry_dt = pd.to_datetime(entry_dt)
            week_end_dt = pd.to_datetime(week_end_dt)
            for gf in guard_filters:
                if gf.get("action") != "force_exit":
                    continue
                applies_to = gf.get("applies_to")
                if applies_to and asset not in applies_to:
                    continue
                state = gf.get("state") or {}
                cond = state.get("cond")
                if cond is None or cond.empty:
                    continue
                window = cond.loc[(cond.index >= entry_dt) & (cond.index <= week_end_dt)]
                crossed = window[window == False]  # noqa: E712
                if crossed.empty:
                    continue
                cross_dt = pd.to_datetime(crossed.index[0])
                timing = gf.get("force_exit_timing") or execution_mode
                price_info = _lookup_force_exit_price(asset, cross_dt, timing)
                if price_info is None:
                    continue
                exit_price_portfolio, exit_price_ccy, exit_date = price_info
                return {
                    "cross_date": cross_dt,
                    "exit_date": exit_date,
                    "exit_price_portfolio": exit_price_portfolio,
                    "exit_price_ccy": exit_price_ccy,
                    "guard": gf,
                    "timing": timing,
                }
            return None

        for idx, week in enumerate(close_df.index):
            target_anchor_date = pd.Timestamp(week).normalize()

            if context.start_date is not None and week < context.start_date:
                # warmup period: skip trading but keep momentum history；仍然更新上一锚点，
                # 以便后续节假日补价逻辑可以感知“前一周”的日历位置。
                prev_anchor_date = target_anchor_date
                continue

            # 当前周的实际最后交易日（全资产取最大日期），用于时间轴/输出
            week_dates_row = close_date_df.loc[week]
            week_event_date = pd.to_datetime(week_dates_row.max()) if isinstance(
                week_dates_row, (pd.Series, pd.Index)
            ) else pd.to_datetime(week_dates_row)
            week_event_date = None if (week_event_date is None or pd.isna(week_event_date)) else week_event_date

            # 若开启 preserve_empty_weeks，整周无价格的锚点周默认只用于“周计数”。
            should_skip_for_holiday = False
            is_holiday_candidate = False
            if preserve_empty_weeks:
                if holiday_week_mask.get(week, False):
                    is_holiday_candidate = True
                elif (
                    week_event_date is not None
                    and not pd.isna(week_event_date)
                    and week_event_date.normalize() < target_anchor_date
                ):
                    is_holiday_candidate = True

            if is_holiday_candidate:
                if prev_anchor_date is None:
                    start_date_str = (
                        context.start_date.date().isoformat() if context.start_date is not None else "NA"
                    )
                    week_anchor_str = target_anchor_date.date().isoformat()
                    week_event_str = (
                        week_event_date.date().isoformat() if week_event_date is not None else "NA"
                    )
                    # 如果第一周锚点就落在节假日周，历史数据无法提供“上一锚点”，
                    # 则在日历上向前推一周，合成一个前一锚点，仅用于工作日间隔计算。
                    import warnings

                    synthetic_prev_anchor = target_anchor_date - pd.Timedelta(days=7)
                    warnings.warn(
                        "Holiday catch-up: no prior anchor week in loaded history; "
                        "using synthetic prev_anchor_date one week earlier. "
                        f"Offending week_anchor={week_anchor_str}, week_event_date={week_event_str}, "
                        f"warmup_start={start_date_str}, synthetic_prev_anchor={synthetic_prev_anchor.date().isoformat()}."
                    )
                    prev_anchor_date = synthetic_prev_anchor
                consecutive_holiday_weeks += 1
                allow_holiday_rebalance = False
                if (
                    holiday_catchup_days is not None
                    and consecutive_holiday_weeks == 1
                    and week_event_date is not None
                    and prev_anchor_date is not None
                ):
                    workday_gap = _count_workdays(prev_anchor_date, week_event_date)
                    if workday_gap > holiday_catchup_days:
                        allow_holiday_rebalance = True
                if not allow_holiday_rebalance:
                    should_skip_for_holiday = True
            else:
                consecutive_holiday_weeks = 0

            if should_skip_for_holiday:
                if current is not None:
                    # 视作“又过了一周”，但因无价格无法实际交易，仅增加持有周数
                    hold_weeks += 1
                prev_anchor_date = target_anchor_date
                continue

            force_exit_ctx: Optional[Dict[str, Any]] = None
            if current is not None and prev_entry_date is not None:
                force_exit_ctx = _detect_force_exit(current, prev_entry_date, week_event_date)

            week_close = close_df.loc[week]
            week_close_ccy = close_ccy_df.loc[week]
            if execution_mode == "next_open":
                week_next_open = next_open_df.loc[week] if week in next_open_df.index else pd.Series(dtype=float)
                week_next_open_ccy = (
                    next_open_ccy_df.loc[week] if week in next_open_ccy_df.index else pd.Series(dtype=float)
                )
                week_next_open_date = (
                    next_open_date_df.loc[week]
                    if week in next_open_date_df.index
                    else pd.Series(dtype="datetime64[ns]")
                )
            else:
                week_next_open = week_close
                week_next_open_ccy = week_close_ccy
                week_next_open_date = close_date_df.loc[week]
                # In close mode we prefer aligned weekly cutoffs; allow small gaps (e.g., suspended asset) within grace days.
                ts_row = last_ts_df.loc[week]
                if ts_row.nunique(dropna=True) > 1:
                    grace_days = int(params.get("cutoff_grace_days", 1))
                    max_ts = ts_row.max()
                    max_ts_dt = _to_naive_timestamp(max_ts.to_timestamp())
                    diffs = max_ts.to_timestamp() - ts_row.astype("datetime64[ns]")

                    def _fmt_cutoff_val(val: Any) -> str:
                        if pd.isna(val):
                            return "NA"
                        if hasattr(val, "to_timestamp"):
                            try:
                                val = val.to_timestamp()
                            except Exception:
                                pass
                        ts = _to_naive_timestamp(val)
                        if ts is None:
                            return str(val)
                        return ts.date().isoformat()

                    def _fmt_delta(delta: Any) -> str:
                        try:
                            td = pd.to_timedelta(delta)
                        except Exception:
                            return str(delta)
                        if pd.isna(td):
                            return "NA"
                        hours = int(td / pd.Timedelta(hours=1))
                        if hours % 24 == 0:
                            return f"{hours // 24}d"
                        return f"{hours}h"

                    too_far = diffs > pd.Timedelta(days=grace_days)
                    if too_far.any():
                        offenders = []
                        tolerated = []
                        for aid, is_far in too_far.items():
                            if not is_far:
                                continue
                            if _covered_by_known_gap(aid, max_ts_dt):
                                tolerated.append(aid)
                            else:
                                offenders.append(aid)
                        if offenders:
                            raise ValueError(
                                "execution='close' requires weekly cutoffs within grace; "
                                f"found mismatch on week {week.date()}: {ts_row.dropna().to_dict()}"
                            )
                        if idx == 0 or ("_cutoff_known_warned" not in locals()):
                            import warnings

                            asset_label = ", ".join(sorted(tolerated)) or "selected assets"
                            warnings.warn(
                                "execution='close': cutoff mismatch beyond %sd on week %s covered by known_missing_ranges "
                                "for %s; auto-accept." % (grace_days, week.date(), asset_label)
                            )
                            _cutoff_known_warned = True
                    else:
                        # within grace: accept and proceed (log once)
                        if idx == 0 or ("_cutoff_warned" not in locals()):
                            import warnings

                            laggers = diffs[diffs > pd.Timedelta(0)].sort_values(ascending=False)
                            lagger_details = []
                            for aid, delta in laggers.items():
                                if pd.isna(delta):
                                    continue
                                lagger_details.append(
                                    f"{aid}(-{_fmt_delta(delta)} @ {_fmt_cutoff_val(ts_row.get(aid))})"
                                )
                            lagger_msg = "; lagging: " + ", ".join(lagger_details) if lagger_details else ""
                            warnings.warn(
                                f"execution='close': cutoff mismatch within {grace_days}d on week {week.date()}, "
                                f"auto-accept (likely single-asset suspension). max_cutoff={_fmt_cutoff_val(max_ts_dt)}{lagger_msg}"
                            )
                            _cutoff_warned = True

            # decide rebalance at period end
            forced_exit_triggered = force_exit_ctx is not None
            need_rebalance = forced_exit_triggered or current is None or hold_weeks >= rebalance_weeks - 1
            if need_rebalance:
                mom_row = momentum.loc[week]
                best_symbol = None
                best_mom = -np.inf
                blocked_best: Optional[str] = None

                week_day = pd.to_datetime(week).normalize()

                def _passes_guard(sym: str, entry_ts: pd.Timestamp) -> bool:
                    if not guard_filters:
                        return True
                    for gf in guard_filters:
                        applies_to = gf.get("applies_to")
                        if applies_to and sym not in applies_to:
                            continue
                        res = gf["checker"](entry_ts)
                        if not res.get("ok", False):
                            extra = res.get("extra") or {}
                            event_scope = gf.get("event_scope")
                            _emit_guard_event(
                                "ma_guard_blocked",
                                {
                                    "week": week_day,
                                    "asset": sym,
                                    "entry_date": pd.to_datetime(entry_ts) if entry_ts is not None else None,
                                    "ref_asset": gf.get("ref_asset"),
                                    "ref_close": extra.get("ref_close"),
                                    "ma": extra.get("ma"),
                                    "upper": extra.get("upper"),
                                    "lower": extra.get("lower"),
                                    "window": gf.get("config", {}).get("window"),
                                    "ma_type": gf.get("config", {}).get("ma_type"),
                                    "op": gf.get("config", {}).get("op"),
                                    "buffer_pct": gf.get("config", {}).get("buffer_pct"),
                                    "recovery_date": res.get("recovery_date"),
                                },
                                scope=event_scope,
                            )
                            return False
                    return True

                exit_block_ts = None
                if force_exit_ctx is not None:
                    exit_block_ts = force_exit_ctx.get("exit_date")
                    if exit_block_ts is not None and not pd.isna(exit_block_ts):
                        exit_block_ts = pd.to_datetime(exit_block_ts)
                for sym in asset_pool:
                    mom_val = mom_row.get(sym)
                    if pd.isna(mom_val) or pd.isna(week_close.get(sym)):
                        continue

                    entry_ts = week_next_open_date.get(sym) if isinstance(week_next_open_date, pd.Series) else None
                    if entry_ts is None or pd.isna(entry_ts):
                        _emit_guard_event(
                            "missing_entry_timestamp",
                            {
                                "week": week_day,
                                "asset": sym,
                                "entry_date": None,
                            },
                            scope=guard_event_scope,
                        )
                        continue
                    entry_ts = pd.to_datetime(entry_ts)
                    if exit_block_ts is not None and entry_ts < exit_block_ts:
                        # 强平尚未完成（entry_ts 早于 exit_date），本周不允许开仓
                        continue

                    if not _passes_guard(sym, entry_ts):
                        if blocked_best is None and (best_symbol is None or mom_val > best_mom):
                            blocked_best = sym
                        continue

                    if mom_val > best_mom:
                        best_mom = mom_val
                        best_symbol = sym

                # 绝对动量 + 允许空仓：
                # 若配置了 abs_mom_threshold 且最强资产动量低于该阈值，则本周目标仓位为空仓（current=None）。
                target_symbol = best_symbol
                if abs_mom_threshold is not None:
                    if target_symbol is None or pd.isna(best_mom) or best_mom < abs_mom_threshold:
                        target_symbol = None

                # 仅在目标仓位发生变化时才平仓/换仓；若 target_symbol 为 None 则平仓后保持空仓。
                if current != target_symbol or force_exit_ctx is not None:
                    # 1) 先平掉原持仓
                    if current is not None and entry_price is not None:
                        if force_exit_ctx is not None:
                            exit_price = _round_px(force_exit_ctx.get("exit_price_portfolio"))
                            exit_price_ccy = _round_px(force_exit_ctx.get("exit_price_ccy"))
                            exit_date = force_exit_ctx.get("exit_date")
                        else:
                            exit_price = _round_px(week_next_open.get(current, np.nan))
                            exit_price_ccy = _round_px(week_next_open_ccy.get(current, np.nan))
                            exit_date = week_next_open_date.get(current, close_date_df.loc[week, current])

                        trade_record = {
                            "asset": current,
                            "currency": context.asset_currencies.get(current, context.base_currency),
                            "entry_date": prev_entry_date,
                            "exit_date": exit_date,
                            "entry_price_portfolio": _round_px(entry_price),
                            "exit_price_portfolio": exit_price,
                            "entry_price_ccy": _round_px(entry_price_ccy),
                            "exit_price_ccy": exit_price_ccy,
                            "return_pct_portfolio": exit_price / entry_price - 1 if entry_price and not pd.isna(exit_price) else np.nan,
                            "return_pct_ccy": exit_price_ccy / entry_price_ccy - 1 if entry_price_ccy and not pd.isna(exit_price_ccy) else np.nan,
                            "ma_guard_force_exit": False,
                            "ma_guard_force_exit_cross_date": pd.NaT,
                            "ma_guard_force_exit_timing": None,
                        }
                        if force_exit_ctx is not None:
                            trade_record["ma_guard_force_exit"] = True
                            trade_record["ma_guard_force_exit_cross_date"] = force_exit_ctx.get("cross_date")
                            trade_record["ma_guard_force_exit_timing"] = force_exit_ctx.get("timing")
                            guard_scope = force_exit_ctx.get("guard", {}).get("event_scope")
                            _emit_guard_event(
                                "ma_guard_force_exit",
                                {
                                    "week": week_day,
                                    "asset": current,
                                    "entry_date": prev_entry_date,
                                    "ref_asset": force_exit_ctx.get("guard", {}).get("ref_asset"),
                                    "ref_close": None,
                                    "ma": None,
                                    "upper": None,
                                    "lower": None,
                                    "window": force_exit_ctx.get("guard", {}).get("config", {}).get("window"),
                                    "ma_type": force_exit_ctx.get("guard", {}).get("config", {}).get("ma_type"),
                                    "op": force_exit_ctx.get("guard", {}).get("config", {}).get("op"),
                                    "buffer_pct": force_exit_ctx.get("guard", {}).get("config", {}).get("buffer_pct"),
                                    "recovery_date": None,
                                    "cross_date": force_exit_ctx.get("cross_date"),
                                    "exit_date": exit_date,
                                },
                                scope=guard_scope,
                            )
                        if exit_date is not None and not pd.isna(exit_date):
                            rebalance_event_dates.append(pd.to_datetime(exit_date).normalize())
                        trade_records.append(trade_record)

                    # 2) 再根据目标建立新仓，若 target_symbol 为 None 则进入空仓
                    if target_symbol is not None and not pd.isna(week_next_open.get(target_symbol)):
                        current = target_symbol
                        entry_price = _round_px(week_next_open.get(target_symbol, week_close[target_symbol]))
                        entry_price_ccy = _round_px(week_next_open_ccy.get(target_symbol, week_close_ccy[target_symbol]))
                        prev_entry_date = week_next_open_date.get(target_symbol, close_date_df.loc[week, target_symbol])
                        hold_weeks = 0
                    else:
                        # 进入空仓：清空持仓相关状态；若 target_symbol 为空或不可交易则转为空仓
                        current = None
                        entry_price = None
                        entry_price_ccy = None
                        prev_entry_date = None
                        hold_weeks = 0
                else:
                    # 目标资产不变：保持持仓但确保下周继续评估
                    if current is not None:
                        hold_weeks = rebalance_weeks - 1
            else:
                hold_weeks += 1

            prev_anchor_date = target_anchor_date

        trades_df: Optional[pd.DataFrame] = pd.DataFrame(trade_records) if trade_records else None

        metrics, equity_series, curve_extras = build_equity_from_trades(
            trades_df,
            daily_price_history,
            execution_mode,
            context.start_date,
            context.end_date,
            price_precision,
        )
        if trades_df is not None:
            metrics.update(_compute_pnl_ratio_by_asset(trades_df))

        extra: Dict[str, Any] = {
            "rebalance_dates": [pd.to_datetime(ts).normalize() for ts in sorted(set(rebalance_event_dates))],
        }
        extra.update(curve_extras)
        if guard_events:
            # keep events for downstream CSV writing and overlay summaries; caller may drop before persisting JSON
            extra["ma_guard_events"] = guard_events

        return RunResult(
            metrics=metrics,
            equity_curve=equity_series,
            trades=trades_df,
            extra=extra,
            capabilities={"per_asset_trade_stats": True},
            metric_hooks=["per_asset_trade_stats"],
        )


class MomentumRotationDaylyStrategy(Strategy):
    """
    Daily momentum rotation driven by signal strength (no fixed rebalance cadence).

    Rules (per user spec):
    - Signal = momentum over lookback_days, optionally smoothed by MA(signal_ma_window).
    - If in cash: enter strongest asset with signal >= entry_threshold (absolute threshold).
    - If holding: switch only if best_signal > current_signal and (best_signal - current_signal) >= switch_delta.
    - No explicit cash-out logic (except ma_guard force_exit may flatten).
    - Supports execution='close' or 'next_open'.
    - Compatible with ma_guard hook params: _ma_guard_filters / _ma_guard_event_scope.
    """

    name = "momentum_rotation_dayly"

    def run(
        self,
        prices: Any,
        asset_pool: List[str],
        params: Dict[str, Any],
        context: RunContext,
    ) -> RunResult:
        lookback_days = int(params.get("lookback_days", 20))

        ma_window_raw = (
            params.get("signal_ma_window")
            or params.get("momentum_ma_window")
            or params.get("mom_ma_window")
            or params.get("ma_window_signal")
            or params.get("ma_window_momentum")
        )
        if ma_window_raw in (None, "", "null", 0, "0", False):
            ma_window: Optional[int] = None
        else:
            try:
                ma_window = int(ma_window_raw)
            except (TypeError, ValueError):
                ma_window = None
            if ma_window is not None and ma_window <= 1:
                ma_window = None

        signal_ma_type = str(params.get("signal_ma_type", "sma")).lower()

        entry_threshold_raw = params.get("entry_threshold", params.get("abs_threshold", 0.03))
        try:
            entry_threshold = float(entry_threshold_raw)
        except (TypeError, ValueError):
            entry_threshold = 0.03

        switch_delta_raw = params.get("switch_delta", params.get("filter_threshold", 0.03))
        try:
            switch_delta = float(switch_delta_raw)
        except (TypeError, ValueError):
            switch_delta = 0.03

        execution_mode = params.get("execution", "close")
        if execution_mode not in ("close", "next_open"):
            raise ValueError("execution must be 'close' or 'next_open'")

        price_precision = getattr(context, "price_precision", None)

        def _round_px(x):
            try:
                return round(x, price_precision) if price_precision is not None and not pd.isna(x) else x
            except Exception:
                return x

        guard_filters = params.get("_ma_guard_filters") or []
        guard_event_scope = params.get("_ma_guard_event_scope")
        guard_events: List[Dict[str, Any]] = []

        def _emit_guard_event(reason: str, payload: Dict[str, Any], scope: Optional[Any] = None) -> None:
            record = dict(payload)
            record["reason"] = reason
            guard_events.append(record)
            if scope is not None and hasattr(scope, "emit"):
                try:
                    scope.emit(event=reason, payload=record)
                except Exception:
                    pass

        visible_start = context.visible_start_date or context.start_date
        start_for_data = context.warmup_start_date or context.start_date

        daily_price_history: Dict[str, pd.DataFrame] = {}
        close_base_series: Dict[str, pd.Series] = {}
        close_ccy_series: Dict[str, pd.Series] = {}
        open_base_series: Dict[str, pd.Series] = {}
        open_ccy_series: Dict[str, pd.Series] = {}
        close_date_series: Dict[str, pd.Series] = {}
        open_date_series: Dict[str, pd.Series] = {}

        for aid in asset_pool:
            if aid not in prices:
                raise KeyError(f"Price data for asset '{aid}' not provided (ensure it is in asset_pool).")
            df = prices[aid].copy().sort_index()
            if start_for_data is not None:
                df = df[df.index >= start_for_data]
            if context.end_date is not None:
                df = df[df.index <= context.end_date]

            ccy = context.asset_currencies.get(aid, context.base_currency)
            if ccy == context.base_currency:
                df["fx_close"] = 1.0
            else:
                pair = f"{ccy}/{context.base_currency}"
                fx_df = context.fx_prices.get(pair)
                if fx_df is None:
                    raise KeyError(f"FX prices for pair {pair} not provided.")
                fx_df = fx_df.rename(columns={"close": "fx_close"})
                df = df.join(fx_df[["fx_close"]], how="inner")

            df["open_base"] = df["open"] * df["fx_close"]
            df["high_base"] = df["high"] * df["fx_close"]
            df["low_base"] = df["low"] * df["fx_close"]
            df["close_base"] = df["close"] * df["fx_close"]

            daily_price_history[aid] = df
            close_base_series[aid] = df["close_base"]
            close_ccy_series[aid] = df["close"]
            open_base_series[aid] = df["open_base"]
            open_ccy_series[aid] = df["open"]
            close_date_series[aid] = pd.Series(df.index, index=df.index)
            open_date_series[aid] = pd.Series(df.index, index=df.index)

        all_days = None
        for series in close_base_series.values():
            all_days = series.index if all_days is None else all_days.union(series.index)
        if all_days is None:
            raise ValueError("No daily data available.")
        idx = all_days.sort_values()

        close_df = pd.concat(close_base_series, axis=1).reindex(idx)
        close_ccy_df = pd.concat(close_ccy_series, axis=1).reindex(idx)
        open_df = pd.concat(open_base_series, axis=1).reindex(idx)
        open_ccy_df = pd.concat(open_ccy_series, axis=1).reindex(idx)
        close_date_df = pd.concat(close_date_series, axis=1).reindex(idx)
        open_date_df = pd.concat(open_date_series, axis=1).reindex(idx)

        if execution_mode == "next_open":
            next_open_df = open_df.shift(-1).bfill()
            next_open_ccy_df = open_ccy_df.shift(-1).bfill()
            next_open_date_df = open_date_df.shift(-1).bfill()

        momentum = close_df / close_df.shift(lookback_days) - 1
        if ma_window is not None:
            if signal_ma_type == "sma":
                signal = momentum.rolling(window=ma_window, min_periods=ma_window).mean()
            elif signal_ma_type == "ema":
                signal = momentum.ewm(span=ma_window, adjust=False, min_periods=ma_window).mean()
            elif signal_ma_type == "rma":
                signal = momentum.ewm(alpha=1 / ma_window, adjust=False, min_periods=ma_window).mean()
            else:
                raise ValueError("signal_ma_type must be one of: sma, ema, rma.")
        else:
            signal = momentum

        def _exec_price_for(asset: str, now_dt: pd.Timestamp) -> Optional[tuple[float, float, pd.Timestamp]]:
            now_dt = pd.to_datetime(now_dt)
            if execution_mode == "next_open":
                px = next_open_df.at[now_dt, asset]
                px_ccy = next_open_ccy_df.at[now_dt, asset]
                ts = next_open_date_df.at[now_dt, asset]
            else:
                px = close_df.at[now_dt, asset]
                px_ccy = close_ccy_df.at[now_dt, asset]
                ts = close_date_df.at[now_dt, asset]
            if ts is None or pd.isna(ts) or pd.isna(px):
                return None
            return float(px), float(px_ccy), pd.to_datetime(ts)

        def _lookup_first_open_on_or_after(asset: str, when: pd.Timestamp) -> Optional[tuple[float, float, pd.Timestamp]]:
            price_df = daily_price_history.get(asset)
            if price_df is None or price_df.empty:
                return None
            when = pd.to_datetime(when)
            future = price_df.loc[price_df.index >= when]
            if future.empty:
                return None
            row = future.iloc[0]
            return float(row["open_base"]), float(row["open"]), pd.to_datetime(row.name)

        def _passes_guard(sym: str, entry_ts: pd.Timestamp, event_day: pd.Timestamp) -> bool:
            if not guard_filters:
                return True
            for gf in guard_filters:
                applies_to = gf.get("applies_to")
                if applies_to and sym not in applies_to:
                    continue
                res = gf["checker"](entry_ts)
                if not res.get("ok", False):
                    extra = res.get("extra") or {}
                    event_scope = gf.get("event_scope")
                    _emit_guard_event(
                        "ma_guard_blocked",
                        {
                            "day": event_day,
                            "asset": sym,
                            "entry_date": pd.to_datetime(entry_ts) if entry_ts is not None else None,
                            "ref_asset": gf.get("ref_asset"),
                            "ref_close": extra.get("ref_close"),
                            "ma": extra.get("ma"),
                            "upper": extra.get("upper"),
                            "lower": extra.get("lower"),
                            "window": gf.get("config", {}).get("window"),
                            "ma_type": gf.get("config", {}).get("ma_type"),
                            "op": gf.get("config", {}).get("op"),
                            "buffer_pct": gf.get("config", {}).get("buffer_pct"),
                            "recovery_date": res.get("recovery_date"),
                        },
                        scope=event_scope,
                    )
                    return False
            return True

        def _lookup_force_exit_price(asset: str, cross_dt: pd.Timestamp, timing: str) -> Optional[tuple[float, float, pd.Timestamp]]:
            price_df = daily_price_history.get(asset)
            if price_df is None or price_df.empty:
                return None
            timing = (timing or execution_mode).lower()
            cross_dt = pd.to_datetime(cross_dt)
            before_or_on = price_df.loc[:cross_dt]
            if before_or_on.empty:
                return None
            if timing == "close":
                row = before_or_on.iloc[-1]
                return float(row["close_base"]), float(row["close"]), pd.to_datetime(row.name)
            future_rows = price_df.loc[price_df.index > cross_dt]
            if future_rows.empty:
                row = before_or_on.iloc[-1]
                return float(row["close_base"]), float(row["close"]), pd.to_datetime(row.name)
            row = future_rows.iloc[0]
            px_base = row.get("open_base", row.get("close_base"))
            px_ccy = row.get("open", row.get("close"))
            return float(px_base), float(px_ccy), pd.to_datetime(row.name)

        def _detect_force_exit(
            asset: Optional[str],
            entry_dt: Optional[pd.Timestamp],
            now_dt: pd.Timestamp,
        ) -> Optional[Dict[str, Any]]:
            if not guard_filters or asset is None or entry_dt is None:
                return None
            entry_dt = pd.to_datetime(entry_dt)
            now_dt = pd.to_datetime(now_dt)
            for gf in guard_filters:
                if gf.get("action") != "force_exit":
                    continue
                applies_to = gf.get("applies_to")
                if applies_to and asset not in applies_to:
                    continue
                state = gf.get("state") or {}
                cond = state.get("cond")
                if cond is None or cond.empty:
                    continue
                window = cond.loc[(cond.index >= entry_dt) & (cond.index <= now_dt)]
                crossed = window[window == False]  # noqa: E712
                if crossed.empty:
                    continue
                cross_dt = pd.to_datetime(crossed.index[0])
                timing = gf.get("force_exit_timing") or execution_mode
                price_info = _lookup_force_exit_price(asset, cross_dt, timing)
                if price_info is None:
                    continue
                exit_px_base, exit_px_ccy, exit_dt = price_info
                return {
                    "cross_date": cross_dt,
                    "exit_date": exit_dt,
                    "exit_price_portfolio": exit_px_base,
                    "exit_price_ccy": exit_px_ccy,
                    "guard": gf,
                    "timing": timing,
                }
            return None

        current: Optional[str] = None
        entry_price: Optional[float] = None
        entry_price_ccy: Optional[float] = None
        entry_date: Optional[pd.Timestamp] = None
        trade_records: List[Dict[str, Any]] = []
        rebalance_event_dates: List[pd.Timestamp] = []
        blocked_until: Optional[pd.Timestamp] = None

        for dt in idx:
            dt = pd.to_datetime(dt)
            day = dt.normalize()
            if visible_start is not None and dt < visible_start:
                continue
            if blocked_until is not None and dt < blocked_until:
                continue

            force_exit_ctx: Optional[Dict[str, Any]] = None
            if current is not None and entry_date is not None:
                force_exit_ctx = _detect_force_exit(current, entry_date, dt)

            sig_row = signal.loc[dt]
            sig_rank = sig_row.sort_values(ascending=False)

            # Decide target.
            target_symbol: Optional[str] = None
            target_entry: Optional[tuple[float, float, pd.Timestamp]] = None

            min_entry_ts: Optional[pd.Timestamp] = None
            if force_exit_ctx is not None:
                exit_ts = force_exit_ctx.get("exit_date")
                min_entry_ts = pd.to_datetime(exit_ts) if exit_ts is not None and not pd.isna(exit_ts) else None

            if current is None or force_exit_ctx is not None:
                # cash-style selection (absolute threshold)
                for sym, sig_val in sig_rank.items():
                    if pd.isna(sig_val):
                        continue
                    if float(sig_val) < entry_threshold:
                        break
                    entry_info = _exec_price_for(sym, dt)
                    if entry_info is None:
                        continue
                    px_base, px_ccy, ts = entry_info
                    if min_entry_ts is not None and ts < min_entry_ts:
                        # Ensure entry cannot occur before forced exit execution date.
                        if execution_mode == "next_open":
                            later = _lookup_first_open_on_or_after(sym, min_entry_ts)
                            if later is None:
                                continue
                            px_base, px_ccy, ts = later
                        else:
                            continue
                    if not _passes_guard(sym, ts, day):
                        continue
                    target_symbol = sym
                    target_entry = (px_base, px_ccy, ts)
                    break
            elif current is not None:
                cur_sig = sig_row.get(current)
                if cur_sig is not None and not pd.isna(cur_sig):
                    cur_sig_val = float(cur_sig)
                    exit_info = _exec_price_for(current, dt)
                    if exit_info is not None:
                        for sym, sig_val in sig_rank.items():
                            if sym == current or pd.isna(sig_val):
                                continue
                            sig_val_f = float(sig_val)
                            if sig_val_f <= cur_sig_val:
                                continue
                            if (sig_val_f - cur_sig_val) < switch_delta:
                                continue
                            entry_info = _exec_price_for(sym, dt)
                            if entry_info is None:
                                continue
                            px_base, px_ccy, ts = entry_info
                            # Enforce sell-then-buy ordering for next_open across markets.
                            if execution_mode == "next_open":
                                exit_ts = pd.to_datetime(exit_info[2])
                                if ts < exit_ts:
                                    later = _lookup_first_open_on_or_after(sym, exit_ts)
                                    if later is None:
                                        continue
                                    px_base, px_ccy, ts = later
                            if not _passes_guard(sym, ts, day):
                                continue
                            target_symbol = sym
                            target_entry = (px_base, px_ccy, ts)
                            break

            # Execute actions: forced exit, switch, enter, or hold.
            if force_exit_ctx is not None and current is not None and entry_price is not None and entry_date is not None:
                exit_px = _round_px(force_exit_ctx.get("exit_price_portfolio"))
                exit_px_ccy = _round_px(force_exit_ctx.get("exit_price_ccy"))
                exit_dt = force_exit_ctx.get("exit_date")
                cross_dt = force_exit_ctx.get("cross_date")
                cross_timing = force_exit_ctx.get("timing")

                trade_records.append(
                    {
                        "asset": current,
                        "currency": context.asset_currencies.get(current, context.base_currency),
                        "entry_date": entry_date,
                        "exit_date": exit_dt,
                        "entry_price_portfolio": _round_px(entry_price),
                        "exit_price_portfolio": exit_px,
                        "entry_price_ccy": _round_px(entry_price_ccy),
                        "exit_price_ccy": exit_px_ccy,
                        "return_pct_portfolio": exit_px / entry_price - 1
                        if entry_price and not pd.isna(exit_px)
                        else np.nan,
                        "return_pct_ccy": exit_px_ccy / entry_price_ccy - 1
                        if entry_price_ccy and not pd.isna(exit_px_ccy)
                        else np.nan,
                        "ma_guard_force_exit": True,
                        "ma_guard_force_exit_cross_date": cross_dt,
                        "ma_guard_force_exit_timing": cross_timing,
                    }
                )

                guard_scope = force_exit_ctx.get("guard", {}).get("event_scope")
                _emit_guard_event(
                    "ma_guard_force_exit",
                    {
                        "day": day,
                        "asset": current,
                        "entry_date": entry_date,
                        "cross_date": cross_dt,
                        "exit_date": exit_dt,
                        "ref_asset": force_exit_ctx.get("guard", {}).get("ref_asset"),
                        "window": force_exit_ctx.get("guard", {}).get("config", {}).get("window"),
                        "ma_type": force_exit_ctx.get("guard", {}).get("config", {}).get("ma_type"),
                        "op": force_exit_ctx.get("guard", {}).get("config", {}).get("op"),
                        "buffer_pct": force_exit_ctx.get("guard", {}).get("config", {}).get("buffer_pct"),
                    },
                    scope=guard_scope,
                )

                if exit_dt is not None and not pd.isna(exit_dt):
                    rebalance_event_dates.append(pd.to_datetime(exit_dt).normalize())

                current = None
                entry_price = None
                entry_price_ccy = None
                entry_date = None

                if execution_mode == "next_open" and exit_dt is not None and not pd.isna(exit_dt):
                    blocked_until = pd.to_datetime(exit_dt).normalize()
                else:
                    blocked_until = None

                # Optional re-entry after forced exit (cash-style selection).
                if target_symbol is not None and target_entry is not None:
                    px_base, px_ccy, ts = target_entry
                    current = target_symbol
                    entry_price = _round_px(px_base)
                    entry_price_ccy = _round_px(px_ccy)
                    entry_date = ts
                    if execution_mode == "next_open" and entry_date is not None and not pd.isna(entry_date):
                        blocked_until = pd.to_datetime(entry_date).normalize()
                continue

            if current is None and target_symbol is not None and target_entry is not None:
                # Enter from cash.
                px_base, px_ccy, ts = target_entry
                current = target_symbol
                entry_price = _round_px(px_base)
                entry_price_ccy = _round_px(px_ccy)
                entry_date = ts
                if execution_mode == "next_open" and entry_date is not None and not pd.isna(entry_date):
                    blocked_until = pd.to_datetime(entry_date).normalize()
                else:
                    blocked_until = None
                continue

            if current is not None and target_symbol is not None and target_symbol != current and target_entry is not None:
                # Switch: exit current, then enter target.
                exit_info = _exec_price_for(current, dt)
                if exit_info is None or entry_price is None or entry_date is None:
                    continue
                exit_px_base, exit_px_ccy, exit_dt = exit_info
                exit_px_base = _round_px(exit_px_base)
                exit_px_ccy = _round_px(exit_px_ccy)

                trade_records.append(
                    {
                        "asset": current,
                        "currency": context.asset_currencies.get(current, context.base_currency),
                        "entry_date": entry_date,
                        "exit_date": exit_dt,
                        "entry_price_portfolio": _round_px(entry_price),
                        "exit_price_portfolio": exit_px_base,
                        "entry_price_ccy": _round_px(entry_price_ccy),
                        "exit_price_ccy": exit_px_ccy,
                        "return_pct_portfolio": exit_px_base / entry_price - 1
                        if entry_price and not pd.isna(exit_px_base)
                        else np.nan,
                        "return_pct_ccy": exit_px_ccy / entry_price_ccy - 1
                        if entry_price_ccy and not pd.isna(exit_px_ccy)
                        else np.nan,
                        "ma_guard_force_exit": False,
                        "ma_guard_force_exit_cross_date": pd.NaT,
                        "ma_guard_force_exit_timing": None,
                    }
                )

                rebalance_event_dates.append(pd.to_datetime(exit_dt).normalize())

                px_base, px_ccy, ts = target_entry
                current = target_symbol
                entry_price = _round_px(px_base)
                entry_price_ccy = _round_px(px_ccy)
                entry_date = ts
                if execution_mode == "next_open" and entry_date is not None and not pd.isna(entry_date):
                    blocked_until = pd.to_datetime(entry_date).normalize()
                else:
                    blocked_until = None
                continue

        # Force-close any open position at the end to realize final NAV.
        if current is not None and entry_price is not None and entry_date is not None:
            price_df = daily_price_history.get(current)
            if price_df is not None and not price_df.empty:
                end_dt = context.end_date or price_df.index.max()
                end_dt = pd.to_datetime(end_dt)
                upto = price_df.loc[:end_dt]
                if not upto.empty:
                    row = upto.iloc[-1]
                    exit_dt = pd.to_datetime(row.name)
                    exit_px_base = _round_px(float(row["close_base"]))
                    exit_px_ccy = _round_px(float(row["close"]))
                    trade_records.append(
                        {
                            "asset": current,
                            "currency": context.asset_currencies.get(current, context.base_currency),
                            "entry_date": entry_date,
                            "exit_date": exit_dt,
                            "entry_price_portfolio": _round_px(entry_price),
                            "exit_price_portfolio": exit_px_base,
                            "entry_price_ccy": _round_px(entry_price_ccy),
                            "exit_price_ccy": exit_px_ccy,
                            "return_pct_portfolio": exit_px_base / entry_price - 1
                            if entry_price and not pd.isna(exit_px_base)
                            else np.nan,
                            "return_pct_ccy": exit_px_ccy / entry_price_ccy - 1
                            if entry_price_ccy and not pd.isna(exit_px_ccy)
                            else np.nan,
                            "ma_guard_force_exit": False,
                            "ma_guard_force_exit_cross_date": pd.NaT,
                            "ma_guard_force_exit_timing": None,
                        }
                    )
                    rebalance_event_dates.append(exit_dt.normalize())

        trades_df: Optional[pd.DataFrame] = pd.DataFrame(trade_records) if trade_records else None

        metrics, equity_series, curve_extras = build_equity_from_trades(
            trades_df,
            daily_price_history,
            execution_mode,
            visible_start,
            context.end_date,
            price_precision,
        )
        if trades_df is not None:
            metrics.update(_compute_pnl_ratio_by_asset(trades_df))

        extra: Dict[str, Any] = {
            "rebalance_dates": [pd.to_datetime(ts).normalize() for ts in sorted(set(rebalance_event_dates))],
            "lookback_days": lookback_days,
            "signal_ma_window": ma_window,
            "signal_ma_type": signal_ma_type if ma_window is not None else None,
            "entry_threshold": entry_threshold,
            "switch_delta": switch_delta,
        }
        extra.update(curve_extras)
        if guard_events:
            extra["ma_guard_events"] = guard_events

        return RunResult(
            metrics=metrics,
            equity_curve=equity_series,
            trades=trades_df,
            extra=extra,
            capabilities={"per_asset_trade_stats": True},
            metric_hooks=["per_asset_trade_stats"],
        )

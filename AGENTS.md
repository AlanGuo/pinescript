# Repository Guidelines

## Project Structure & Module Organization
- Root contains standalone TradingView Pine Script indicators/strategies (`*.pine`).
- `momentum_rotation/` holds the ETF rotation scripts (`daily.pine`, `weekly.pine`) and supporting notes like `anchor_holiday_rules.md`.
- `percentile_grid/` contains the grid-based percentile tools (`grid.pine`, `grid_percentile_panel.pine`).
- `.claude/` includes local editor/agent settings; `.git/` is standard git metadata.

## Build, Test, and Development Commands
This repo does not use a CLI build or test system. Development is done in TradingView:
- Open a `.pine` file in the TradingView Pine Editor and click **Add to chart** to compile/run.
- Use chart replays or timeframe changes to validate behavior.
If you want local linting/formatting, configure it in your editor; none is enforced here.

## Coding Style & Naming Conventions
- Pine Script v6 is used (see `//@version=6` at file tops).
- Indentation: 4 spaces; avoid tabs.
- Naming: `lowerCamelCase` for variables/functions (e.g., `lookbackDays`, `calcSignal()`), `snake_case` for filenames (e.g., `global_liquidity_index.pine`).
- Keep section headers as comment blocks (`// === Section ===`) to match existing files.

## Testing Guidelines
- No automated tests in this repo.
- Manual checks: compile in TradingView, ensure no warnings, and confirm plots/labels/alerts behave as expected.
- When changing signal logic, validate on multiple symbols/timeframes.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative summaries (e.g., `update`, `fix`). Keep messages concise and action-oriented.
- PRs should include:
  - A short description of the change and affected scripts/paths.
  - Any chart screenshots or before/after notes when visual output changes.
  - Steps to reproduce or validate in TradingView if behavior changes.

## Configuration & Safety Notes
- Avoid hard-coding symbols unless required; prefer input parameters.
- Keep default inputs conservative to reduce false alerts on live charts.

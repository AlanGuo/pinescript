Anchor Holiday Handling
=======================

背景
----

当周锚点日（Week Anchor）因节假日停市时，策略需要决定是否在节前补一次调仓，还是顺延到节后重新评估。为保持周度节奏又避免用过旧的价格换仓，脚本引入了“补换仓最小工作日间隔”与“保留空周开关”两套参数。

相关参数
--------

- `preserveEmptyWeeks`  
  - `true`: 任何被跳过的锚点周都会写入 anchor 记录；  
  - `false`: 只有当节前最后一个交易日确实晚于上一锚点日（存在“新交易日”）时，才补录一个锚点；整周都无交易则完全跳过。

- `catchupWorkdayThreshold`（默认 3）  
  - 计算上一真实换仓日 `D_prevTrade = anchorDate0` 与节前最后一个交易日 `D_currPrice = time[1]` 之间的工作日距离（仅数周一~周五）。  
  - 若 `distance > 阈值`：允许在节前价格上补一次调仓；  
  - 若 `distance ≤ 阈值`：认为离上次换仓太近，不做补仓，仅记录锚点。

判定流程
--------

1. **检测缺失周数**  
   - `skippedAnchorCount = countSkippedAnchors(...)`，只要大于 0 就进入补逻辑。

2. **是否存在“新交易日”**（仅当 `preserveEmptyWeeks=false` 时生效）  
   - `hasNewTradingSincePrevAnchor = time[1] > anchorDate0`；若为 `false`，直接跳过全部补录与调仓。

3. **决定补录次数**  
   - `preserveEmptyWeeks=true` → 补录全部 `skippedAnchorCount`。  
   - `preserveEmptyWeeks=false` → 仅在 `hasNewTradingSincePrevAnchor` 成立时补录 1 周；否则 0 周。

4. **决定是否执行补换仓**  
   - 仅对 `weekIdx = 0` 评估补仓；若 `catchupWorkdayGap > catchupWorkdayThreshold`，执行一次补动量 + 补换仓并记录价格；  
   - 其他 `weekIdx` 或 `allowCatchupRebalance=false` 时，只写 anchor，不触发调仓。

5. **记录与日志**  
   - 无论是否补仓，都调用 `shiftAnchorHistory` 写入当周的目标日期。  
   - 日志中包含：`workdayGap`、阈值、是否补仓、记录日期与目标 Anchor 日，便于回放。

场景示例
--------

| 场景 | 描述 | 结果 |
| --- | --- | --- |
| 正常锚点开市 | 当周锚点日有交易 | 正常换仓 |
| 锚点休市、次日非锚点（单周） | `skippedAnchorCount=1`，`workdayGap=1` | `workdayGap <= 阈值` → 不补，只记录 |
| 锚点休市、下一锚点直接开盘（单周） | 节后一开盘就是锚点日，`workdayGap` ≥ 4 | `workdayGap > 阈值` → 在节前价补一次，然后照常处理新锚点 |
| 锚点休市、跨两周、节后非锚点 | `skippedAnchorCount≥2` 且节前最后价离上一锚点很近 | 仅记录每周 anchor，完全顺延到节后 |
| 锚点休市、跨两周、节后为锚点日 | `skippedAnchorCount≥2` 且 `workdayGap > 阈值` | 节前价补一次后，节后锚点照常换仓 |

要点
----

- “补换仓”最多只在节前最后一个交易日执行一次，避免同价多次补仓。  
- `preserveEmptyWeeks` 决定 anchor 数组是否必须保持完整周序；关闭时完全空周不会占用 slot。  
- 所有决定都在日志中展示：`补换仓判定`、`补换仓周锚点`、`目标Anchor日` 等字段可帮助复盘。


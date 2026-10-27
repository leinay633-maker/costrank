# 数据来源、契约与缺陷

核实日期：**2026-10-09**。以下区分“官网/文档可核实”“代码适配已实现”“真实服务调用已验证”。本轮**没有调用真实行情 API，也没有下载社区行情包**。2026-10-12 补充：已用 BaoStock 实采 pilot；2026-10-13 补充：完整开发范围已采集封存。见验证记录文末。

## 来源选择

| 来源 | 本轮状态 | 主要缺陷 / 使用限制 |
|---|---|---|
| Qlib 官方预打包数据 | 官方 README 仍说明暂时停用；不作为运行命令依赖 | 不能继续照抄失效官方下载流程 |
| BaoStock 免费接口 | 原始日线、交易日历、证券基础信息、历史交易日证券快照的下载/重试/缓存适配已实现；mock 测试通过；2026-10-12 用 baostock 0.9.4 在本机实采 120 只 pilot，2026-10-13 采完 2016–2024 完整范围 3341 只，都通过 validate 与数据审计（见验证记录） | 数据修订/退市覆盖/历史 ST 发布时间未独立认证，无集合竞价深度；首版公司行为是参考价再投资近似 |
| chenditc/investment_data | Qlib 官方 README 推荐的社区替代入口；本轮没有下载或做字段适配 | README 说明混合多个数据来源。不可直接把社区复权价当作实际下单价；原始价、ST、停牌、退市和历史成分完整性均需另审计 |
| AKShare | 官方股票数据文档可查；**未实现独立采集器** | 是多站点接口集合，不是统一 PIT 数据质量保证；单位、复权和接口稳定性需逐项核验 |
| Tushare / 其他授权数据 | **未调用、未付费、未实现自动付费采集器**；`import-csv` 是真实可运行的切换边界 | 数据权限和预算由用户决定；不要把学生积分说明当作已获得权限。需要转换成以下统一契约并自证历史覆盖 |

BaoStock 采集采用 `adjustflag="3"` 的不复权日线。证券集合为基础证券表与每年首个交易日/最后日期历史证券快照的并集，保留有历史覆盖的退市证券，不按当前上市状态筛掉它们。历史快照是对照补充，不等于完整 PIT 指数成分历史。对每个证券按上市/退市区间取原始日线，任何一只请求失败或缺有效字段都会报错，而不是“成功下载谁就回测谁”。BaoStock 的 `outDate` 是摘牌日，即契约里不再有效的第一天，但它在这一天仍返回一行停牌占位（2026-10-12 pilot 的 7 只退市股都是如此）；采集时去掉摘牌日及以后的停牌行，其中若有正常交易行则报错，不猜退市日。

2026-10-12 全量采集（3341 只、637 万行）另有 7 行交易状态与量额矛盾，只在这一行自身字段能说明实际情况时修正：① 标为可交易、无量无额、四价都等于前收，按停牌处理（sh.600656 最后交易日之后、摘牌日之前的一天）；② 标为停牌、成交额为 0、四价等于前收，成交量却非零（停牌首日照抄前一日成交量，2 行），成交量清零；③ 标为停牌，但价格有波动、有量有额，且成交均价（额/量）落在当日最低价与最高价之间（退市整理期 4 只在 2024-06-13），改为可交易。其余矛盾仍由 validate 拒绝。每条修正连同原值写进 `metadata.json` 的 `provider_repairs`。

完整范围还查出两类问题，v0.2 采集器起处理（细节见验证记录 2026-10-13 一节）：

- 换代码：BaoStock 在新代码下保留全部旧历史，老代码又在换代码日“退市”（全量只有 sz.000022→sz.001872、sz.000043→sz.001914 两对，都在 2020 年前）。采集时老代码若与另一只上市日相同、且新代码覆盖它的全部交易日、四价和成交量至少 99% 一致，就当别名去掉，记进 metadata 的 `dropped_aliases`。
- 换股吸收合并与现金对价：`reference/delisting_resolutions.csv` 是按交易所和公司公告人工核实的处置表（每行附来源链接）。`collect --resolutions` 把它和 BaoStock 的 `outDate` 逐条核对（日期不一致就报错），落在本快照内的写进 `delistings.csv`。表里没有的摘牌仍按零回收。

`--limit-symbols N` 依证券代码 SHA-256 排序截前 N，目的是低成本检查流程，非随机统计代表样本或指数复刻。包含最后日期的证券候选集合会随采集终点变化，因此抽样 pilot 禁止用于最终冻结。正式开发必须 limit=0，并审查 `metadata.json` 的 limitations。

## 统一 CSV 契约

每个数据快照目录包含 `metadata.json`、`calendar.csv`、`instruments.csv`、`bars.csv.gz`、`actions.csv`；schema v2 起还有 `delistings.csv`，同样由清单绑定哈希并计入历史前缀哈希（v1 快照读作空表，前缀哈希算法不变）。UTF-8，日期统一 `YYYY-MM-DD`。原始数据不进 Git；清单存储文件 SHA-256，研究运行再次记录快照清单与历史前缀哈希。

| 文件 | 列 / 单位 |
|---|---|
| calendar.csv | `date`：完整交易所开市日，排序、唯一 |
| instruments.csv | `symbol,listed,delisted`：上市日；`delisted` 为不再有效的第一天，空表示尚无已知移除日期；生存区间 `[listed,delisted)` |
| bars.csv.gz | `date,symbol,open,high,low,close,preclose,volume,amount,tradable,is_st` |
| actions.csv | `date,symbol,split_ratio,cash_per_share,pay_date`：除权日、证券、股数乘数、每一旧股税前现金、支付日 |
| delistings.csv（v2） | `symbol,date,kind,acquirer,ratio,cash_per_share,settle_date`：摘牌时持有人拿到的对价。`date` 必须等于证券表的退市日；`share_exchange` 为每 1 股换吸并方 `ratio` 股、`settle_date` 为新增股份上市日（此前不可卖）；`cash` 为每股 `cash_per_share` 元、`settle_date` 到账 |

价格为**原始不复权人民币元/股**；`preclose` 必须是当天用于除权/价格参考的前收参考值，不是无条件昨日原始 close。`volume` 为股，`amount` 为人民币元，不能把“手”“千元”等单位原样塞进来。`tradable/is_st` 必须是历史当日 0/1，不接受当前状态替代。主板标识规则 `sh.60xxxx/sz.00xxxx`；首版不接科创板、创业板、北交所、基金或指数。

每只股票在有效生存区间内必须拥有每个交易日的行。停牌行必须明确 `tradable=0`，量额为零，允许把此前已知价格延续为标记价；首段从未观察到价格时保持未知，不从未来向前补价。缺行、重复行、未知证券、非交易日、活跃行情缺价格/量额、缺历史 ST、OHLC 关系不成立，均失败。暂停收盘标记不是成交许可。

支持数据范围 2015-08-01 至 2025-12-31。元数据需至少有 `source`、`corporate_actions`、`start`、`end`、`limitations`；`seal` 会写入 `schema_version=1` 与文件哈希。自定义导入示例 metadata：

```json
{
  "source": "USER_LICENSED_CSV",
  "corporate_actions": "explicit_events",
  "start": "2016-01-01",
  "end": "2024-12-31",
  "coverage": "Describe the historical security master and validation evidence here",
  "limitations": ["Dividend withholding taxes and exchange auction queues are not modeled"]
}
```

这是契约示例，不是已获得许可的声明。对应命令是 `costrank import-csv`，参数见 `--help`。导入器只本地读取、转换文件路径并校验，不会猜单位、猜缺失事件或使用付费 Token。

## 公司行为：两个模式都要讲清楚

**`reference_reinvestment_proxy`（免费首轮，必须显式 `--allow-proxy`）**：除权当日开盘前将旧持股数及待成交目标股数乘以 `previous_raw_close/current_preclose`。效果是以参考价合成再投资份额，使纯除权不被错误记成损失；会产生不可实际交易的合成小数股。它不是现金分红到账、税后权益、配股参与和送转登记的真实还原。每次变动写 events；禁止同时提供显式 actions，避免双算。即使这个模式获得正收益，仍只能称为受数据假设限制的探索结果。

**`explicit_events`**：记录旧仓的税前现金股息应收、股数乘数和付款日；应收款计入 NAV，不在支付前当作可花现金。除权当日更新股数，现金于首个 `day>=pay_date` 的交易日入账。必须逐证券逐日满足 `(上次原始close - cash_per_share)/split_ratio ≈ 当日preclose`（容差 0.02 元）；未知参考价跳变、缺事件、或凭空添加股息都会失败。送转登记延迟、股息税、配股出资与精确回收款尚未覆盖；`cash_per_share<0` 不支持，不能拿它冒充配股模型。合成测试覆盖的现金流恒等式，不等于已做真实除权日验证。

退市在首次不再上市时对持仓计零回收损失，记录金额，不做凭空按最后收盘卖出的乐观处理。该值是**保守估算/压力假设**，不是每只退市证券真实回收率；正式结论需要授权数据和对应权益事件支持。v2 快照里有处置记录的摘牌例外：换股所得吸并方股份记入持仓并加锁（交收前不可卖，吸并方尚未上市时按旧标记价结转），现金对价记应收、到期入账。零股送股等尾差忽略。

## 缓存、失败与重新取得历史数据

采集计划包括来源、起止日、抽样限制；相同未封存目录可以复用 `raw/` 继续，计划不同则拒绝。请求失败串行重连并最多重试 3 次，使用 30 秒 socket timeout，保留失败 JSONL；服务自身故障/字段变化仍可能使采集失败。无并发 socket 抢占，不使用成功子集替代声明股票范围。

一旦 `metadata.json` 已封存，禁止原目录重采。校验失败也保留现场：查清证券/日期与供应商原始响应，另建目录或修正独立 CSV 导入，保留旧现场。最终重新下载 2016–2025 时，若 2016–2024 历史发生修订，prefix hash 会拒绝；不要删冻结文件重做“干净最终测试”。先调查修订来源，必要时将这次研究标记为失败/受污染并另立协议。

## v0.4：全 A 快照（schema 3）

`costrank acquire` 只负责联网取数，每个请求一份原始文件，中断后按缓存续采；每发一页计一次请求，按北京日期落盘计数，到每日预算（默认 4.5 万，低于 BaoStock 每 IP 每天 5 万次）干净停下，次日重跑同一命令续采。`costrank assemble` 离线把原始缓存封存成 Parquet 快照，表与列见 `data.py` 的 `TABLES3`，前缀哈希按内容计算、与存储格式无关。`acquire --through <日期>` 只取该日及以前的条目（计划不变、缓存共用，之后的全量运行不重复请求）；`scripts/subset_raw.py` 把截至该日的原始缓存硬链接到计划截止日更早的新目录，供封存 2019 年及以前的调试快照。

- 日线与估值：`query_daily_history_k_AStock(date)` 一次返回某交易日全部 A 股（含停牌行）的未复权日 K、换手率和 `peTTM/pbMRQ/psTTM/pcfNcfTTM`。2026-10-13 抽 2016-06-30、2024-06-28 两天与 dev-4 的逐只缓存比对：主板代码集合相同（2358、3195 只），四价、前收、量额、交易状态、ST 逐值一致。完整采集后按 RESEARCH §9 的集合相等条件再核一次。别名代码、摘牌日占位行和三条供应商修正与主板采集器同规则。
- 除权事件：`query_daily_adjust_factor(date)` 给出当日有复权因子变化的股票，再按（代码, 年）查分红送转明细；能对上除权参考价（±0.02 元）的记为显式事件，其余前收跳变按参考价近似，每个跳变或事件的处理方式记在 `reconciliation` 表（`hybrid_explicit_proxy` 模式，`run` 同样要显式加 `--allow-proxy`）。送股（`dividStocksPs`）与转增（`dividReserveToStockPs`）分列，红利税只对送股按面值计。送转新增的股份从 `dividStockMarketDate`（红股上市日）起才能卖出：2016–2019 年 2180 个送转事件中，上交所的都在除权日之后（通常是下一个交易日），深交所都是除权日当天，2 个缺上市日的按除权日处理；schema 3 的 `actions.parquet` 为此增加可选的末列 `share_listing_date`（RESEARCH §9 补充九），账本在除权日把新增股份记为锁定股，到上市日释放（事件 `split_shares_listed`），此前卖单只到未锁定的部分。
- 行业：`query_stock_industry(date)` 返回该日最近一期周度快照（证监会分类），按每月第一个交易日取一次。沪深 300、中证 500 成分同样按月取；指数日线取 sh.000300、sh.000905、sh.000852、sh.000906，中证全指 sh.000985 在 BaoStock 没有数据。
- 30 分钟线：2020-01-02 起才有（2016、2019 年探测为空）。某代码第 y 年的分钟线，只在它截至 y 年年底已经进过全部 A 股 20 日平均成交额前 2500 时才取（每年取哪些只取决于该年年底以前的数据，延伸区间不会补出更早的年份），汇总成每日首末半小时价格、成交量额和已实现方差，用于容量网格的 VWAP 成交与参与率。
- 业绩预告、快报：按代码取全区间，发布日之后的下一个交易日起才进入因子；快报若被修订，BaoStock 只留最新版本，所以以发布日与更新日中较晚者为可用日（`available_date`）。这类按整段区间查询的缓存（含指数日线）按区间分段命名（`<代码>__<起>__<止>.csv`），组装时每条记录只从可用日所在分段读取，延伸区间时只新增分段。
- 退市：创业板 31 只、科创板 2 只 2016–2024 区间内退市，2026-10-13 由 grok 子代理按交易所摘牌公告核实，全部是无对价的强制退市，公告摘牌日与 BaoStock `outDate` 一致（逐只来源见 `reference/delisting_review_chinext_star.md`），处置表不增条目。2025 年保留区间内摘牌 29 只（不含改代码的 sz.300114→sz.302132），2026-10-14 冻结后、组装 2025 年快照前同样按公告核对（`reference/delisting_review_2025.md`）：24 只强制退市无对价；海通证券（换 0.62 股国泰君安）、中国重工（换 0.1339 股中国船舶）记换股，中航产融（3.54 元）、玉龙股份（13.20 元）、天茂集团（1.60 元）记现金选择权，5 行写入处置表。公告摘牌日与 `outDate` 全部一致。

## 板块交易规则（2016–2025，回测采用）

2026-10-13 由 grok 子代理按交易所规则和财税文件原文核对（出处见下文第 13–19 条）：

| 规则 | 主板 | 创业板 | 科创板 |
|---|---|---|---|
| 涨跌幅 | 10%；ST、*ST 5%（改为 10% 的 2026 年修订 2026-07-06 才施行） | 2020-08-23 前 10%、ST 5%；2020-08-24 起全部 20%（含存量与 ST） | 20%，含 ST |
| 涨跌停价 | 前收 ×（1 ± 比例）四舍五入到 0.01 元；离前收不足一个价位时为前收 ± 0.01 元 | 同左 | 同左 |
| 新股 | 2023-04-10 前首日 +44%/−36%，之后前 5 日不设限 | 改革后前 5 日不设限 | 前 5 日不设限 |
| 买入数量 | 100 股整数倍 | 100 股整数倍 | 单笔不少于 200 股，超出部分可按 1 股递增 |
| 卖出 | 余额不足 100 股须一次卖出 | 同左 | 单笔同样不少于 200 股；余额不足 200 股须一次卖出 |

红利税（个人）：财税〔2015〕101 号自 2015-09-08 起，持股 1 个月以内（含）实际税负 20%，1 个月至 1 年（含）10%，超过 1 年免；持股 1 年以内的派息时不扣，转让时由结算公司按持有期计算、证券公司扣收，先进先出，持有期按自然月从取得之日算到转让交割日前一日。送红股按面值计入股息所得、资本公积转增是否计税取决于来源，这两点的原件未打开（把握低），回测只对送股按面值计税、转增不计，结论里保留这一限定。长期停牌复牌、退市整理期首日、恢复/重新上市首日等特殊情形的价带不单独建模，开盘价超出常规价带的会话一律拒绝成交。

## 原始出处

1. Qlib 官方 README（数据停用说明、社区替代链接）：https://github.com/microsoft/qlib
2. 社区数据来源与发布说明：https://github.com/chenditc/investment_data
3. BaoStock 维护方包说明及 raw K 线调用示例：https://pypi.org/project/baostock/
4. BaoStock API / 日线文档：https://www.baostock.com/mainContent?file=pythonAPI.md ；https://www.baostock.com/mainContent?file=stockKData.md
5. Qlib 原生 DataHandler 文档：https://qlib.readthedocs.io/en/latest/component/data.html
6. AKShare 官方股票文档：https://akshare.akfamily.xyz/data/stock/stock.html
7. Tushare 股票接口文档：https://tushare.pro/document/2?doc_id=25 ；学生积分申请说明：https://tushare.pro/document/1?doc_id=360
8. 上交所 2023-08-27 交易费用调整通知：https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20230827_5725662.shtml
9. 税务总局证券交易印花税减半公告：https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html
10. 中信证券 2023-08-28 实施通知（卖方 0.1%→0.05%）：https://www.citics.com/newsite/tzgg/ywgg/202308/t20230828_1181105.html
11. 湘财证券 2022-04-29 过户费实施通知（引用中国结算，双边 0.02‰→0.01‰）：https://www.xcsc.com/main/a/20220429/1022871784.shtml
12. 上交所 2026 年修订规则通知（2026-07-06 起施行，首版未外推）：https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml
13. 上交所《交易规则（2023 年修订）》（第 3.3.8、3.3.13、3.3.17、4.4.10、6.1.6–6.1.8 条）：https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml
14. 上交所《交易规则（2026 年修订）》新闻稿（主板风险警示 5% 改 10%，2026-07-06 施行）：https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml
15. 深交所《创业板交易特别规定》（深证上〔2020〕515 号）：https://docs.static.szse.cn/www/lawrules/rule/repeal/rules/P020231230545310237980.pdf
16. 深交所交易规则（2013 年修订文本，第 3.3.13–3.3.17 条）：http://docs.static.szse.cn/www/disclosure/notice/W020180328432928783546.pdf
17. 上交所《科创板股票交易特别规定》（上证发〔2019〕23 号）：https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10118601/files/f6fc4a1d4c1f469183a013c4dc36a535.pdf
18. 财税〔2015〕101 号及政策问答：https://guangdong.chinatax.gov.cn/gdsw/grsdsgg_hmqsc_pyzbsc_ssgs/2021-08/31/content_0e7cfaeb5b7e45af9d0137ad2c52447f.shtml
19. 财税〔2012〕85 号：https://shanghai.chinatax.gov.cn/zcfw/zcfgk/grsds/201211/t400938.html

网页可访问、文档列出接口，不等于本轮已经验证该账户/该网络能取得完整历史数据。佣金 3 bps/最低 5 元、滑点网格、ADV 参与率、风险预算、零回收率均是研究设定或估算，不是从这些来源引用的实测值。

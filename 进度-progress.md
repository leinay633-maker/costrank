# 进度

### 2026-10-27 15:48 · 本地 Claude Code · main
- 改动：建公开仓库。内容是私有开发仓库 cost-aware-stock-ranking 在 v04 分支 187f281 的快照，用开发仓库的 `scripts/export_public.py` 导出：不含开发日志、验证记录、上机手册和数据采集工作流；服务器名、依赖锁文件的下载源和指向未公开文档的引用按导出规则改写（锁文件只换成 PyPI 官方地址，版本和哈希不变）；README、AGENTS、LICENSE 是公开版专用的。
- 验证：导出脚本词表自查无命中；`uv.lock` 的 2027 个下载地址在 files.pythonhosted.org 上全部返回 200；`pytest -q`（Windows，Python 3.12）：245 通过 / 1 跳过（未装 qlib），与开发仓库相同；独立子代理逐文件审核 115 个文件，未发现泄露，README 的 4 处口径问题已按 `docs/RESEARCH.md` 原文改正。测试跑在 README 改正之前，之后只改了 README。
- 未完成 / 下一步：开发仓库有新版本时重新导出、在这里提交，不在这里直接改代码；推送后看 GitHub Actions 的 tests 结果。

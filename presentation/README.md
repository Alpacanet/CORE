# AVE / TRM / LLM 比较汇报

[core_comparison.pptx](core_comparison.pptx) 是可编辑的 PowerPoint 汇报，`core_comparison.tex` 是从相同数据生成的 Beamer 参考源文件。

PPT 共 9 页，保留研究问题、CORE + LLM 结构、实验设置、三模型比较、结论边界、表示空间差异、训练成本和后续诊断。实测数据仍来自 2026-10-02 / 2026-10-05 的实验记录；整理文件和幻灯片不代表新增实验。

## 文件结构

```text
presentation/
  core_comparison.pptx
  core_comparison.tex
  build.mjs
  build.ps1
  data/results.yaml
```

## 更新与构建

修改 `data/results.yaml` 中的会议信息、`experiment.models` 和 `planning`，然后在项目根目录运行：

```powershell
.\presentation\build.ps1
```

构建使用 Codex Desktop 的 Node.js、Python 和 `@oai/artifact-tool`。脚本自动定位已安装的 presentations 技能版本，也可通过 `-RuntimeRoot` / `-SkillRoot` 指定环境路径。数据文件使用 JSON 语法，是合法的 YAML 1.2，不依赖额外 YAML 解析器。

生成器先在 `.build/` 中导出草稿，核验结构、几何尺寸、可编辑表格和图表，再复制到最终文件。每页预览与验证记录位于 `.build/` / `.validated/`，这些构建缓存不纳入版本管理。

只用实测结果更新数值，并保留每个模型的 `source`。计划与假设必须标明尚未执行。当前是单 seed 验证比较，LLM 全量运行在完成 7 轮后中断，三组匹配实验均没有最终测试集结果。

完整实验证据见 [README_LLM.md](../README_LLM.md)，训练操作见 [README_TRAINING.md](../README_TRAINING.md)。历史 `results/phase1/` 目录保留原名称，以维持实测证据和检查点路径。

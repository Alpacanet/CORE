import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const projectRoot = process.env.PRESENTATION_PROJECT_ROOT;
const skillDir = process.env.PRESENTATION_SKILL_DIR;
const runtimePython = process.env.PRESENTATION_RUNTIME_PYTHON;
if (!projectRoot || !skillDir || !runtimePython) {
  throw new Error("Run presentation/build.ps1 so the task-specific runtime paths are configured.");
}

const presentationDir = path.join(projectRoot, "presentation");
const buildDir = path.join(presentationDir, ".build");
const validatedDir = path.join(presentationDir, ".validated");
const renderDir = path.join(buildDir, "rendered");
const dataPath = path.join(presentationDir, "data", "results.yaml");
const finalPptx = path.join(presentationDir, "core_comparison.pptx");
const texPath = path.join(presentationDir, "core_comparison.tex");
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(validatedDir, { recursive: true });
await fs.rm(renderDir, { recursive: true, force: true });
await fs.mkdir(renderDir, { recursive: true });

const data = JSON.parse(await fs.readFile(dataPath, "utf8"));
const utils = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = utils;
const font = resolvePresentationFont({ fontFamily: "Aptos" });

const W = 1280;
const H = 720;
const C = {
  bg: "#FCFDFD",
  ink: "#15242B",
  muted: "#5F6F75",
  faint: "#E7EFF1",
  pale: "#EAF5F6",
  accent: "#087F8C",
  accentDark: "#075B63",
  warn: "#B56536",
  red: "#9A3D45",
  green: "#2D7A5B",
  white: "#FFFFFF",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left, top, width, height },
    fill: opts.fill ?? "none",
    line: opts.line ?? { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: opts.fontSize ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.alignment ?? "left",
    autoFit: "none",
  };
  return shape;
}

function addBox(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: opts.geometry ?? "roundRect",
    name: opts.name,
    position: { left, top, width, height },
    fill: opts.fill ?? C.white,
    line: opts.line ?? { style: "solid", fill: opts.lineColor ?? C.faint, width: opts.lineWidth ?? 1.2 },
    borderRadius: opts.radius ?? 12,
  });
  if (text) {
    shape.text = text;
    shape.text.style = {
      typeface: font,
      fontSize: opts.fontSize ?? 21,
      bold: opts.bold ?? false,
      color: opts.color ?? C.ink,
      alignment: opts.alignment ?? "center",
      autoFit: "none",
    };
  }
  return shape;
}

function connect(slide, from, to, opts = {}) {
  return slide.shapes.connect(from, to, {
    kind: opts.kind ?? "elbow",
    fromSide: opts.fromSide,
    toSide: opts.toSide,
    line: { style: opts.style ?? "solid", fill: opts.color ?? C.accent, width: opts.width ?? 2 },
    head: { type: "none" },
    tail: opts.head === false ? { type: "none" } : { type: "triangle", width: "sm", length: "sm" },
  });
}

function addHeader(slide, title, number) {
  slide.background.fill = C.bg;
  addText(slide, title, 64, 38, 1100, 58, { fontSize: 42, bold: true });
  slide.shapes.add({ geometry: "line", position: { left: 64, top: 105, width: 1152, height: 0 }, line: { style: "solid", fill: C.faint, width: 1.2 }, fill: "none" });
  slide.shapes.add({ geometry: "line", position: { left: 64, top: 105, width: (1152 / 9) * number, height: 0 }, line: { style: "solid", fill: C.accent, width: 3 }, fill: "none" });
  addText(slide, "CORE + LLM  •  Research progress", 64, 682, 420, 20, { fontSize: 14, color: C.muted });
  addText(slide, String(number).padStart(2, "0"), 1160, 679, 56, 22, { fontSize: 15, color: C.muted, alignment: "right" });
}

function addSectionLabel(slide, text, left, top, width = 220, color = C.accent) {
  addText(slide, text.toUpperCase(), left, top, width, 24, { fontSize: 15, bold: true, color });
}

function setNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
}

function styleTable(table, headerFill = C.accentDark) {
  table.borders.assign({ style: "solid", fill: C.faint, width: 1 });
  for (let r = 0; r < table.rows.count; r += 1) {
    for (let c = 0; c < table.columns.count; c += 1) {
      const cell = table.getCell(r, c);
      cell.text.style = { typeface: font, fontSize: 18, color: C.ink, alignment: c === 0 ? "left" : "center" };
    }
  }
  for (let c = 0; c < table.columns.count; c += 1) {
    const cell = table.getCell(0, c);
    cell.fill = headerFill;
    cell.text.style = { typeface: font, fontSize: 18, bold: true, color: C.white, alignment: c === 0 ? "left" : "center" };
  }
}

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  slide.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 18, height: H }, fill: C.accent, line: { fill: "none", width: 0 } });
  addText(slide, data.meta.title, 86, 178, 980, 130, { fontSize: 58, bold: true, color: C.ink });
  addText(slide, "Research Progress", 88, 320, 620, 58, { fontSize: 34, color: C.accentDark });
  slide.shapes.add({ geometry: "line", position: { left: 88, top: 404, width: 250, height: 0 }, line: { style: "solid", fill: C.accent, width: 4 }, fill: "none" });
  addText(slide, data.meta.subtitle, 88, 432, 620, 38, { fontSize: 24, color: C.muted });
  addText(slide, `${data.meta.presenter}\n${data.meta.meeting_date}  ·  ${data.meta.meeting_label}`, 88, 565, 620, 72, { fontSize: 20, color: C.muted });
  addText(slide, "pure item IDs  |  no names, brands, descriptions, or semantic metadata", 745, 585, 450, 52, { fontSize: 17, color: C.accentDark, alignment: "right" });
  setNotes(slide, ["Recurring research-meeting deck generated from presentation/data/results.yaml.", `Sources: ${data.sources.experiment_record}; ${data.sources.topic_note}`]);
}

// Slide 2
{
  const slide = presentation.slides.add(); addHeader(slide, "Research question", 2);
  addText(slide, "Can pretrained LLM sequence modeling transfer to pure-ID session recommendation?", 100, 145, 1080, 90, { fontSize: 37, bold: true, alignment: "center", color: C.accentDark });
  const qs = [
    "Can an LLM process learned item-ID embeddings?",
    "Can it improve next-item ranking over CORE?",
    "Does language pretraining itself add value?",
    "Does the value concentrate on unseen-in-session exploration?",
  ];
  qs.forEach((q, i) => {
    addText(slide, String(i + 1).padStart(2, "0"), 128, 286 + i * 66, 54, 36, { fontSize: 20, bold: true, color: C.accent });
    addText(slide, q, 196, 280 + i * 66, 900, 44, { fontSize: 24 });
  });
  addBox(slide, "Pure-ID constraint: no item name, brand, description, category, or semantic metadata", 146, 574, 988, 58, { fill: C.pale, lineColor: C.accent, fontSize: 20, bold: true, color: C.accentDark });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, research questions and pure-ID definition.`, `Source: ${data.sources.topic_note}, pure-ID scenario revision.`]);
}

// Slide 3
{
  const slide = presentation.slides.add(); addHeader(slide, "CORE + LLM architecture", 3);
  const history = addBox(slide, "Click history", 72, 170, 190, 62, { fill: C.pale, lineColor: C.accent, bold: true });
  const emb = addBox(slide, "Learned item embedding\n100 dimensions", 325, 160, 220, 82, { bold: true });
  const core = addBox(slide, "COREave\nsimple baseline", 650, 142, 220, 78, { fill: C.white });
  const proj = addBox(slide, "Projection + LayerNorm", 650, 266, 220, 64, { fill: C.white });
  const qwen = addBox(slide, "Qwen3-0.6B + LoRA\npretrained sequence encoder", 935, 250, 250, 92, { fill: C.pale, lineColor: C.accent, bold: true });
  const gate = addBox(slide, "Gated fusion", 650, 428, 220, 64, { fill: C.white, bold: true });
  const rank = addBox(slide, "Full-item ranking", 935, 428, 250, 64, { fill: C.accentDark, lineColor: C.accentDark, color: C.white, bold: true });
  connect(slide, history, emb, { kind: "straight", fromSide: "right", toSide: "left" });
  connect(slide, emb, core, { fromSide: "right", toSide: "left" });
  connect(slide, emb, proj, { fromSide: "right", toSide: "left" });
  connect(slide, proj, qwen, { kind: "straight", fromSide: "right", toSide: "left" });
  connect(slide, core, gate, { fromSide: "bottom", toSide: "top" });
  connect(slide, qwen, gate, { fromSide: "bottom", toSide: "right" });
  connect(slide, gate, rank, { kind: "straight", fromSide: "right", toSide: "left" });
  addText(slide, "LLM input", 936, 218, 180, 24, { fontSize: 15, bold: true, color: C.accent });
  addText(slide, "Continuous learned embeddings enter through inputs_embeds. Item IDs never become text tokens.", 174, 570, 930, 62, { fontSize: 24, bold: true, alignment: "center", color: C.accentDark });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, model implementation section.`, "The Qwen language embedding table does not process item IDs in this architecture."]);
}

// Slide 4
{
  const slide = presentation.slides.add(); addHeader(slide, "Experimental setup", 4);
  const table = slide.tables.add({
    rows: 8, columns: 2, left: 80, top: 142, width: 670, height: 446,
    columnTracks: [{ mode: "fr", value: 1.15 }, { mode: "fr", value: 1.85 }],
    values: [
      ["Setting", "Value"],
      ["Dataset", data.experiment.dataset],
      ["Train / valid / test", `${data.experiment.split.train.toLocaleString()} / ${data.experiment.split.valid.toLocaleString()} / ${data.experiment.split.test.toLocaleString()}`],
      ["Items", `${data.experiment.items_including_padding.toLocaleString()} including padding ID 0`],
      ["Maximum session length", String(data.experiment.max_session_length)],
      ["Ranking", "Full catalog; padding excluded; repeat items allowed"],
      ["Metrics", "Recall@10/20 and MRR@10/20"],
      ["Checkpoint selection", data.experiment.metric_selection],
    ],
  });
  styleTable(table);
  addSectionLabel(slide, "Models", 820, 154, 180);
  ["COREave", "COREtrm", "CORE + Qwen3-0.6B + LoRA"].forEach((m, i) => {
    addText(slide, String(i + 1), 830, 210 + i * 76, 30, 30, { fontSize: 18, bold: true, color: C.accent });
    addText(slide, m, 875, 204 + i * 76, 315, 42, { fontSize: 22, bold: i === 2 });
  });
  addBox(slide, data.experiment.comparison_label, 816, 470, 376, 72, { fill: C.pale, lineColor: C.accent, fontSize: 20, color: C.accentDark, bold: true });
  addText(slide, "No matched full test result yet", 838, 562, 334, 30, { fontSize: 18, color: C.warn, bold: true, alignment: "center" });
  setNotes(slide, [`Sources: ${data.sources.experiment_record}; saved manifest and metrics under each model source path in results.yaml.`, "All counts and evaluation settings come from the repository's recorded Phase 1 runs."]);
}

// Slide 5
{
  const slide = presentation.slides.add(); addHeader(slide, "Current result", 5);
  const m = data.experiment.models;
  const table = slide.tables.add({
    rows: 4, columns: 3, left: 86, top: 166, width: 746, height: 286,
    columnTracks: [{ mode: "fr", value: 1.7 }, { mode: "fr", value: 1 }, { mode: "fr", value: 1 }],
    values: [
      ["Model", "Recall@20", "MRR@20"],
      [m.coreave.label, m.coreave.recall20.toFixed(4), m.coreave.mrr20.toFixed(4)],
      [m.coretrm.label, m.coretrm.recall20.toFixed(4), m.coretrm.mrr20.toFixed(4)],
      [m.llm.label, m.llm.recall20.toFixed(4), m.llm.mrr20.toFixed(4)],
    ],
  });
  styleTable(table);
  for (let c = 0; c < 3; c += 1) table.getCell(3, c).fill = C.pale;
  const rel = ((m.llm.mrr20 - m.coreave.mrr20) / m.coreave.mrr20) * 100;
  addText(slide, "LLM vs CORE", 892, 174, 276, 28, { fontSize: 17, bold: true, color: C.muted, alignment: "center" });
  addText(slide, `${rel.toFixed(2)}%`, 872, 214, 316, 90, { fontSize: 64, bold: true, color: C.red, alignment: "center" });
  addText(slide, "MRR@20", 922, 305, 220, 32, { fontSize: 20, color: C.muted, alignment: "center" });
  addText(slide, "Current default configuration\ndoes not outperform CORE.", 858, 370, 346, 80, { fontSize: 26, bold: true, color: C.ink, alignment: "center" });
  addBox(slide, `Diagnostic signal: Recall@20 reached ${m.llm.later_recall20.toFixed(4)} at epoch ${m.llm.later_recall20_epoch}, while MRR@20 fell to ${m.llm.later_mrr20.toFixed(4)}. This was not the selected checkpoint.`, 108, 515, 1064, 78, { fill: C.pale, lineColor: C.accent, fontSize: 19, color: C.accentDark });
  addText(slide, "Validation only · seed 2020 · LLM run interrupted after 7 completed epochs", 205, 620, 870, 26, { fontSize: 16, color: C.muted, alignment: "center" });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, matched validation comparison.`, `Run sources: ${m.coreave.source}; ${m.coretrm.source}; ${m.llm.source}.`, "The relative difference is descriptive and is not a significance test."]);
}

// Slide 6
{
  const slide = presentation.slides.add(); addHeader(slide, "Current findings", 6);
  const xs = [100, 470, 840];
  const labels = ["ENGINEERING FEASIBILITY", "RECOMMENDATION QUALITY", "ATTRIBUTION"];
  const marks = ["✓", "×", "?"];
  const colors = [C.green, C.red, C.warn];
  const body = [
    "Pure-ID embeddings pass through pretrained Qwen.\n\nTraining, full-catalog validation, checkpointing, and recovery work.",
    "The current configuration trails both matched CORE baselines on validation MRR@20.",
    "Architecture and language pretraining remain confounded.\n\nThe random-backbone full control has not been run.",
  ];
  xs.forEach((x, i) => {
    addText(slide, marks[i], x, 160, 90, 84, { fontSize: 64, bold: true, color: colors[i], alignment: "center" });
    addText(slide, labels[i], x - 25, 258, 300, 28, { fontSize: 16, bold: true, color: colors[i], alignment: "center" });
    addText(slide, body[i], x - 20, 310, 310, 188, { fontSize: 22, color: C.ink, alignment: "center" });
  });
  addBox(slide, "This experiment tests direct representation transfer. Explicit intent reasoning remains untested.", 165, 562, 950, 62, { fill: C.pale, lineColor: C.accent, fontSize: 22, bold: true, color: C.accentDark });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, current conclusion and limitations.`, "The random_lora full comparison remains untested; smoke runs do not count as effectiveness evidence."]);
}

// Slide 7
{
  const slide = presentation.slides.add(); addHeader(slide, "Representation mismatch", 7);
  addText(slide, "A pretrained language model receives a completely new alphabet", 140, 134, 1000, 58, { fontSize: 34, bold: true, alignment: "center", color: C.accentDark });
  const lang = addBox(slide, "Language pretraining\n\nwords\nsyntax\nsemantic structure", 130, 240, 310, 250, { fill: C.white, lineColor: C.faint, fontSize: 24, bold: true });
  const mismatch = addBox(slide, "representation\nmismatch", 510, 300, 260, 118, { fill: C.pale, lineColor: C.accent, fontSize: 25, bold: true, color: C.accentDark });
  const ids = addBox(slide, "Pure-ID input\n\narbitrary labels\nlearned vectors\nbehavioral signal only", 840, 240, 310, 250, { fill: C.white, lineColor: C.faint, fontSize: 24, bold: true });
  connect(slide, lang, mismatch, { kind: "straight", fromSide: "right", toSide: "left", head: false, color: C.muted });
  connect(slide, mismatch, ids, { kind: "straight", fromSide: "right", toSide: "left", head: false, color: C.muted });
  addText(slide, "Preliminary implication", 234, 548, 260, 25, { fontSize: 16, bold: true, color: C.warn });
  addText(slide, "Sequence ability may require behavioral structure or task specialization before it transfers.", 234, 582, 840, 48, { fontSize: 24, bold: true });
  setNotes(slide, [`Sources: ${data.sources.experiment_record}; ${data.sources.topic_note}.`, "This slide states a preliminary implication, not an experimental conclusion."]);
}

// Slide 8
{
  const slide = presentation.slides.add(); addHeader(slide, "Compute consideration", 8);
  const m = data.experiment.models;
  const chart = slide.charts.add("bar", {
    position: { left: 82, top: 152, width: 740, height: 430 },
    categories: ["COREave", "COREtrm", "Qwen + LoRA"],
    series: [{ name: "Seconds per epoch", values: [m.coreave.epoch_time_sec, m.coretrm.epoch_time_sec, m.llm.epoch_time_sec], fill: C.accent }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 65 },
    hasLegend: false,
    xAxis: { visible: true, title: "Recorded training seconds per epoch", min: 0, max: 4000, majorUnit: 1000, majorGridlines: { style: "solid", fill: C.faint, width: 1 }, textStyle: { fontSize: 16, fill: C.muted } },
    yAxis: { visible: true, textStyle: { fontSize: 18, fill: C.ink }, line: { fill: "none", width: 0 } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0", textStyle: { fontSize: 17, fill: C.ink, bold: true } },
    chartFill: C.bg,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: C.bg,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: font });
  const ratio = m.llm.epoch_time_sec / m.coreave.epoch_time_sec;
  addText(slide, `${ratio.toFixed(0)}×`, 906, 170, 250, 92, { fontSize: 66, bold: true, color: C.red, alignment: "center" });
  addText(slide, "slower than COREave\nin recorded training time", 882, 268, 300, 66, { fontSize: 21, color: C.ink, alignment: "center" });
  addBox(slide, "Recorded training cost must be considered alongside ranking quality.", 856, 388, 352, 100, { fill: C.pale, lineColor: C.accent, fontSize: 21, color: C.accentDark, bold: true });
  addText(slide, "Recorded training only\nNot a speed benchmark", 916, 518, 232, 52, { fontSize: 18, bold: true, color: C.warn, alignment: "center" });
  addText(slide, "Same GPU model; background load and IDE mode were not controlled as a strict speed benchmark.", 148, 618, 984, 28, { fontSize: 15, color: C.muted, alignment: "center" });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, recorded average training seconds per epoch.`, "The training times are observations, not a controlled speed benchmark."]);
}

// Slide 9
{
  const slide = presentation.slides.add(); addHeader(slide, "Immediate plan", 9);
  const now = addBox(slide, "NOW", 72, 154, 112, 56, { fill: C.accentDark, lineColor: C.accentDark, color: C.white, bold: true });
  const actions = [
    ["1", data.planning.immediate_actions[0]],
    ["2", data.planning.immediate_actions[1]],
    ["3", data.planning.immediate_actions[2]],
    ["4", data.planning.immediate_actions[3]],
  ];
  const actionBoxes = actions.map((a, i) => {
    addText(slide, a[0], 246 + i * 245, 146, 34, 28, { fontSize: 16, bold: true, color: C.accent, alignment: "center" });
    return addBox(slide, a[1], 220 + i * 245, 180, 205, 86, { fill: i === 2 ? C.pale : C.white, lineColor: i === 2 ? C.accent : C.faint, fontSize: 20, bold: true });
  });
  connect(slide, now, actionBoxes[0], { kind: "straight", fromSide: "right", toSide: "left" });
  for (let i = 0; i < actionBoxes.length - 1; i += 1) connect(slide, actionBoxes[i], actionBoxes[i + 1], { kind: "straight", fromSide: "right", toSide: "left", head: false, color: C.faint });
  addText(slide, "Decision after validation", 74, 340, 330, 32, { fontSize: 23, bold: true, color: C.accentDark });
  const positive = addBox(slide, "Positive, stable signal", 115, 405, 260, 70, { fill: C.pale, lineColor: C.accent, bold: true });
  const noGain = addBox(slide, "No stable gain", 115, 535, 260, 70, { fill: C.white, lineColor: C.faint, bold: true });
  const scale = addBox(slide, "Paired seeds\nRandom-backbone control\nLocked final-test evaluation", 535, 388, 400, 122, { fill: C.white, lineColor: C.faint, fontSize: 21 });
  const shrink = addBox(slide, "Shrink the LLM direction", 535, 540, 400, 66, { fill: C.white, lineColor: C.faint, fontSize: 22, bold: true });
  connect(slide, positive, scale, { kind: "straight", fromSide: "right", toSide: "left" });
  connect(slide, noGain, shrink, { kind: "straight", fromSide: "right", toSide: "left", color: C.muted });
  addBox(slide, "Research goal: test recommendation gains and pretraining value under matched budgets.", 974, 396, 248, 204, { fill: C.accentDark, lineColor: C.accentDark, color: C.white, fontSize: 22, bold: true });
  setNotes(slide, [`Source: ${data.sources.experiment_record}, recommended diagnostics and controls.`, "Listed comparisons are planned. Refactoring does not add new scientific results."]);
}

function texEscape(value) {
  return String(value)
    .replaceAll("\\", "\\textbackslash{}")
    .replaceAll("&", "\\&")
    .replaceAll("%", "\\%")
    .replaceAll("$", "\\$")
    .replaceAll("#", "\\#")
    .replaceAll("_", "\\_")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}");
}

const m = data.experiment.models;
const rel = ((m.llm.mrr20 - m.coreave.mrr20) / m.coreave.mrr20) * 100;
const tex = String.raw`\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{default}
\usefonttheme{professionalfonts}
\usepackage{helvet}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{accent}{HTML}{087F8C}
\definecolor{accentdark}{HTML}{075B63}
\definecolor{ink}{HTML}{15242B}
\definecolor{muted}{HTML}{5F6F75}
\setbeamercolor{normal text}{fg=ink,bg=white}
\setbeamercolor{frametitle}{fg=ink,bg=white}
\setbeamercolor{structure}{fg=accent}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{frametitle}{\vspace{0.3em}\insertframetitle\par\vspace{-0.2em}\color{accent}\rule{\textwidth}{0.6pt}}
\setbeamertemplate{footline}{\hfill\color{muted}\scriptsize CORE + LLM \quad \insertframenumber/\inserttotalframenumber\hspace{1.2em}\vspace{0.7em}}
\newcommand{\status}[1]{\textcolor{accentdark}{\textbf{#1}}}
\title{${texEscape(data.meta.title)}\\[0.35em]\Large Research Progress}
\subtitle{${texEscape(data.meta.subtitle)}}
\author{${texEscape(data.meta.presenter)}}
\date{${texEscape(data.meta.meeting_date)} -- ${texEscape(data.meta.meeting_label)}}
\begin{document}
\begin{frame}[plain]\titlepage\end{frame}

\begin{frame}{Research question}
\centering\Large\textbf{Can pretrained LLM sequence modeling transfer to pure-ID session recommendation?}
\vspace{1em}
\begin{enumerate}\normalsize
\item Can an LLM process learned item-ID embeddings?
\item Can it improve next-item ranking over CORE?
\item Does language pretraining itself add value?
\item Does value concentrate on unseen-in-session exploration?
\end{enumerate}
\vfill\status{Pure-ID: no names, brands, descriptions, or semantic metadata.}
\end{frame}

\begin{frame}{CORE + LLM architecture}
\centering
\begin{tikzpicture}[node distance=9mm and 11mm, every node/.style={font=\small}, box/.style={draw=accent!45,rounded corners,minimum height=8mm,minimum width=25mm,align=center}, arr/.style={-Latex,thick,draw=accent}]
\node[box,fill=accent!8] (hist) {Click history};
\node[box,right=of hist] (emb) {Learned item embedding};
\node[box,above right=of emb] (core) {COREave};
\node[box,below right=of emb] (proj) {Projection};
\node[box,fill=accent!8,right=of proj] (qwen) {Qwen3-0.6B + LoRA};
\node[box,right=of core] (gate) {Gated fusion};
\node[box,fill=accentdark,text=white,right=of gate] (rank) {Full-item ranking};
\draw[arr] (hist)--(emb); \draw[arr] (emb)--(core); \draw[arr] (emb)--(proj); \draw[arr] (proj)--(qwen); \draw[arr] (core)--(gate); \draw[arr] (qwen)--(gate); \draw[arr] (gate)--(rank);
\end{tikzpicture}
\vfill\textbf{Continuous learned embeddings enter through \texttt{inputs\_embeds}. Item IDs never become text tokens.}
\end{frame}

\begin{frame}{Experimental setup}
\begin{tabular}{ll}\toprule Setting & Value\\\midrule
Dataset & ${texEscape(data.experiment.dataset)}\\
Train / valid / test & ${data.experiment.split.train.toLocaleString()} / ${data.experiment.split.valid.toLocaleString()} / ${data.experiment.split.test.toLocaleString()}\\
Items & ${data.experiment.items_including_padding.toLocaleString()} including padding\\
Max session length & ${data.experiment.max_session_length}\\
Metrics & Recall@10/20, MRR@10/20\\
Selection & ${texEscape(data.experiment.metric_selection)}\\\bottomrule
\end{tabular}
\vspace{1em}
\textbf{Models:} COREave, COREtrm, CORE + Qwen3-0.6B + LoRA\\
\status{Single-seed preliminary validation comparison. No matched full test result yet.}
\end{frame}

\begin{frame}{Current result}
\centering
\begin{tabular}{lrr}\toprule Model & Recall@20 & MRR@20\\\midrule
COREave & ${m.coreave.recall20.toFixed(4)} & ${m.coreave.mrr20.toFixed(4)}\\
COREtrm & ${m.coretrm.recall20.toFixed(4)} & ${m.coretrm.mrr20.toFixed(4)}\\
CORE + LLM & ${m.llm.recall20.toFixed(4)} & ${m.llm.mrr20.toFixed(4)}\\\bottomrule
\end{tabular}
\vspace{1.1em}
{\Huge\textcolor{accentdark}{\textbf{${rel.toFixed(2)}\%}}}\\
LLM vs CORE on MRR@20\\[0.8em]
\textbf{Current default configuration does not outperform CORE.}\\[0.8em]
\small Diagnostic: Recall@20 reached ${m.llm.later_recall20.toFixed(4)} at epoch ${m.llm.later_recall20_epoch}, while MRR@20 fell to ${m.llm.later_mrr20.toFixed(4)}.
\end{frame}

\begin{frame}{Current findings}
\begin{columns}[T]
\column{0.31\textwidth}\textcolor{accentdark}{\Large\textbf{Engineering}}\\Pure-ID embeddings pass through Qwen. The full training and validation pipeline works.
\column{0.31\textwidth}\textcolor{accentdark}{\Large\textbf{Quality}}\\The current model trails matched CORE baselines on validation MRR@20.
\column{0.31\textwidth}\textcolor{accentdark}{\Large\textbf{Attribution}}\\Pretraining and architecture remain confounded. The full random-backbone control is untested.
\end{columns}
\vfill\status{This experiment tests direct representation transfer. Explicit intent reasoning remains untested.}
\end{frame}

\begin{frame}{Representation mismatch}
\centering\Large\textbf{A pretrained language model receives a completely new alphabet}
\vspace{1em}
\begin{columns}
\column{0.34\textwidth}\centering\textbf{Language pretraining}\\words\\syntax\\semantic structure
\column{0.28\textwidth}\centering\status{representation mismatch}
\column{0.34\textwidth}\centering\textbf{Pure-ID input}\\arbitrary labels\\learned vectors\\behavioral signal only
\end{columns}
\vfill\textbf{Preliminary implication:} sequence ability may require behavioral structure or task specialization before it transfers.
\end{frame}

\begin{frame}{Compute consideration}
\begin{tabular}{lr}\toprule Model & Recorded seconds / epoch\\\midrule
COREave & ${m.coreave.epoch_time_sec.toFixed(2)}\\
COREtrm & ${m.coretrm.epoch_time_sec.toFixed(2)}\\
Qwen + LoRA & ${m.llm.epoch_time_sec.toFixed(2)}\\\bottomrule\end{tabular}
\vspace{1em}
{\Huge\textcolor{accentdark}{\textbf{${(m.llm.epoch_time_sec / m.coreave.epoch_time_sec).toFixed(0)}$\times$}}} slower than COREave in recorded training time.\\[1em]
\status{Recorded training cost must be considered alongside ranking quality.}
\end{frame}

\begin{frame}{Immediate plan}
\textbf{Now:} diagnose gate and branch behavior, run matched learning-rate searches, add a random-backbone control, and validate paired seeds.\\[1.2em]
\textbf{If a stable positive signal appears:} paired seeds, random-backbone control, locked final-test evaluation.\\[1.2em]
\textbf{If no stable gain appears:} shrink the LLM direction.\\[1.5em]
\status{Next-stage goal: identify where LLM reasoning adds measurable value before increasing model size.}
\end{frame}
\end{document}
`;
await fs.writeFile(texPath, tex, "utf8");

const candidatePath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const stamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
const validatedPath = path.join(validatedDir, `validated-${stamp}.pptx`);
const receiptPath = path.join(buildDir, `validation-${stamp}.json`);
const requirements = {
  explicitTotalSlideCount: 9,
  requiredNativeTableOwnerSlides: [4, 5],
  requiredNativeChartOwnerSlides: [8],
  materializeLiteralChartWorkbooks: true,
};
await finalizePresentation({
  ...requirements,
  workspaceDir: projectRoot,
  candidatePath,
  finalPath: validatedPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "4",
    "--require-native-table-slide", "5",
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  requiredNativeChartOwnerSlides: requirements.requiredNativeChartOwnerSlides,
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath,
});
await fs.copyFile(validatedPath, finalPptx);

const checked = await PresentationFile.importPptx(await FileBlob.load(finalPptx));
const checkedSlideCount = checked.toProto().slides.length;
for (let i = 0; i < checkedSlideCount; i += 1) {
  const slide = checked.slides.getItem(i);
  const png = await checked.export({ slide, format: "png", scale: 1.25 });
  await fs.writeFile(path.join(renderDir, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(renderDir, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const montage = await checked.export({ format: "png", montage: true, scale: 0.7 });
await fs.writeFile(path.join(renderDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));

console.log(JSON.stringify({
  pptx: finalPptx,
  tex: texPath,
  slides: checkedSlideCount,
  font,
  validationReceipt: receiptPath,
  renderDir,
}, null, 2));

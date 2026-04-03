const fs = require("fs");
const path = require("path");

const ExcelJS = require("exceljs");
const {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const REPORTS_DIR = path.resolve(__dirname);
const DOCX_PATH = path.join(REPORTS_DIR, "lab_meeting_layer0_sae_summary_20260319.docx");
const XLSX_PATH = path.join(REPORTS_DIR, "lab_meeting_layer0_sae_metrics_20260319.xlsx");
const SOURCE_MD_PATH = path.join(REPORTS_DIR, "lab_meeting_layer0_sae_source_20260319.md");
const FEATURE_1419_PATH = path.join(
  REPORTS_DIR,
  "..",
  "runs",
  "sae",
  "layer0_audit_bace_top100_20260319",
  "reports",
  "feature_cards",
  "bace_classification",
  "feature_1419.json",
);
const DOWNSTREAM_CHART_PATH = path.join(REPORTS_DIR, "layer0_downstream_choice_chart_20260319.png");
const CAUSAL_CHART_PATH = path.join(REPORTS_DIR, "layer0_causal_effect_chart_20260319.png");
const ACTIVATION_CHART_PATH = path.join(REPORTS_DIR, "bace1419_top_activation_chart_20260319.png");

const downstreamRows = [
  { task: "BBBP", baseline: 0.7190, sae1536: 0.6918, sae2048: 0.7041 },
  { task: "BACE", baseline: 0.7637, sae1536: 0.7620, sae2048: 0.7726 },
  { task: "ClinTox", baseline: 0.9993, sae1536: 0.9978, sae2048: 1.0000 },
];

const causalRows = [
  {
    task: "BBBP",
    group: "1237,1492,1402",
    mode: "zero",
    targetDelta: -0.00333,
    controlDelta: -0.00057,
    targetAuc: 0.69826,
    controlAuc: 0.70102,
    baselineAuc: 0.70159,
    note: "task-linked group가 random control보다 더 크게 성능을 흔듦",
  },
  {
    task: "BBBP",
    group: "1237,1492,1402",
    mode: "force_on",
    targetDelta: -0.00491,
    controlDelta: -0.00020,
    targetAuc: 0.69668,
    controlAuc: 0.70139,
    baselineAuc: 0.70159,
    note: "강하게 켜면 예측 점수가 움직이고 AUC가 감소함",
  },
  {
    task: "BACE",
    group: "327,1419,1785",
    mode: "zero",
    targetDelta: 0.00063,
    controlDelta: -0.00007,
    targetAuc: 0.72676,
    controlAuc: 0.72607,
    baselineAuc: 0.72613,
    note: "zero만으로는 큰 효과가 아님",
  },
  {
    task: "BACE",
    group: "327,1419,1785",
    mode: "force_on",
    targetDelta: -0.00424,
    controlDelta: 0.00017,
    targetAuc: 0.72189,
    controlAuc: 0.72631,
    baselineAuc: 0.72613,
    note: "force_on에서 task-specific effect가 분명함",
  },
];

const featureRows = [
  {
    task: "BACE",
    feature: 1419,
    coefMean: 1.4908,
    auc: 0.4756,
    dominantScaffold: "O=C1NC=NC1(c1ccccc1)c1ccccc1",
    mcsAtoms: 18,
    localization: "localizable",
    summary: "diaryl + carbonyl-containing N-rich heterocycle/amidine-like core",
    talk: "사람이 보는 scaffold/headgroup family와 가장 잘 맞음",
  },
  {
    task: "BACE",
    feature: 1785,
    coefMean: 1.4368,
    auc: 0.5735,
    dominantScaffold: "heteroaryl/bulky aromatic family",
    mcsAtoms: 9,
    localization: "CLS-global",
    summary: "aza-heteroaryl이 섞인 bulky aromatic scaffold family",
    talk: "구조 family는 있으나 위치성보다 전역 요약 신호에 가까움",
  },
  {
    task: "BBBP",
    feature: 1011,
    coefMean: -1.0199,
    auc: 0.5706,
    dominantScaffold: "tricyclic / aromatic ring families",
    mcsAtoms: 6,
    localization: "CLS-global",
    summary: "ring-rich / aromatic global factor",
    talk: "token-level motif보다 분자 타입 전체를 요약하는 축",
  },
  {
    task: "BBBP",
    feature: 1237,
    coefMean: 2.0021,
    auc: 0.5611,
    dominantScaffold: "mixed; phenyl and imide-like families",
    mcsAtoms: 3,
    localization: "CLS-global",
    summary: "다양한 분자군을 묶는 broad global factor",
    talk: "causal effect는 있지만 atom-level 이름 붙이기 어려움",
  },
  {
    task: "BBBP",
    feature: 1492,
    coefMean: 1.2254,
    auc: 0.5760,
    dominantScaffold: "mixed small cyclic / aromatic families",
    mcsAtoms: 0,
    localization: "CLS-global",
    summary: "small ring / hydrophobic-type global factor",
    talk: "전역 분자 특성이 CLS에 요약된 형태",
  },
];

const takeaways = [
  "Layer 0 결과는 완성본이 아니라 파일럿 결과다.",
  "2048 / l0=0.05는 성능이 아주 크게 무너지지 않은 출발점이었다.",
  "BACE에서는 사람이 이해할 수 있는 구조 계열 feature 하나가 보였다.",
  "BBBP는 특정 부분구조보다 분자 전체 성질을 요약한 신호가 더 강했다.",
  "즉 계속 볼 가치는 있지만, 아직 큰 주장으로 가기엔 이르다.",
];

const talkTrack = [
  "오늘 결과는 layer 0 전체 성공 보고라기보다 파일럿 결과로 보는 게 맞습니다.",
  "그래도 BACE에서는 사람이 이해할 수 있는 구조 계열 feature 하나가 나왔습니다.",
  "반면 BBBP는 특정 부분구조보다 분자 전체 요약 신호가 더 강했습니다.",
  "그래서 다음 단계는 중간 레이어와 추가 검증입니다.",
];

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, options = {}) {
  return new Paragraph({
    ...options,
    children: [new TextRun({ text })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text })],
  });
}

function tableCell(text, width, shaded = false) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: shaded ? { fill: "E5EEF7", type: "clear" } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [p(text)],
  });
}

function imageParagraph(imagePath, width, height, title) {
  if (!fs.existsSync(imagePath)) {
    return null;
  }
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(imagePath),
        transformation: { width, height, rotation: 0 },
        altText: { title, description: title, name: title },
      }),
    ],
  });
}

function feature1419ExampleRows(limitPerSplit = 6) {
  const payload = JSON.parse(fs.readFileSync(FEATURE_1419_PATH, "utf8"));
  const examples = [
    ...(payload.top_train_examples || []).slice(0, limitPerSplit),
    ...(payload.top_test_examples || []).slice(0, limitPerSplit),
  ];
  return examples.map((row) => ({
    split: row.split,
    rank: row.rank,
    activation: row.activation,
    label: row.label,
    smiles: row.smiles,
  }));
}

function num(v, digits = 4) {
  return Number(v).toFixed(digits);
}

function delta(v) {
  return `${v >= 0 ? "+" : ""}${num(v, 4)}`;
}

async function buildDocx() {
  const children = [];
  const baceExamples = feature1419ExampleRows(4);
  children.push(
    new Paragraph({
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "ChemBERTa SAE Layer 0 랩미팅 정리", bold: true, size: 32 })],
    }),
  );
  children.push(p("날짜: 2026-03-19"));
  children.push(p("대상 후보: layer 0 / JumpReLU / n_latents=2048 / base_l0=0.05"));
  children.push(p(`소스 메모: ${SOURCE_MD_PATH}`));

  children.push(p("한 줄 결론", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p(
      "Layer 0 결과는 완성된 결론이 아니라 파일럿 결과다. 다만 그 파일럿에서, BACE 쪽에서는 사람이 이해할 수 있는 feature 후보 하나가 보였다.",
    ),
  );

  children.push(p("초보자용으로 아주 쉽게 말하면", { heading: HeadingLevel.HEADING_1 }));
  [
    "feature는 분자 표현을 나눈 작은 조각이라고 생각하면 된다.",
    "local feature는 특정 부분구조에 반응하는 조각이다.",
    "CLS-global feature는 분자 전체 분위기나 타입에 반응하는 조각이다.",
    "이번 layer 0에서는 BACE는 전자에 가까운 예가 있었고, BBBP는 후자에 가까웠다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("핵심 Takeaway", { heading: HeadingLevel.HEADING_1 }));
  takeaways.forEach((line) => children.push(bullet(line)));

  children.push(p("왜 이 후보를 골랐는가", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p("2048 / 0.05가 성능을 아주 크게 망가뜨리지 않으면서 해석 실험을 시작해볼 수 있는 가장 무난한 출발점이었다."),
  );
  const downstreamChart = imageParagraph(DOWNSTREAM_CHART_PATH, 620, 345, "Layer 0 candidate choice chart");
  if (downstreamChart) {
    children.push(downstreamChart);
  }

  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1800, 1600, 1600, 1600, 1600, 1160],
      rows: [
        new TableRow({
          children: [
            tableCell("Task", 1800, true),
            tableCell("Baseline", 1600, true),
            tableCell("1536/0.05", 1600, true),
            tableCell("Delta", 1600, true),
            tableCell("2048/0.05", 1600, true),
            tableCell("Delta", 1160, true),
          ],
        }),
        ...downstreamRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.task, 1800),
                tableCell(num(row.baseline), 1600),
                tableCell(num(row.sae1536), 1600),
                tableCell(delta(row.sae1536 - row.baseline), 1600),
                tableCell(num(row.sae2048), 1600),
                tableCell(delta(row.sae2048 - row.baseline), 1160),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("이번에 실제로 한 작업", { heading: HeadingLevel.HEADING_1 }));
  [
    "Feature audit로 task-linked latent를 정리했다.",
    "zero / force_on causal intervention으로 random control보다 큰 효과가 나는지 확인했다.",
    "substructure grounding으로 공통 scaffold와 MCS를 확인했다.",
    "token localization으로 local feature인지 CLS-global feature인지 구분했다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("Causal Intervention 핵심 결과", { heading: HeadingLevel.HEADING_1 }));
  const causalChart = imageParagraph(CAUSAL_CHART_PATH, 620, 345, "Causal effect chart");
  if (causalChart) {
    children.push(causalChart);
  }
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1200, 1400, 1400, 1400, 1400, 2520],
      rows: [
        new TableRow({
          children: [
            tableCell("Task", 1200, true),
            tableCell("Mode", 1400, true),
            tableCell("Target ΔAUC", 1400, true),
            tableCell("Control ΔAUC", 1400, true),
            tableCell("Target Group", 1400, true),
            tableCell("한줄 해석", 2520, true),
          ],
        }),
        ...causalRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.task, 1200),
                tableCell(row.mode, 1400),
                tableCell(delta(row.targetDelta), 1400),
                tableCell(delta(row.controlDelta), 1400),
                tableCell(row.group, 1400),
                tableCell(row.note, 2520),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("해석적으로 가장 중요한 포인트", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1000, 900, 1100, 2700, 1100, 2550],
      rows: [
        new TableRow({
          children: [
            tableCell("Task", 1000, true),
            tableCell("Feature", 900, true),
            tableCell("Type", 1100, true),
            tableCell("구조 요약", 2700, true),
            tableCell("MCS atoms", 1100, true),
            tableCell("발표 때 말할 포인트", 2550, true),
          ],
        }),
        ...featureRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.task, 1000),
                tableCell(String(row.feature), 900),
                tableCell(row.localization, 1100),
                tableCell(row.summary, 2700),
                tableCell(String(row.mcsAtoms), 1100),
                tableCell(row.talk, 2550),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("CLS-global feature를 쉬운 말로 설명하면", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p("local feature는 특정 원자나 특정 substructure 위치에서 켜지는 축이고, CLS-global feature는 분자 전체가 어떤 타입인지 요약한 축이다."),
  );
  children.push(
    p("이번 layer 0에서는 BBBP의 핵심 feature들이 후자에 더 가까웠고, BACE 1419만 비교적 localizable한 쪽으로 보였다."),
  );

  children.push(p("BACE 1419를 실제 SMILES로 보면", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p("아래 표는 feature 1419가 특히 강하게 켜진 대표 분자들이다. activation 숫자가 클수록 이 feature가 더 강하게 반응한 것이다."),
  );
  const activationChart = imageParagraph(ACTIVATION_CHART_PATH, 620, 345, "BACE 1419 top activation chart");
  if (activationChart) {
    children.push(activationChart);
  }
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [900, 700, 1100, 900, 5760],
      rows: [
        new TableRow({
          children: [
            tableCell("Split", 900, true),
            tableCell("Rank", 700, true),
            tableCell("Activation", 1100, true),
            tableCell("Label", 900, true),
            tableCell("SMILES", 5760, true),
          ],
        }),
        ...baceExamples.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.split, 900),
                tableCell(String(row.rank), 700),
                tableCell(num(row.activation), 1100),
                tableCell(String(row.label), 900),
                tableCell(row.smiles, 5760),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("다음 단계", { heading: HeadingLevel.HEADING_1 }));
  [
    "global feature와 molecular descriptor(logP, MW, aromatic ring count 등)의 상관 확인",
    "layer 1~2에 같은 파이프라인 적용해서 더 local한 feature가 나오는지 비교",
    "counterfactual molecule pair로 motif 유무에 따른 latent/logit 변화를 확인",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("발표 때 바로 읽을 수 있는 문장", { heading: HeadingLevel.HEADING_1 }));
  talkTrack.forEach((line) => children.push(bullet(line)));

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 28, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 0 },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [
            {
              level: 0,
              format: LevelFormat.BULLET,
              text: "•",
              alignment: AlignmentType.LEFT,
              style: { paragraph: { indent: { left: 720, hanging: 360 } } },
            },
          ],
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 12240, height: 15840 },
            margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
          },
        },
        children,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(DOCX_PATH, buffer);
}

async function buildXlsx() {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "Codex";
  workbook.created = new Date("2026-03-19T00:00:00Z");
  const baceExamples = feature1419ExampleRows(20);

  const downstream = workbook.addWorksheet("Downstream");
  downstream.columns = [
    { header: "Task", key: "task", width: 14 },
    { header: "Baseline", key: "baseline", width: 12 },
    { header: "1536_0.05", key: "sae1536", width: 12 },
    { header: "Delta_1536", key: "delta1536", width: 12 },
    { header: "2048_0.05", key: "sae2048", width: 12 },
    { header: "Delta_2048", key: "delta2048", width: 12 },
    { header: "Chosen", key: "chosen", width: 16 },
  ];
  downstreamRows.forEach((row, idx) => {
    const r = idx + 2;
    downstream.addRow({
      task: row.task,
      baseline: row.baseline,
      sae1536: row.sae1536,
      delta1536: { formula: `C${r}-B${r}` },
      sae2048: row.sae2048,
      delta2048: { formula: `E${r}-B${r}` },
      chosen: "2048 / 0.05",
    });
  });

  const causal = workbook.addWorksheet("Causal");
  causal.columns = [
    { header: "Task", key: "task", width: 12 },
    { header: "Group", key: "group", width: 20 },
    { header: "Mode", key: "mode", width: 12 },
    { header: "Baseline_AUC", key: "baselineAuc", width: 14 },
    { header: "Target_AUC", key: "targetAuc", width: 14 },
    { header: "Target_Delta", key: "targetDelta", width: 14 },
    { header: "Control_AUC", key: "controlAuc", width: 14 },
    { header: "Control_Delta", key: "controlDelta", width: 14 },
    { header: "Effect_Gap", key: "effectGap", width: 14 },
    { header: "Note", key: "note", width: 40 },
  ];
  causalRows.forEach((row, idx) => {
    const r = idx + 2;
    causal.addRow({
      task: row.task,
      group: row.group,
      mode: row.mode,
      baselineAuc: row.baselineAuc,
      targetAuc: row.targetAuc,
      targetDelta: row.targetDelta,
      controlAuc: row.controlAuc,
      controlDelta: row.controlDelta,
      effectGap: { formula: `F${r}-H${r}` },
      note: row.note,
    });
  });

  const features = workbook.addWorksheet("Features");
  features.columns = [
    { header: "Task", key: "task", width: 12 },
    { header: "Feature", key: "feature", width: 10 },
    { header: "CoefMean", key: "coefMean", width: 12 },
    { header: "SingleFeatureAUC", key: "auc", width: 16 },
    { header: "Localization", key: "localization", width: 14 },
    { header: "DominantScaffold", key: "dominantScaffold", width: 38 },
    { header: "MCSAtoms", key: "mcsAtoms", width: 10 },
    { header: "Summary", key: "summary", width: 42 },
    { header: "TalkTrack", key: "talk", width: 42 },
  ];
  featureRows.forEach((row) => features.addRow(row));

  const examples = workbook.addWorksheet("BACE1419_Examples");
  examples.columns = [
    { header: "Split", key: "split", width: 12 },
    { header: "Rank", key: "rank", width: 10 },
    { header: "Activation", key: "activation", width: 14 },
    { header: "Label", key: "label", width: 10 },
    { header: "SMILES", key: "smiles", width: 96 },
  ];
  baceExamples.forEach((row) => examples.addRow(row));

  const sheets = [downstream, causal, features, examples];
  sheets.forEach((sheet) => {
    sheet.getRow(1).font = { bold: true, name: "Arial" };
    sheet.getRow(1).fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "DDEBF7" },
    };
    sheet.eachRow((row, rowNumber) => {
      row.eachCell((cell) => {
        cell.font = { name: "Arial", size: 10 };
        cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
        if (rowNumber > 1 && typeof cell.value === "number") {
          cell.numFmt = "0.0000";
        }
      });
    });
    sheet.views = [{ state: "frozen", ySplit: 1 }];
  });

  await workbook.xlsx.writeFile(XLSX_PATH);
}

async function main() {
  if (!fs.existsSync(SOURCE_MD_PATH)) {
    throw new Error(`source markdown not found: ${SOURCE_MD_PATH}`);
  }
  await buildDocx();
  await buildXlsx();
  console.log(DOCX_PATH);
  console.log(XLSX_PATH);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

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
const DOCX_PATH = path.join(REPORTS_DIR, "bace_decision_memo_20260319.docx");
const XLSX_PATH = path.join(REPORTS_DIR, "bace_decision_metrics_20260319.xlsx");
const SOURCE_MD_PATH = path.join(REPORTS_DIR, "bace_decision_memo_source_20260319.md");
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
const CAUSAL_CHART_PATH = path.join(REPORTS_DIR, "layer0_causal_effect_chart_20260319.png");
const ACTIVATION_CHART_PATH = path.join(REPORTS_DIR, "bace1419_top_activation_chart_20260319.png");

const evidenceRows = [
  {
    evidence: "Task linkage",
    signal: "BACE feature 1419",
    result: "coef_mean = +1.4908, seed-stable",
    meaning: "BACE 예측에 꾸준히 쓰이는 feature라는 뜻",
  },
  {
    evidence: "Causal effect",
    signal: "force_on target vs control",
    result: "target ΔAUC = -0.00424, control ΔAUC = +0.00017",
    meaning: "아무 feature가 아니라 BACE와 관련된 feature일 가능성이 큼",
  },
  {
    evidence: "Structure grounding",
    signal: "top scaffold / MCS",
    result: "dominant scaffold 5/10, MCS = 18 atoms",
    meaning: "비슷한 구조 계열을 실제로 잡고 있음",
  },
  {
    evidence: "Token locality",
    signal: "non-CLS vs CLS",
    result: "non-CLS max mean = 2.8492, CLS mean = 0.4988",
    meaning: "분자 전체 요약만이 아니라 특정 구조 위치 신호도 보임",
  },
];

const decisionRows = [
  {
    question: "지금 당장 계속할 가치가 있나?",
    answer: "조심스러운 YES",
    reason: "feature 1419가 사람이 이해할 수 있는 구조 계열과 연결되는 첫 강한 사례이기 때문",
  },
  {
    question: "넓게 계속해야 하나?",
    answer: "NO",
    reason: "전 레이어 sweep보다 BACE 중심 좁은 2차 검증이 우선",
  },
  {
    question: "중단 기준은 무엇인가?",
    answer: "1419만 예외일 때",
    reason: "추가 localizable BACE feature가 안 나오면 general claim이 약함",
  },
];

const nextStepRows = [
  {
    step: "1",
    action: "feature 1419 상위 샘플 확대 검증",
    success: "top 10이 아니라 더 넓은 구간에서도 scaffold enrichment 유지",
  },
  {
    step: "2",
    action: "추가 BACE localizable feature 탐색",
    success: "2개 이상 추가 latent가 task-linked + causal + structural 조건 만족",
  },
  {
    step: "3",
    action: "descriptor / fingerprint 상관 분석",
    success: "activation이 화학 descriptor와 해석 가능한 축으로 연결",
  },
];

function loadFeature1419Payload() {
  return JSON.parse(fs.readFileSync(FEATURE_1419_PATH, "utf8"));
}

function feature1419SummaryRows() {
  const payload = loadFeature1419Payload();
  const summary = payload.summary;
  const topTrain = payload.top_train_examples?.[0];
  const topTest = payload.top_test_examples?.[0];
  return [
    {
      metric: "Activation frequency",
      value: `${(summary.activation_frequency * 100).toFixed(2)}%`,
      meaning: "전체 샘플 중 이 feature가 켜지는 비율",
    },
    {
      metric: "Positive mean activation",
      value: summary.positive_mean_activation.toFixed(4),
      meaning: "positive 샘플에서 평균적으로 얼마나 켜지는지",
    },
    {
      metric: "Negative mean activation",
      value: summary.negative_mean_activation.toFixed(4),
      meaning: "negative 샘플에서 평균적으로 얼마나 켜지는지",
    },
    {
      metric: "Top train activation",
      value: topTrain ? topTrain.activation.toFixed(4) : "-",
      meaning: "가장 세게 켜진 train 예시의 activation",
    },
    {
      metric: "Top test activation",
      value: topTest ? topTest.activation.toFixed(4) : "-",
      meaning: "가장 세게 켜진 test 예시의 activation",
    },
  ];
}

function feature1419ExampleRows(limitPerSplit = 4) {
  const payload = loadFeature1419Payload();
  const examples = [
    ...(payload.top_train_examples || []).slice(0, limitPerSplit),
    ...(payload.top_test_examples || []).slice(0, limitPerSplit),
  ];
  return examples.map((row) => ({
    split: row.split,
    rank: row.rank,
    activation: Number(row.activation).toFixed(4),
    label: String(row.label),
    smiles: row.smiles,
  }));
}

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

async function buildDocx() {
  const children = [];
  const featureSummary = feature1419SummaryRows();
  const featureExamples = feature1419ExampleRows(4);
  children.push(
    new Paragraph({
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "BACE 중심 의사결정 메모", bold: true, size: 32 })],
    }),
  );
  children.push(p("날짜: 2026-03-19"));
  children.push(p("핵심 질문: 이 SAE 해석 실험을 계속할 가치가 있는가?"));

  children.push(p("내 결론", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p("예. 하지만 layer 0 전체가 이미 잘 해석된다고 말하기는 이르다. 지금은 BACE feature 1419라는 첫 번째 성공 사례가 보였다고 이해하는 것이 맞다."),
  );

  children.push(p("왜 BACE가 핵심인가", { heading: HeadingLevel.HEADING_1 }));
  [
    "현재 결과 중 사람이 가장 설명하기 쉬운 신호가 BACE에서 나왔다.",
    "BBBP는 변화는 보였지만, 대부분 분자 전체 분위기를 요약한 신호에 가까웠다.",
    "반면 BACE 1419는 예측과 연결되고, 구조 계열과도 연결되고, 위치성도 어느 정도 보였다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("쉬운 말로 하면", { heading: HeadingLevel.HEADING_1 }));
  [
    "이 실험의 질문은 'SAE 노드 하나가 사람이 알아볼 수 있는 개념을 담고 있나?'이다.",
    "지금까지는 BACE feature 1419가 그 질문에 가장 가까운 예시다.",
    "즉 '될 수도 있겠다'는 첫 증거는 봤지만, 아직 크게 일반화할 단계는 아니다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("핵심 증거 체인", { heading: HeadingLevel.HEADING_1 }));
  const causalChart = imageParagraph(CAUSAL_CHART_PATH, 620, 345, "Causal effect chart");
  if (causalChart) {
    children.push(causalChart);
  }
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1800, 1800, 2200, 3560],
      rows: [
        new TableRow({
          children: [
            tableCell("증거 종류", 1800, true),
            tableCell("무엇을 봤나", 1800, true),
            tableCell("결과", 2200, true),
            tableCell("왜 중요한가", 3560, true),
          ],
        }),
        ...evidenceRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.evidence, 1800),
                tableCell(row.signal, 1800),
                tableCell(row.result, 2200),
                tableCell(row.meaning, 3560),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("이 feature가 의미하는 것", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p(
      "현재 해석으로 feature 1419는 carbonyl이 있고, nitrogen이 많은 core가 있고, 양쪽에 방향족 고리가 붙은 구조 계열을 잡는 feature로 보인다.",
    ),
  );
  children.push(
    p(
      "쉽게 말하면, 사람이 BACE 관련 분자를 볼 때 '비슷한 계열이다'라고 느끼는 구조 특징을 latent 하나가 어느 정도 따로 담고 있다는 뜻이다.",
    ),
  );

  children.push(p("실제 SMILES와 activation을 같이 보면", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    p("activation 숫자는 이 feature가 얼마나 세게 켜졌는지를 뜻한다. 숫자가 클수록 그 분자에서 이 feature가 더 강하게 활성화된 것이다."),
  );
  const activationChart = imageParagraph(ACTIVATION_CHART_PATH, 620, 345, "BACE 1419 activation chart");
  if (activationChart) {
    children.push(activationChart);
  }
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2400, 1800, 5160],
      rows: [
        new TableRow({
          children: [
            tableCell("지표", 2400, true),
            tableCell("값", 1800, true),
            tableCell("쉽게 말한 뜻", 5160, true),
          ],
        }),
        ...featureSummary.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.metric, 2400),
                tableCell(row.value, 1800),
                tableCell(row.meaning, 5160),
              ],
            }),
        ),
      ],
    }),
  );
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
        ...featureExamples.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.split, 900),
                tableCell(String(row.rank), 700),
                tableCell(row.activation, 1100),
                tableCell(row.label, 900),
                tableCell(row.smiles, 5760),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("왜 사람이 보는 관점과 연결된다고 말할 수 있나", { heading: HeadingLevel.HEADING_1 }));
  [
    "사람도 BACE 쪽 분자를 볼 때 어떤 core가 있고 어떤 고리가 붙어 있는지를 본다.",
    "feature 1419는 carbonyl, N-rich core, diaryl family를 함께 잡는다.",
    "즉 모델이 완전히 엉뚱한 패턴을 잡은 것이 아니라, 사람이 보는 구조 관점과 겹치는 축을 일부 담고 있다고 해석할 수 있다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("아직 조심해야 할 점", { heading: HeadingLevel.HEADING_1 }));
  [
    "이 feature 하나가 BACE 작동 원리를 전부 설명한다고 말할 수는 없다.",
    "layer 0 전체가 다 이렇게 해석 가능하다고도 아직 말할 수 없다.",
    "현재 수준은 좋은 첫 증거이지, 최종 결론은 아니다.",
  ].forEach((line) => children.push(bullet(line)));

  children.push(p("그래서 계속해야 하는가", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2600, 1400, 5360],
      rows: [
        new TableRow({
          children: [
            tableCell("질문", 2600, true),
            tableCell("판단", 1400, true),
            tableCell("이유", 5360, true),
          ],
        }),
        ...decisionRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.question, 2600),
                tableCell(row.answer, 1400),
                tableCell(row.reason, 5360),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("바로 다음 단계", { heading: HeadingLevel.HEADING_1 }));
  children.push(
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [800, 3200, 5360],
      rows: [
        new TableRow({
          children: [
            tableCell("순서", 800, true),
            tableCell("해야 할 일", 3200, true),
            tableCell("성공 기준", 5360, true),
          ],
        }),
        ...nextStepRows.map(
          (row) =>
            new TableRow({
              children: [
                tableCell(row.step, 800),
                tableCell(row.action, 3200),
                tableCell(row.success, 5360),
              ],
            }),
        ),
      ],
    }),
  );

  children.push(p("발표 때 바로 쓸 문장", { heading: HeadingLevel.HEADING_1 }));
  [
    "오늘 결과는 layer 0 전체 성공 보고가 아니라, BACE에서 의미 있는 첫 사례를 본 정도로 이해하면 됩니다.",
    "그 사례가 바로 BACE feature 1419이고, 이 feature는 실제 구조 계열과 연결됩니다.",
    "그래서 실험을 접을 단계는 아니지만, 아직 과하게 일반화하면 안 되고 BACE 중심으로 더 확인해야 합니다.",
  ].forEach((line) => children.push(bullet(line)));

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
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
  const featureSummary = feature1419SummaryRows();
  const featureExamples = feature1419ExampleRows(25);

  const evidence = workbook.addWorksheet("BACE_Evidence");
  evidence.columns = [
    { header: "Evidence", key: "evidence", width: 18 },
    { header: "Signal", key: "signal", width: 22 },
    { header: "Result", key: "result", width: 28 },
    { header: "Meaning", key: "meaning", width: 44 },
  ];
  evidenceRows.forEach((row) => evidence.addRow(row));

  const decision = workbook.addWorksheet("Decision");
  decision.columns = [
    { header: "Question", key: "question", width: 28 },
    { header: "Answer", key: "answer", width: 14 },
    { header: "Reason", key: "reason", width: 52 },
  ];
  decisionRows.forEach((row) => decision.addRow(row));

  const next = workbook.addWorksheet("Next_Steps");
  next.columns = [
    { header: "Step", key: "step", width: 10 },
    { header: "Action", key: "action", width: 34 },
    { header: "Success", key: "success", width: 58 },
  ];
  nextStepRows.forEach((row) => next.addRow(row));

  const featureStats = workbook.addWorksheet("Feature1419_Stats");
  featureStats.columns = [
    { header: "Metric", key: "metric", width: 24 },
    { header: "Value", key: "value", width: 16 },
    { header: "Meaning", key: "meaning", width: 54 },
  ];
  featureSummary.forEach((row) => featureStats.addRow(row));

  const examples = workbook.addWorksheet("Feature1419_Examples");
  examples.columns = [
    { header: "Split", key: "split", width: 12 },
    { header: "Rank", key: "rank", width: 10 },
    { header: "Activation", key: "activation", width: 14 },
    { header: "Label", key: "label", width: 10 },
    { header: "SMILES", key: "smiles", width: 96 },
  ];
  featureExamples.forEach((row) => examples.addRow(row));

  [evidence, decision, next, featureStats, examples].forEach((sheet) => {
    sheet.getRow(1).font = { bold: true, name: "Arial" };
    sheet.getRow(1).fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "DDEBF7" },
    };
    sheet.eachRow((row) => {
      row.eachCell((cell) => {
        cell.font = { name: "Arial", size: 10 };
        cell.alignment = { vertical: "middle", horizontal: "left", wrapText: true };
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

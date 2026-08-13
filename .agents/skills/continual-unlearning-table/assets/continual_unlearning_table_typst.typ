#let accent = rgb("#2f5578")
#let header-fill = rgb("#eef3f7")
#let rule-gray = rgb("#7f8992")
#let note-gray = rgb("#68727b")

#set page(
  width: 8.5in,
  height: auto,
  margin: (x: 0.38in, y: 0.30in),
  fill: white,
)
#set text(font: "Libertinus Serif", size: 8.6pt, fill: rgb("#17191b"))
#set par(leading: 0.58em)

#let best(value) = strong(value)
#let metric(name, arrow) = text(weight: "semibold")[
  #name#h(2pt)#text(fill: accent, weight: "bold")[#arrow]
]

#block(width: 100%)[
  #text(weight: "bold", fill: accent)[Table 1.]
  #h(2pt)
  #text(weight: "semibold")[Continual unlearning performance on TOFU
  #raw("forget05") for the selected NPO and RoAdBlock runs.]
  Each task is evaluated immediately after its corresponding forget request.
  Lower FR is better; higher FQ and MU are better.
]

#v(6pt)

#table(
  columns: (1.32fr,) + (1fr,) * 15,
  inset: (x: 3.2pt, y: 3.0pt),
  align: (left,) + (center,) * 15,
  stroke: none,

  table.hline(y: 0, stroke: 0.9pt + accent),

  table.cell(rowspan: 2, fill: header-fill, align: left + horizon)[*Method*],
  table.cell(colspan: 3, fill: header-fill, align: center)[*Task 1*],
  table.cell(colspan: 3, fill: header-fill, align: center)[*Task 2*],
  table.cell(colspan: 3, fill: header-fill, align: center)[*Task 3*],
  table.cell(colspan: 3, fill: header-fill, align: center)[*Task 4*],
  table.cell(colspan: 3, fill: header-fill, align: center)[*Task 5*],

  table.hline(y: 1, start: 1, end: 4, stroke: 0.42pt + rule-gray),
  table.hline(y: 1, start: 4, end: 7, stroke: 0.42pt + rule-gray),
  table.hline(y: 1, start: 7, end: 10, stroke: 0.42pt + rule-gray),
  table.hline(y: 1, start: 10, end: 13, stroke: 0.42pt + rule-gray),
  table.hline(y: 1, start: 13, end: 16, stroke: 0.42pt + rule-gray),

  table.cell(fill: header-fill)[#metric("FR", "↓")],
  table.cell(fill: header-fill)[#metric("FQ", "↑")],
  table.cell(fill: header-fill)[#metric("MU", "↑")],
  table.cell(fill: header-fill)[#metric("FR", "↓")],
  table.cell(fill: header-fill)[#metric("FQ", "↑")],
  table.cell(fill: header-fill)[#metric("MU", "↑")],
  table.cell(fill: header-fill)[#metric("FR", "↓")],
  table.cell(fill: header-fill)[#metric("FQ", "↑")],
  table.cell(fill: header-fill)[#metric("MU", "↑")],
  table.cell(fill: header-fill)[#metric("FR", "↓")],
  table.cell(fill: header-fill)[#metric("FQ", "↑")],
  table.cell(fill: header-fill)[#metric("MU", "↑")],
  table.cell(fill: header-fill)[#metric("FR", "↓")],
  table.cell(fill: header-fill)[#metric("FQ", "↑")],
  table.cell(fill: header-fill)[#metric("MU", "↑")],

  table.hline(y: 2, stroke: 0.65pt + rule-gray),

  [NPO],
  [.389], [#best[.416]], [.595],
  [.394], [#best[.096]], [.588],
  [.278], [.052], [.583],
  [.391], [.166], [.577],
  [.389], [.045], [.571],

  [RoAdBlock],
  [#best[.361]], [.016], [#best[.600]],
  [#best[.315]], [.071], [#best[.600]],
  [#best[.198]], [#best[.550]], [#best[.600]],
  [#best[.258]], [#best[.789]], [#best[.600]],
  [#best[.217]], [#best[.550]], [#best[.600]],

  table.hline(y: 4, stroke: 0.9pt + accent),
)

#v(4pt)

#align(center)[
  #text(size: 7.35pt, fill: note-gray, style: "italic")[
    Notes.
    #text(style: "normal")[FR: Forget QA-ROUGE; FQ: Forget Quality
    (KS $p$-value); MU: model utility. Bold denotes the better value within
    each task and metric.]
  ]
]

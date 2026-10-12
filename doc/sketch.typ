// Geometry sketch: origin, UMA-16 array, one source.
// Compile: typst compile geometry.typ geometry.svg
#import "@preview/cetz:0.5.2": canvas, draw

#set page(width: auto, height: auto, margin: 2mm, fill: none)
#set text(size: 9pt)

#let accent = rgb("#c0392b")
#let muted = luma(55%)

#canvas(length: 1.6cm, {
  import draw: *
  set-style(
    stroke: 0.6pt,
    mark: (scale: 0.55, fill: black),
  )

  // 1 unit = 10 cm; pitch 42 mm
  let pitch = 0.42
  let grid = (-1.5, -0.5, 0.5, 1.5).map(v => v * pitch)
  let mic = (grid.at(0), grid.at(3), 0)   // highlighted mic m_i
  let src = (0.5, 1.1, 2.2)               // source r_k

  ortho(x: 20deg, y: -55deg, sorted: false, {
    // board and mics
    on-xy({
      rect((-0.85, -0.85), (0.85, 0.85),
        stroke: (paint: muted, thickness: 0.4pt), fill: luma(96%))
      for x in grid { for y in grid {
        circle((x, y), radius: 0.025, fill: muted, stroke: none)
      }}
    })

    // axes
    for (p, l) in (((1.2, 0, 0), $x$), ((0, 1.1, 0), $y$), ((0, 0, 2.6), $z$)) {
      line((0, 0, 0), p, stroke: (paint: muted, thickness: 0.4pt),
        mark: (end: ">", fill: muted, stroke: muted))
      content(p, text(fill: muted, l), anchor: "south-west", padding: 1pt)
    }

    // vectors
    line((0, 0, 0), mic, mark: (end: ">"))
    line((0, 0, 0), src, stroke: accent, mark: (end: ">", fill: accent))
    line(mic, src, stroke: (dash: (2pt, 1.5pt)), mark: (end: ">"))

    // named points, drawn as true circles after the projection
    anchor("O", (0, 0, 0))
    anchor("mic", mic)
    anchor("src", src)

    // labels
    content((0, 0, 0), $O$, anchor: "north-west", padding: 2pt)
    content(mic, $bold(m)_i$, anchor: "east", padding: 3pt)
    content(src, text(fill: accent)[source $k$], anchor: "south", padding: 4pt)
    content((src.map(v => v * 0.55)), text(fill: accent, $bold(r)_k$),
      anchor: "north", padding: 3pt)
    content((mic.zip(src).map(((a, b)) => a + 0.5 * (b - a))), $bold(r)_(i k)$,
      anchor: "south", padding: 3pt)
  })

  circle("mic", radius: 0.035, fill: black, stroke: none)
  circle("src", radius: 0.055, fill: accent, stroke: none)
  circle("O", radius: 0.03, fill: white)
})



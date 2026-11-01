# Logo

- `logo.svg`: the logo (4×4 microphone array as a heat map in a camera viewfinder).
- `logo-small.svg`: simplified version for 32 px and below (favicon, small icons).
- `logo-wordmark.svg`: logo with the name, e.g. for a README banner or slides.
- `logo-{512,256,128,64}.png` from `logo.svg`, `logo-{32,16}.png` from `logo-small.svg`, rendered
  with resvg. `logo-256.png` is the GUI window icon (`include_bytes!` in `crates/app/src/main.rs`).

Colors match the GUI theme (`crates/app/src/theme.rs`): background #181b1f, amber #f5a524,
frame #e6e8eb, inactive mics #4a5058.

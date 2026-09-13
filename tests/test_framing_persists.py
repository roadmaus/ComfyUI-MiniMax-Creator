"""A framing and a plate's panels survive the blob.

    python3 tests/test_framing_persists.py

`serializeAssets` is what every commit writes into the node's widget, and it is
"stripped to what compile.py reads". It stripped two things compile.py reads:
the framing `picture.js` sets, so a window drawn in the editor rendered whole
and was forgotten on reload, and a plate's panels, so a cut-out came back as
the flat composite with its panels' handles gone from the piece. Found while
handing a frame off a clip its window (#81).

Skips itself if node is not installed.
"""

import layout
from harness import check, passed

layout.skip_without_node()
passed("a framing and a plate's panels come back out of the blob")

CHECK = """
const S = await import("./web/creator/state.js");
const crop = { x: 0.1, y: 0.2, w: 0.5, h: 0.6, turn: 90, mirror: "h" };
const panels = [
  { handle: "img-2", filename: "b.png", cut: true, points: [{ x: 0.2, y: 0.3, include: true }] },
  { handle: "img-3", filename: "c.png", rect: [0, 0, 0.5, 0.5], crop: { x: 0, y: 0, w: 0.5, h: 1 } },
];
const piece = S.parseTimeline(JSON.stringify({
  version: 2, prompt: "x",
  assets: [{ handle: "ref-1", kind: "video", role: "reference", filename: "clip.mp4", crop }],
  segments: [
    { prompt: "a", duration_s: 5, assets: [
      { handle: "img-1", kind: "image", role: "reference", filename: "sheet.png", panels },
      { handle: "img-4", kind: "image", role: "first_frame", filename: "d.png", crop },
      { handle: "img-5", kind: "image", role: "reference", filename: "e.png" },
    ] },
    { prompt: "b", duration_s: 5 }] }));
const once = JSON.parse(S.serializeTimeline(piece));
const twice = JSON.parse(S.serializeTimeline(S.parseTimeline(JSON.stringify(once))));
console.log(JSON.stringify({ once, same: JSON.stringify(once) === JSON.stringify(twice) }));
"""

with layout.pack(skip=["atlas"]) as target:
    got = layout.in_pack(CHECK, target)

blob = got["once"]
card = {a["handle"]: a for a in blob["segments"][0]["assets"]}
check("a pool clip keeps its framing", blob["assets"][0].get("crop"),
      {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.6, "turn": 90, "mirror": "h"})
check("a keyframe keeps its framing", card["img-4"].get("crop"),
      {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.6, "turn": 90, "mirror": "h"})
check("a plate keeps its panels, each with what it carries", card["img-1"].get("panels"), [
    {"handle": "img-2", "filename": "b.png", "cut": True, "points": [{"x": 0.2, "y": 0.3, "include": True}]},
    {"handle": "img-3", "filename": "c.png", "rect": [0, 0, 0.5, 0.5], "crop": {"x": 0, "y": 0, "w": 0.5, "h": 1}},
])
check("a picture used whole grows no keys", sorted(card["img-5"]), ["filename", "handle", "kind", "role"])
check("and the blob is stable across a second round trip", got["same"], True)

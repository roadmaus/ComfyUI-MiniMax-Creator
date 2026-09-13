"""`stylelib.styleAttributes` cuts a descriptor into the chips a restyle sends.

    python3 tests/test_restyle_attributes.py

The style-transfer file wants three to five attributes a painter would copy,
never the scene. The cut is by the descriptor's own joints, and this pins the
shape on real atlas descriptors so a change to the regexes is a decision made
against the catalogue rather than a surprise on a card. Skips without node.
"""

import layout

layout.skip_without_node()

from harness import FAILURES, check, passed

passed("a descriptor cuts into the attributes a restyle sends")

SCRIPT = """
const lib = await import("./web/creator/presets/stylelib.js");
const cases = JSON.parse(process.argv[1]);
console.log(JSON.stringify(cases.map((text) => ({
  attributes: lib.styleAttributes(text),
  marked: lib.styleAttributes(text).map((a) => Boolean(lib.attributeWarning(a))),
}))));
"""

CASES = [
    "Claymation with visible fingerprint texture and gently stuttering stop-motion movement",
    "LEGO brickfilm stop motion with bright plastic sheen under warm set lighting",
    "2D cutout-paper stop-motion animation, hand-cut colored-paper figures pinned at brass joints against a layered paper backdrop",
    "Black-and-white 1940s film noir on grainy 35mm with hard chiaroscuro shadows and faint gate flicker.",
    "Claymation, Aardman-style stop motion with visible fingerprint texture and tool marks pressed into every clay surface, inside a cluttered attic workshop",
]

with layout.pack(skip=["atlas"]) as tree:
    out = layout.in_pack(SCRIPT, tree, CASES)

check("clay: three attributes, the scene gone", out[0]["attributes"],
      ["claymation", "visible fingerprint texture", "gently stuttering stop-motion movement"])
check("lego: the light stays, and the franchise is marked",
      (out[1]["attributes"], out[1]["marked"]),
      (["lego brickfilm stop motion", "bright plastic sheen under warm set lighting"],
       [True, False]))
check("paper: the backdrop is scene", "against" in " ".join(out[2]["attributes"]), False)
check("noir: hyphens hold, the stock stays, the full stop goes, nothing is marked",
      (out[3]["attributes"], any(out[3]["marked"])),
      (["black-and-white 1940s film noir on grainy 35mm", "hard chiaroscuro shadows",
        "faint gate flicker"], False))
check("aardman: the studio is marked, the workshop is gone",
      (out[4]["marked"][1], "workshop" in " ".join(out[4]["attributes"])), (True, False))
check("never more than five", all(len(row["attributes"]) <= 5 for row in out), True)

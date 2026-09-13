"""`state.js`'s guide-LoRA block still agrees with `guidelora.py`.

    python3 tests/test_guidelora_mirror.py

The pill clamps a block on the way in and the node reads it again at queue
time; both are safe only while the two sides agree on the strength's stops,
the checkpoints, the default checkpoint, and what an off block writes.
`guidelora.py` is authoritative. A block switched on with no file is the one
case the two sides answer differently on purpose: the pill keeps it (the user
is about to pick one), the node refuses it.

Skips itself if node is not installed.
"""

import layout

layout.skip_without_node()

MIRROR = layout.js("state.js")

pkg = layout.load("canvas", "h3_declare", "guidelora")
gl = pkg.guidelora

from harness import FAILURES, check

SCRIPT = """
const s = await import(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
const out = {
  strength: s.GUIDE_LORA_STRENGTH, checkpoint: s.GUIDE_LORA_DEFAULT_CHECKPOINT,
  parsed: {}, serialized: {},
  piece: JSON.parse(s.serializeState(s.parseState(JSON.stringify({ version: 2, guide_lora: cases.full })))).guide_lora,
  piece_off: "guide_lora" in JSON.parse(s.serializeState(s.parseState(JSON.stringify({ version: 2 })))),
  timeline: JSON.parse(s.serializeTimeline(s.parseTimeline(JSON.stringify({ guide_lora: cases.full })))).guide_lora,
  timeline_off: "guide_lora" in JSON.parse(s.serializeTimeline(s.parseTimeline("{}"))),
  captions: Object.fromEntries(JSON.parse(process.argv[3]).map((name) => [name, s.guideLoraCaption(name)])),
  style: s.guideLoraStyle(),
  roles: s.guideLoraRoles(),
  role: Object.fromEntries(JSON.parse(process.argv[3]).map((name) => [name, s.guideLoraRole(name)?.key ?? null])),
  caption: s.restyleCaption(["Claymation", " visible fingerprint texture", ""]),
  restyle: s.isRestyle(s.parseGuideLora(cases.full)),
  sharpen: s.isRestyle(s.parseGuideLora(cases.strings)),
};
for (const [name, raw] of Object.entries(cases)) {
  out.parsed[name] = s.parseGuideLora(raw);
  out.serialized[name] = s.serializeGuideLora(s.parseGuideLora(raw));
}
console.log(JSON.stringify(out));
"""

# A file with no published caption: the node fills an empty prompt from the
# table at queue time and the pill fills it on pick, so on a captioned file
# the two parses differ by design. The table itself is mirrored below.
GUIDE = "h3/minimax_h3_style_transfer_v1.0_r64.safetensors"
CASES = {
    "full": {"on": True, "lora": GUIDE, "strength": 1.3, "prompt": "sharp", "checkpoint": "fl2va",
             "picture": "atlas:000006", "look": "Claymation"},
    "off": {"on": False, "lora": GUIDE},
    "empty": {},
    "garbage": "yes",
    "clamped": {"on": True, "lora": GUIDE, "strength": 9, "checkpoint": "krea2", "prompt": 4},
    "strings": {"on": "true", "lora": f" {GUIDE} ", "strength": "0.5"},
    "spaces": {"on": True, "lora": f" {GUIDE} ", "prompt": "  a  "},
}

NAMES = ["h3/Minimax_H3_LMS_v1.0_r64.safetensors", GUIDE, "lms.safetensors", ""]
js = layout.run(SCRIPT, MIRROR, CASES, NAMES)

for name in NAMES:
    check(f"caption for {name!r}", js["captions"][name], gl.caption_for(name))
check("the style grammar is the node's", js["style"], gl.STYLE)
check("the roles are the node's", js["roles"], gl.ROLES)
for name in NAMES:
    stem = name.split("/")[-1].lower()
    want = next((role["key"] for role in gl.ROLES if role["match"] in stem), None)
    check(f"role of {name!r}", js["role"][name], want)
check("a restyle caption is the trigger, the picture form, the attributes",
      js["caption"], f"{gl.STYLE['prefix']} {gl.STYLE['picture_form']} claymation, visible fingerprint texture.")
check("a block with a picture is a restyle", js["restyle"], True)
check("...and one without is not", js["sharpen"], False)

check("strength stops", js["strength"],
      {"min": gl.MIN_STRENGTH, "max": gl.MAX_STRENGTH, "step": 0.05, "default": gl.DEFAULT_STRENGTH})
check("default checkpoint", js["checkpoint"], gl.DEFAULT_CHECKPOINT)

for name, raw in CASES.items():
    want = gl.Request.of({"guide_lora": raw}).as_dict()
    check(f"{name}: parsed alike", js["parsed"][name], want)
    check(f"{name}: serialized alike", js["serialized"][name],
          {"guide_lora": want} if want["on"] else {})

full = gl.Request.of({"guide_lora": CASES["full"]}).as_dict()
check("a piece round-trips the block", js["piece"], full)
check("a timeline round-trips the block", js["timeline"], full)
check("a piece that never asked writes no block", js["piece_off"], False)
check("a timeline that never asked writes no block", js["timeline_off"], False)

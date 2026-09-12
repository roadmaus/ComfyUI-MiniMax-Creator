"""`guidelora.py`: the guide-LoRA pass's request, its stack, its frame grid.

    python3 tests/test_guidelora.py

Pure: no torch, no ComfyUI.
"""

import layout

pkg = layout.load("canvas", "h3_declare", "guidelora")
gl = pkg.guidelora

from harness import FAILURES, check, passed

passed("the guide-LoRA request reads, clamps, refuses and stacks as written")


class Run:
    def __init__(self, dropped=""):
        self.dropped = dropped


GUIDE = "h3/minimax_h3_lms_v1.0_r64.safetensors"
DISTILL = "h3/turbo.safetensors"
STACK = [{"name": DISTILL, "strength": 0.6, "audio": 0.5, "modes": ["ref2va"]},
         {"name": "h3/character.safetensors", "strength": 0.8}]

# --- reading the block ---------------------------------------------------------------

check("no block is off", bool(gl.Request.of({})), False)
check("garbage is off", bool(gl.Request.of({"guide_lora": "yes"})), False)
check("a string 'true' is not on", bool(gl.Request.of({"guide_lora": {"on": "true", "lora": GUIDE}})), False)
check("off carries no stack", gl.Request.of({"guide_lora": {"on": False, "lora": GUIDE}}).entries, [])

on = gl.Request.of({"guide_lora": {"on": True, "lora": f"  {GUIDE} ", "strength": "1.3",
                                   "prompt": " sharp ", "checkpoint": "fl2va"}})
check("on", bool(on), True)
check("the name is trimmed", on.lora, GUIDE)
check("the strength is a number", on.strength, 1.3)
check("the prompt is trimmed", on.prompt, "sharp")
check("the checkpoint may be the plain one", on.checkpoint, "fl2va")
check("as_dict round-trips", gl.Request.of({"guide_lora": on.as_dict()}), on)

clamped = gl.Request.of({"guide_lora": {"on": True, "lora": GUIDE, "strength": 9,
                                        "checkpoint": "krea2"}})
check("the strength is clamped", clamped.strength, gl.MAX_STRENGTH)
check("an unknown checkpoint falls to the trained one", clamped.checkpoint, gl.DEFAULT_CHECKPOINT)
check("a NaN strength falls to the default",
      gl.Request.of({"guide_lora": {"on": True, "lora": GUIDE, "strength": "nan"}}).strength,
      gl.DEFAULT_STRENGTH)

try:
    gl.Request.of({"guide_lora": {"on": True, "lora": ""}})
    FAILURES.append("on with no file should refuse")
except ValueError as exc:
    check("on with no file refuses, naming the pill", "pill" in str(exc), True)

# --- the stack -------------------------------------------------------------------------

plain = gl.Request.of({"guide_lora": {"on": True, "lora": GUIDE}, "loras": STACK})
check("no turbo: the guide file alone, at the block's strength",
      plain.entries, [{"name": GUIDE, "strength": 1.0}])

turbo = {"guide_lora": {"on": True, "lora": GUIDE, "strength": 1.3},
         "turbo": {"on": True, "lora": DISTILL}, "loras": STACK}
check("turbo: the piece's distill entry as it sits, then the guide — never the rest",
      gl.Request.of(turbo).entries,
      [STACK[0], {"name": GUIDE, "strength": 1.3}])
check("the distill entry is a copy", gl.Request.of(turbo).entries[0] is STACK[0], False)
check("turbo switched off leaves the distill out",
      gl.Request.of({**turbo, "turbo": {"on": False, "lora": DISTILL}}).entries,
      [{"name": GUIDE, "strength": 1.3}])
check("a merged turbo (no file) has nothing to add",
      gl.Request.of({**turbo, "turbo": {"on": True, "merged": True}}).entries,
      [{"name": GUIDE, "strength": 1.3}])
check("a distill disabled in the stack is not worn",
      gl.Request.of({**turbo, "loras": [{**STACK[0], "enabled": False}]}).entries,
      [{"name": GUIDE, "strength": 1.3}])
check("under VDN the dropped distill stays out",
      gl.Request.of(turbo, Run(dropped=DISTILL)).entries, [{"name": GUIDE, "strength": 1.3}])

# --- the grid ----------------------------------------------------------------------------

check("the grid", [gl.padded_frames(n) for n in (1, 5, 6, 22, 23, 102, 124)],
      [5, 5, 22, 22, 39, 107, 124])
try:
    gl.padded_frames(0)
    FAILURES.append("no frames should refuse")
except ValueError:
    pass

"""The guide-LoRA pass, as a piece asks for it: which file, how hard, what it is
told, which checkpoint — and the LoRA stack the pass wears.

**What a guide LoRA is.** An adapter trained with the source clip packed into
the transformer's sequence as an *aligned guide*: the clip is VAE-encoded and
pinned at frame 0 of the target's own timeline, at the target's own canvas, so
guide token `(t, y, x)` sits on target token `(t, y, x)` and the model is
handed the correspondence rather than made to search for it. The guide is held
near-clean in training while the target is noised normally, so what the file
learns is a pixel-level map from one video to another — a sharpener, a style
transfer, a restorer — rather than a description of it. Alissonerdx's
`minimax_h3_lms` ("a little more sharpness") and `minimax_h3_style_transfer`
are the two published against Ref2VA, trained on ostris's ai-toolkit fork with
guide-latent support; both are meant as a second pass over an already-generated
clip. See `guidepass.py` for the pass itself.

**Why this is its own pass and not the refine.** The two-pass refine re-noises
the first pass's latent partway down the schedule; the picture that comes back
is the one that went in, resolved. A guide LoRA generates *from noise*, the
whole schedule, with the finished pass as its guide: what comes back is a new
video that matches the old one pixel for pixel because the file was trained to
make it so. That is a different mechanism with a different promise, and it
runs after every pass is written, at the size it was written — like the DLSS
refiner and for the refiner's reason: a sharpened tail handed to the next seam
would feed the drift the seam already has (a sharpener applied to its own
output compounds; measured on the DLSS refiner, the same ratchet).

**The stack the pass wears is the distillation and the guide file, nothing
else.** The piece's character and style LoRAs are what drew the shot, and the
guide pass is not drawing the shot — it is mapping the drawn one. The
published rig runs exactly the turbo distill and the guide file, and that is
what `entries` builds: the piece's turbo entry as it sits in the piece's stack
(strength, soundtrack dial and claim intact), then the guide file. A piece with
turbo off wears the guide file alone, at the piece's own sampler row, which is
the honest reading of "run this pass the way this piece samples".

No torch, no ComfyUI: `guidepass.py` is what loads the file.
"""

from . import declare

# The blob key. One block on the piece, like `neural`; absent while off, so a
# blob that never asked round-trips to the bytes it always did.
KEY = "guide_lora"

# The checkpoint the published files were trained against, and the one a block
# that names none runs on. FL2VA is offered because the file's own card says it
# should work there too, less tested.
CHECKPOINTS = declare.ROUTED
DEFAULT_CHECKPOINT = "ref2va"

# How hard the guide file is applied. 1.0 is the trainer's own unit; the
# published card runs it there and mentions 1.3 on one example, which is why
# the ceiling is 2 and not 1.
DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 0.0
MAX_STRENGTH = 2.0

# What the published files are told, by filename. A guide file's caption is
# its instruction and the files ship without a sidecar carrying it, so the
# pill and the node both read it from here: the pill writes it into the
# prompt box when the file is picked and the box is empty, the node uses it
# at queue time when the box was left empty. A style file has no fixed
# caption — the style is the prompt — so it is not in the table. Matched
# case-insensitively against the file's name, first hit wins.
CAPTIONS = [
    {"match": "lms",
     "prompt": "Enhance this video with sharp, crisp details while preserving a "
               "natural photorealistic appearance."},
]

# The style-transfer file's caption grammar. Every caption it was trained on
# opens with the trigger; with a reference picture in the layout the sentence
# is a verb, "in the style of <Picture 1>", then three to five attributes a
# painter would copy — never the picture's subject, never an artist or a
# franchise, which resolve the task from text alone and leave the picture
# inert. Without a picture the style is named in words. The frontend composes
# the sentence from a look's descriptor (`presets/stylelib.js`); this is the
# fixed part, served through the manifest so both sides spell it once.
STYLE = {
    "match": "style_transfer",
    "prefix": "style_transfer:",
    "picture_form": "Re-render this video in the style of <Picture 1>:",
}


def caption_for(name):
    """The published caption for the file `name`, or ""."""
    stem = str(name or "").split("/")[-1].lower()
    for entry in CAPTIONS:
        if entry["match"] in stem:
            return entry["prompt"]
    return ""


def _number(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return min(high, max(low, number))


class Request:
    """The pass as a piece asks for it. Read off the blob's `guide_lora` block.

    Clamped rather than refused, on the neural refiner's rule: the numbers come
    off controls that already hold the bounds, so anything out of range is a
    stale frontend or a hand edit and takes the nearest honest value. The one
    thing refused is a block switched on with no file named — that is a pass
    with nothing to be, and the render says so before a loader is built.
    """

    __slots__ = ("on", "lora", "strength", "prompt", "checkpoint", "picture", "look",
                 "entries")

    def __init__(self, on=False, lora="", strength=DEFAULT_STRENGTH, prompt="",
                 checkpoint=DEFAULT_CHECKPOINT, picture="", look="", entries=None):
        # A real boolean only, on both sides of the wire.
        self.on = on is True
        # Strings only, like the pill: a number where a name should be is a
        # hand edit, and reading it as a name would be inventing one.
        self.lora = lora.strip() if isinstance(lora, str) else ""
        self.strength = _number(strength, MIN_STRENGTH, MAX_STRENGTH, DEFAULT_STRENGTH)
        self.prompt = prompt.strip() if isinstance(prompt, str) else ""
        self.checkpoint = checkpoint if checkpoint in CHECKPOINTS else DEFAULT_CHECKPOINT
        # A reference picture beside the guide — a look's frame (`atlas:000123`)
        # or a file under input/ — presented to the model as <Picture 1>. The
        # style file reads it; the sharpener has no use for one.
        self.picture = picture.strip() if isinstance(picture, str) else ""
        # What the look is called, for the pill; nothing here reads it.
        self.look = look.strip() if isinstance(look, str) else ""
        self.entries = list(entries or [])

    @classmethod
    def of(cls, data, run=None):
        """The blob's block, or an off request where there is none.

        `run` is the family's per-queue context (`render.LeadIn`): under VDN-H3
        the distill file is left out of every stack because the stage's own
        adapter is the distillation, and this pass follows the same rule.
        """
        raw = (data or {}).get(KEY) if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return cls()
        request = cls(on=raw.get("on"), lora=raw.get("lora"), strength=raw.get("strength"),
                      prompt=raw.get("prompt"), checkpoint=raw.get("checkpoint"),
                      picture=raw.get("picture"), look=raw.get("look"))
        if request.on and not request.lora:
            raise ValueError(
                "The guide LoRA pass is switched on and no file has been picked. "
                "Open the pass's pill and choose a guide LoRA from models/loras, "
                "or switch the pass off.")
        if request.on and not request.prompt:
            request.prompt = caption_for(request.lora)
        request.entries = entries(data, request, run)
        return request

    def __bool__(self):
        return self.on

    def as_dict(self):
        return {"on": self.on, "lora": self.lora, "strength": self.strength,
                "prompt": self.prompt, "checkpoint": self.checkpoint,
                "picture": self.picture, "look": self.look}

    def __eq__(self, other):
        return isinstance(other, Request) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return f"Request({self.as_dict()})"


def entries(data, request, run=None):
    """The LoRA stack the pass wears: the piece's turbo distill, then the guide.

    The distill entry is taken *from the piece's stack*, not rebuilt: the turbo
    switch put it there with the preset's strength and the user's soundtrack
    dial, and those are the numbers every pass on this piece samples under. A
    turbo switched on that names no file (a merged checkpoint) has nothing to
    add. Under VDN the file is dropped from the piece (`run.dropped`) and so
    from here.
    """
    if not request.on:
        return []
    stack = []
    turbo = (data or {}).get("turbo") if isinstance(data, dict) else None
    distill = str((turbo or {}).get("lora") or "").strip() if isinstance(turbo, dict) \
        and turbo.get("on") is True else ""
    if distill and distill != getattr(run, "dropped", ""):
        for entry in (data.get("loras") or []):
            if isinstance(entry, dict) and entry.get("name") == distill \
                    and entry.get("enabled") is not False:
                stack.append(dict(entry))
                break
    stack.append({"name": request.lora, "strength": request.strength})
    return stack


def padded_frames(count):
    """The length a pass of `count` frames is re-rendered at.

    Every H3 pass is sampled on the `17k+5` grid, but a blended seam trims the
    inherited run off the pass before it is written, so a pass on the reel can
    be any length. The model still wants the grid, and the guide has to be as
    long as the target — so the pass is padded *up* with its own last frame,
    re-rendered whole, and trimmed back after decode. Up rather than down for
    the reason `redetail.padded_frames` gives: down would come back short, with
    the tail silently missing.
    """
    count = int(count)
    if count < 1:
        raise ValueError("a pass with no frames cannot be re-rendered")
    step, offset = declare.RULES.frame_step, declare.RULES.frame_offset
    if count <= offset:
        return offset
    return -(-(count - offset) // step) * step + offset

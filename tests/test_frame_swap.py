"""One frame of a reference clip takes the clip's place as a picture (#81).

    python3 tests/test_frame_swap.py

A clip attached for the one frame in it that is the reference costs every
frame of it through every sampling step, and the picture editor's scrub on a
clip looked like the way to pick one. The editor now answers `{frame, crop}`
for it, and the two hosts fold that answer: the card swaps the clip for an
image row under an image handle and rewrites every citation; the pool keeps
its kind-blind `ref-N` handle and changes the row in place. Both are checked
here with the editor's answer handed to each host's fold directly, since the
saving half is a browser's canvas and upload.

Skips itself if node is not installed.
"""

import layout
from domshim import DOM
from harness import check, passed

layout.skip_without_node()
passed("a frame off a clip stands in the clip's place on both faces")

CHECK = """
await import("./dom.mjs");
const S = await import("./web/creator/state.js");
const { CreatorEditor } = await import("./web/creator/editor.js");
const { TimelineBody } = await import("./web/creator/timeline.js");

const out = {};
const answer = { frame: { path: "prestage_frames/clip_t1.50s.png", time: 1.5 },
                 crop: { x: 0.1, y: 0.1, w: 0.5, h: 0.5 }, source: "clip.mp4", plate: null };

// ---- the card ---------------------------------------------------------------
{
  const piece = S.parseTimeline(JSON.stringify({
    version: 2, prompt: "x",
    segments: [{ prompt: "@vid-1 walks past @img-1 and @vid-10", duration_s: 5,
                 soundscape: "wind over @vid-1",
                 refined: { enabled: true, body: "rewritten @vid-1" },
                 assets: [
                   { handle: "img-1", kind: "image", role: "reference", filename: "a.png" },
                   { handle: "vid-1", kind: "video", role: "reference", filename: "clip.mp4",
                     trim: { start: 1, end: 3 }, track: "picture+sound", ref_size: "match" },
                   { handle: "vid-10", kind: "video", role: "reference", filename: "other.mp4" },
                 ] }] }));
  const state = piece.segments[0];
  const editor = new CreatorEditor({ state, piece, nodeId: () => 1 });
  editor.pieceRef = piece;
  const clip = state.assets.find((a) => a.handle === "vid-1");
  editor.useFrame(clip, answer);
  const still = state.assets.find((a) => a.filename === answer.frame.path);
  out.card = {
    order: state.assets.map((a) => a.handle),
    still: still && { kind: still.kind, role: still.role, crop: still.crop, ref_size: still.ref_size,
                      trim: still.trim ?? null, track: still.track ?? null },
    prompt: state.prompt, soundscape: state.soundscape, refined: state.refined.body,
    clipGone: !state.assets.some((a) => a.kind === "video" && a.filename === "clip.mp4"),
  };
}

// ---- the pool ---------------------------------------------------------------
{
  const store = { value: JSON.stringify({
    version: 2, prompt: "@ref-1 stands there",
    assets: [{ handle: "ref-1", kind: "video", role: "reference", filename: "clip.mp4",
               trim: { start: 0, end: 2 }, track: "picture+sound", ref_size: "max" }],
    subjects: [{ handle: "vera", from: ["ref-1"], motion: "ref-1", description: "a woman" }],
    segments: [{ prompt: "a", duration_s: 5 }, { prompt: "b", duration_s: 5 }] }) };
  const body = new TimelineBody({
    read: () => store.value, write: (next) => { store.value = next; },
    widgets: {}, nodeId: () => 1 });
  const asset = body.timeline.assets[0];
  // The strip's fold is `S.stillForClip` and a commit; the window's own method
  // is the pair of them, on a modal the shim cannot mount.
  S.stillForClip(body.timeline, asset, answer.frame.path, answer.crop);
  body.commit();
  const saved = JSON.parse(store.value);
  const row = saved.assets[0];
  out.pool = {
    handle: row.handle, kind: row.kind, filename: row.filename,
    trim: row.trim ?? null, track: row.track ?? null, crop: row.crop ?? null,
    prompt: saved.prompt,
    subject: { from: saved.subjects[0].from, motion: saved.subjects[0].motion ?? null },
  };
}

console.log(JSON.stringify(out));
"""

with layout.pack(skip=["atlas"]) as target:
    got = layout.in_pack(CHECK.replace('await import("./dom.mjs");', DOM), target)

card = got["card"]
check("card: the still sits where the clip sat", card["order"], ["img-1", "img-2", "vid-10"])
check("card: as an image reference with the window kept",
      card["still"], {"kind": "image", "role": "reference", "crop": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                      "ref_size": "match", "trim": None, "track": None})
check("card: the clip is gone", card["clipGone"], True)
check("card: the prompt cites the still — and @vid-10 is not @vid-1",
      card["prompt"], "@img-2 walks past @img-1 and @vid-10")
check("card: so does the soundscape", card["soundscape"], "wind over @img-2")
check("card: and the rewrite", card["refined"], "rewritten @img-2")

pool = got["pool"]
check("pool: the handle survives", pool["handle"], "ref-1")
check("pool: the row is the still", {"kind": pool["kind"], "filename": pool["filename"]},
      {"kind": "image", "filename": "prestage_frames/clip_t1.50s.png"})
check("pool: what only a clip carries comes off", {"trim": pool["trim"], "track": pool["track"]},
      {"trim": None, "track": None})
check("pool: the window is kept", pool["crop"], {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5})
check("pool: the prompt is untouched", pool["prompt"], "@ref-1 stands there")
check("pool: the member keeps the picture and loses the motion",
      pool["subject"], {"from": ["ref-1"], "motion": None})

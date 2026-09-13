"""What the guide-LoRA pass adds to a graph — and that off, it adds nothing.

    COMFYUI_PATH=~/ComfyUI <comfy-venv>/bin/python3 tests/test_guidepass_graph.py

Nothing is sampled and no weights are loaded: this is the emitted subgraph. The
claims worth pinning are the ones a wrong graph would fail silently on — that
the pass sits after the whole reel and before ReDetail and the refiner, that it
samples on the checkpoint the file was trained against even when no card routes
there, that its stack is the published distill where one is installed — else
the piece's — plus the guide file and nothing else, that its row is its own
and its seed the render's, and that a piece which never asked emits exactly
the graph it always did.

Skips itself with a message if ComfyUI cannot be imported.
"""

import asyncio
import contextlib
import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.basename(ROOT)
COMFY = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))
BASE = os.environ.get("COMFYUI_BASE", COMFY)


def _boot():
    sys.path.insert(0, COMFY)
    sys.argv = ["main.py", "--base-directory", BASE]
    import nodes
    import server

    loop = asyncio.new_event_loop()
    server.PromptServer(loop)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=False))

    sys.path.insert(0, os.path.dirname(ROOT))
    return nodes


try:
    comfy_nodes = _boot()
except Exception as exc:  # noqa: BLE001
    print(f"skipped: ComfyUI not importable ({type(exc).__name__}: {exc})")
    sys.exit(0)

package = importlib.import_module(PACKAGE)
cn = importlib.import_module(f"{PACKAGE}.creator.creator_node")
guidepass = importlib.import_module(f"{PACKAGE}.creator.families.h3.guidepass")
guidelora = importlib.import_module(f"{PACKAGE}.creator.families.h3.guidelora")

from harness import FAILURES, check, passed

passed("the guide-LoRA pass sits after the reel and before ReDetail and the "
       "refiner, on the file's own checkpoint, and off it is not there")

MODEL, PASS = guidepass.MODEL_NODE, guidepass.PASS_NODE

H3_MODELS = {
    "fl2va": "h3/fl2va.safetensors",
    "ref2va": "h3/ref2va.safetensors",
    "clip": "h3/text_encoder.safetensors",
    "vae": "h3/video_vae.safetensors",
    "audio_vae": "h3/audio_vae.safetensors",
}
UPSCALE_MODELS = {
    "dit": "ltx/dit.safetensors",
    "clip": "ltx/gemma4-with-proj.safetensors",
    "vae": "ltx/video-vae.safetensors",
    "audio_vae": "ltx/audio-vae.safetensors",
    "ic_lora": "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
}
NEURAL_ON = {"on": True, "profile": "standard", "scale": 1, "detail": 1.0,
             "colour": 0.0, "intensity": 1.0, "precision": "reference"}

GUIDE = "h3/minimax_h3_lms_v1.0_r64.safetensors"
DISTILL = "h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
CAPTION = "Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance."
ON = {"on": True, "lora": GUIDE, "strength": 1.0, "prompt": CAPTION}
NODE_ID = "7"


def with_id(node_class, unique_id, run):
    from comfy_api.latest import io as comfy_io

    previous = node_class.hidden
    node_class.hidden = comfy_io.HiddenHolder(
        unique_id=unique_id, prompt=None, extra_pnginfo=None, dynprompt=None,
        auth_token_comfy_org=None, api_key_comfy_org=None)
    try:
        return run()
    finally:
        node_class.hidden = previous


def piece(block=None, segments=None, **extra):
    return json.dumps({
        "version": 2, "prompt": "", "models": H3_MODELS, "aspect": "16:9",
        "short_edge": 768,
        **({"guide_lora": block} if block is not None else {}),
        **extra,
        "segments": segments or [{"prompt": "a woman crossing a market", "duration_s": 5}],
    })


def video(data, seed=100, steps=20, sampler="res_multistep", scheduler="simple"):
    return with_id(cn.MiniMaxH3Timeline, NODE_ID,
                   lambda: cn.MiniMaxH3Timeline.execute(
                       timeline_data=data, seed=seed, steps=steps, cfg=1.0,
                       sampler_name=sampler, scheduler=scheduler)).expand


@contextlib.contextmanager
def installed_loras(names):
    """models/loras as the pass sees it: `render.finish_request` asks
    `folder_paths` what is installed, and the test machine has nothing."""
    import folder_paths

    real = folder_paths.get_filename_list
    folder_paths.get_filename_list = lambda folder: list(names) if folder == "loras" \
        else real(folder)
    try:
        yield
    finally:
        folder_paths.get_filename_list = real


def normalised(graph):
    text = json.dumps(graph, sort_keys=True)
    prefix = next(iter(graph)).rsplit(".", 1)[0] + "."
    return text.replace(prefix, "#")


def by_class(graph):
    out = {}
    for node_id, node in graph.items():
        out.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    return out


def loaders(kinds):
    """{unet_name: node id} for every UNETLoader in the graph."""
    return {inputs["unet_name"]: node_id for node_id, inputs in kinds.get("UNETLoader", [])}


# --- off is nothing -------------------------------------------------------------

check("a piece with no block emits no pass", PASS in by_class(video(piece())), False)
check("...and one with the block off is the graph it always was",
      normalised(video(piece({"on": False, "lora": GUIDE}))), normalised(video(piece())))
check("...and a text-only piece loads the plain checkpoint alone",
      sorted(loaders(by_class(video(piece())))), ["h3/fl2va.safetensors"])

try:
    video(piece({"on": True, "lora": ""}))
    FAILURES.append("a pass switched on with no file should refuse")
except ValueError as exc:
    check("a pass with no file names the pill", "guide LoRA" in str(exc), True)


# --- the pass ---------------------------------------------------------------------

graph = video(piece(ON))
kinds = by_class(graph)
check("one pass for the whole reel", len(kinds.get(PASS, [])), 1)
check("one model node for it", len(kinds.get(MODEL, [])), 1)
pass_id, pass_inputs = kinds[PASS][0]
model_id, model_inputs = kinds[MODEL][0]
reel_id, _ = kinds["MiniMaxH3Reel"][0]
check("it takes the finished reel", pass_inputs["reel"], [reel_id, 0])
check("the file is written from what it hands back",
      kinds["MiniMaxH3Save"][0][1]["reel"], [pass_id, 0])
check("the model node wears the guide file alone on a piece without turbo",
      json.loads(model_inputs["loras"]), [{"name": GUIDE, "strength": 1.0}])
check("it samples on the reference checkpoint", model_inputs["checkpoint"], "ref2va")
check("...whose loader was built although no card routes there",
      model_inputs["model"], [loaders(kinds)["h3/ref2va.safetensors"], 0])
check("the plain checkpoint the card routes to is still loaded",
      "h3/fl2va.safetensors" in loaders(kinds), True)


def upstream(graph, link, want):
    """Whether following `model` inputs back from `link` reaches node `want`."""
    while link:
        if link[0] == want:
            return True
        link = graph[link[0]]["inputs"].get("model")
    return False


check("the pass's model is the stacked one, behind the patches",
      upstream(graph, pass_inputs["model"], model_id), True)
check("the prompt is the caption", pass_inputs["prompt"], CAPTION)
check("its row is its own, not the piece's",
      {key: pass_inputs[key] for key in ("steps", "cfg", "sampler_name", "scheduler")},
      {key: guidelora.ROW[key] for key in ("steps", "cfg", "sampler_name", "scheduler")})
check("its seed is the render's", pass_inputs["seed"], 100)
check("a re-rolled piece seeds it differently",
      by_class(video(piece(ON), seed=101))[PASS][0][1]["seed"], 101)
check("the encoder and the VAE are the render's own",
      (pass_inputs["clip"], pass_inputs["vae"]),
      (kinds["MiniMaxH3Reel"][0][1]["vae"] and [kinds["CLIPLoader"][0][0], 0],
       kinds["MiniMaxH3Reel"][0][1]["vae"]))

# The stack under turbo, with no published distill installed: the piece's
# distill entry as it sits in the piece's stack, then the guide file — and
# nothing else the piece wears. The row stays the pass's own.
turbo = piece(ON, turbo={"on": True, "lora": DISTILL, "quality": "medium"},
              loras=[{"name": DISTILL, "strength": 0.6, "audio": 0.5},
                     {"name": "h3/some_character.safetensors", "strength": 0.8}])
kinds = by_class(video(turbo, steps=8, sampler="euler", scheduler="beta"))
check("under turbo the stack is the piece's distill then the guide",
      json.loads(kinds[MODEL][0][1]["loras"]),
      [{"name": DISTILL, "strength": 0.6, "audio": 0.5}, {"name": GUIDE, "strength": 1.0}])
check("...still on the pass's own row",
      (kinds[PASS][0][1]["steps"], kinds[PASS][0][1]["scheduler"]),
      (guidelora.ROW["steps"], guidelora.ROW["scheduler"]))

# With the published distill installed the pass wears that at 1.0 instead,
# whatever the piece's turbo says.
PUBLISHED = "h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
with installed_loras([PUBLISHED, DISTILL]):
    kinds = by_class(video(turbo, steps=8, sampler="euler", scheduler="beta"))
check("with the published distill installed the pass wears it",
      json.loads(kinds[MODEL][0][1]["loras"]),
      [{"name": PUBLISHED, "strength": guidelora.DISTILL_STRENGTH},
       {"name": GUIDE, "strength": 1.0}])

# The strength and the checkpoint are the block's; the strength is clamped.
kinds = by_class(video(piece({**ON, "strength": 5, "checkpoint": "fl2va"})))
check("the strength is clamped onto the file's range",
      json.loads(kinds[MODEL][0][1]["loras"])[0]["strength"], guidelora.MAX_STRENGTH)
check("the checkpoint may be the plain one", kinds[MODEL][0][1]["checkpoint"], "fl2va")
check("...and then only the plain one is loaded", sorted(loaders(kinds)),
      ["h3/fl2va.safetensors"])

# The piece's flow shift does not reach the pass: it samples on the
# checkpoints' own shifts, so no shift node sits between the stack and it.
shifted = json.loads(piece(ON))
shifted["sampling"] = {"shift_video": 6, "shift_audio": 3}
kinds = by_class(video(json.dumps(shifted)))
shift = [inputs for _, inputs in kinds.get("MiniMaxH3SigmaShift", [])
         if inputs["model"] == [kinds[MODEL][0][0], 0]]
check("the piece's flow shift stays off the pass", len(shift), 0)

# The picture: a look's frame as <Picture 1>, only written when there is one.
check("a sharpen writes no picture input", "picture" in pass_inputs, False)
kinds = by_class(video(piece({**ON, "picture": "atlas:000006", "look": "Claymation"})))
check("a restyle hands the pass its picture", kinds[PASS][0][1]["picture"], "atlas:000006")

# --- where it sits ----------------------------------------------------------------

kinds = by_class(video(piece(ON, upscale="redetail", upscale_models=UPSCALE_MODELS)))
pass_id, pass_inputs = kinds[PASS][0]
redetail_id, redetail_inputs = kinds["MiniMaxReDetailPass"][0]
check("ReDetail reads the finished reel", redetail_inputs["reel"], [pass_id, 0])
check("...and the pass read the written one", pass_inputs["reel"],
      [kinds["MiniMaxH3Reel"][0][0], 0])

kinds = by_class(video(piece(ON, neural=NEURAL_ON)))
check("the refiner reads the finished reel",
      kinds["ContinuityNeuralPass"][0][1]["reel"], [kinds[PASS][0][0], 0])

# A strip of two: the takes come off the finished reel, not off each pass.
two = piece(ON, segments=[{"prompt": "a", "duration_s": 3}, {"prompt": "b", "duration_s": 3}])
kinds = by_class(video(two))
check("on a strip the passes do not write their own takes",
      "ContinuityTake" in kinds, False)
check("...the save node keeps them off the finished reel",
      json.loads(kinds["MiniMaxH3Save"][0][1]["takes"])["cards"], [1, 2])

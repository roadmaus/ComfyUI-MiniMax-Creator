"""The guide-LoRA pass: every generated pass on the reel, generated again from
noise with itself as the aligned guide, under a file trained to map one to the
other.

`guidelora.py` decides — what the request is, which files the pass wears, the
frame grid. This is the half that holds weights and pixels.

**The arrangement, and why it is the seam's.** The finished pass is read back,
encoded by the video VAE, and pinned as one guide block at frame 0 of an empty
AV latent of the same canvas and the same length — `minimax_keyframes` with
`resolved_frame_index: 0`, exactly what core's `MiniMaxH3AddGuide` writes and
exactly what the published files were trained against. The blended seam
already does this with the run a pass inherits (`encode._context_keyframes`);
here the run is the whole pass. Then the schedule runs from noise, the whole
of it: this is not a refine that resumes partway down, the file was trained to
generate the target from the guide and it is given the same job here.

**The sound is not regenerated, and not conditioned either.** The AV latent's
audio half samples from noise beside the picture — the model was trained to
denoise the pair and is handed the pair — and what it makes is thrown away:
`spill.rewrite` points the new pass at the soundtrack the pass already had,
which is the one the user heard when they decided the picture wanted
sharpening. The published rig does the same. Pinning the source's own sound as
an audio guide is an experiment this pass does not run.

**One block, not one per latent step.** The seam pins its run as one block per
step because a feathered run's steps land at offsets the layout has to be told
one by one. A guide anchored at 0 covering the whole target needs no such
thing, and one block is the shape the trainer packed — so one block it is.

**Per pass, in play order, on the reel, at the end.** After every pass is
written and before ReDetail and the DLSS refiner: it keeps the size, so it
could run inline, and it runs at the end for the refiner's reason — a
sharpened tail handed on as the next seam's anchor is a ratchet, and the
form worth measuring first is the uniform one. Each part is generated on its
own, so across a feathered seam two independent generations meet; the guide
is near-clean in training and the output is pixel-locked to it, so the join
should hold. That is the open measurement, written here rather than left to
be discovered.
"""

import json
import logging

import numpy as np
import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
from comfy_api.latest import io

from ... import lora, media, spill
from ...timeline import REEL_TYPE
from . import guidelora

MODEL_NODE = "MiniMaxH3GuideModel"
PASS_NODE = "MiniMaxH3GuidePass"

# How many frames are read off the memmap at a time on the way into the guide.
CHUNK = 16


class GuidePassError(RuntimeError):
    """The guide-LoRA pass could not run on this reel."""


class MiniMaxH3GuideModel(io.ComfyNode):
    """The checkpoint wearing the pass's own stack: the distill and the guide file.

    Its own node rather than a patch inside the pass, so the sigma shift, the
    accelerators and the preview decoder go on *after* the LoRAs, in the order
    every other sampler in this family runs behind (`render.patched`) — a
    block-cache accelerator refuses to sit downstream of a block replacement,
    and the vendored stack may run a file as a live branch.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=MODEL_NODE,
            display_name="MiniMax H3 Guide Model",
            category="Continuity/internal",
            description="The checkpoint with the guide-LoRA pass's stack on it. "
                        "Written into the graph by render.emit.",
            is_dev_only=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("loras", multiline=True,
                    tooltip="The stack, as JSON: the piece's turbo distill where it "
                            "has one, then the guide file."),
                io.String.Input("checkpoint",
                    tooltip="Which of H3's checkpoints `model` is, for the claims."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, loras, checkpoint) -> io.NodeOutput:
        entries = json.loads(loras) if str(loras or "").strip() else []
        if not entries:
            raise GuidePassError("the guide-LoRA pass was given no file to wear")
        return io.NodeOutput(lora.apply(model, entries, checkpoint))


def _guide(frames, length):
    """The finished pass as the model's guide: `[length, H, W, 3]`, 0..1 float,
    padded up to `length` with its own last frame (`guidelora.padded_frames`)."""
    count, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    out = torch.empty((length, height, width, 3), dtype=torch.float32)
    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        block = torch.from_numpy(np.array(frames[start:stop, ..., :3]))
        out[start:stop] = block.float().div_(255.0)
    if length > count:
        out[count:] = out[count - 1]
    return out


def _reference(image):
    """A reference picture at the reference pipeline's own size: down-only to
    the 2048 short edge on the /32 grid, the `max` setting of a cast look —
    the same arithmetic as `encode.encode_image`, for the same file."""
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, _resize

    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(width, height))
    snap = lambda value: max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(image, snap(width * scale), snap(height * scale), "disabled")


def _blocks(decoded, count):
    """The re-rendered pass, a chunk at a time, trimmed to the length that went in."""
    for start in range(0, count, CHUNK):
        yield decoded[start:min(start + CHUNK, count), ..., :3].clamp(0.0, 1.0)


class MiniMaxH3GuidePass(io.ComfyNode):
    """Every generated pass on a reel, generated again with itself as the guide."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=PASS_NODE,
            display_name="MiniMax H3 Guide Pass",
            category="Continuity/internal",
            description="Re-generates every pass on the reel from noise with the pass "
                        "itself pinned as an aligned guide, under a guide LoRA. "
                        "Written into the graph by render.emit.",
            is_dev_only=True,
            inputs=[
                io.Model.Input("model", tooltip="The checkpoint wearing the pass's stack."),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Custom(REEL_TYPE).Input("reel"),
                io.String.Input("prompt", multiline=True,
                    tooltip="What the file is told — its trigger caption, or for a "
                            "style file, the style."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=8, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.String.Input("picture", default="", optional=True,
                    tooltip="A reference picture presented as <Picture 1> beside the "
                            "guide — a look's frame (atlas:000123) or a file under "
                            "input/. The style file reads it."),
            ],
            outputs=[io.Custom(REEL_TYPE).Output(display_name="reel")],
        )

    @classmethod
    def execute(cls, model, clip, vae, reel, prompt, seed, steps, cfg, sampler_name,
                scheduler, picture="") -> io.NodeOutput:
        parts = list(reel or [])
        passes = [index for index, part in enumerate(parts) if "pass" in part]
        if not passes:
            # Held takes are compiled to clips; assembling them must not need
            # the weights on a machine that only wants to save the reel.
            if parts and all("clip" in part for part in parts):
                return io.NodeOutput(parts)
            raise GuidePassError("the guide-LoRA pass was given a reel with nothing "
                                 "generated on it — there is nothing to re-render.")

        # The caption, once: the guide is what differs between the parts, and
        # the text is the same sentence over every one of them. So is the
        # picture, where there is one — a reference block the way `encode.py`
        # builds a Ref2VA reference, presented to the tokenizer as <Picture 1>
        # and laid out in front of the target, beside the guide pinned on it.
        items, blocks = [], []
        if str(picture or "").strip():
            resized = _reference(media.load_image(picture.strip()))
            items.append({"type": "image", "data": resized})
            blocks.append({"kind": "image",
                           "latent_h": int(resized.shape[1]) // 16,
                           "latent_w": int(resized.shape[2]) // 16,
                           "latent": vae.encode(resized)})
        tokens = clip.tokenize(prompt, minimax_ref_items=items) if items \
            else clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if blocks:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": blocks})

        out = list(parts)
        for position, index in enumerate(passes):
            comfy.model_management.throw_exception_if_processing_interrupted()
            out[index] = {"pass": cls._one(
                parts[index]["pass"], model, conditioning, vae, seed + position,
                steps, cfg, sampler_name, scheduler)}
        return io.NodeOutput(out)

    @classmethod
    def _one(cls, source, model, conditioning, vae, seed, steps, cfg, sampler_name,
             scheduler):
        """One pass off the reel, re-generated. -> the spec for its replacement."""
        import nodes
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        frames = spill.open_frames(source)
        count = int(source["frames"])
        width, height = int(source["width"]), int(source["height"])
        length = guidelora.padded_frames(count)
        logging.info("[MiniMax] guide pass: %d frames (%d on the grid) at %dx%d, "
                     "%d steps", count, length, width, height, steps)

        latent, aligned = _empty_av_latent(width, height, length)
        if aligned != length:
            raise GuidePassError(
                f"a {length}-frame pass is not on H3's frame grid ({aligned} is) — "
                f"`guidelora.padded_frames` and core disagree about the grid")
        shell_video, _ = latent["samples"].unbind()

        guide = vae.encode(_guide(frames, length))
        if guide.ndim == 4:                       # [B,C,H,W] -> [1,C,T,H,W]
            guide = guide.unsqueeze(0).movedim(1, 2)
        if guide.shape[-3:] != shell_video.shape[-3:]:
            raise GuidePassError(
                f"the pass encoded to {tuple(guide.shape[-3:])} where a "
                f"{length}-frame target wants {tuple(shell_video.shape[-3:])} — "
                f"is the H3 video VAE wired to 'vae'?")

        # The whole pass, one block, anchored at frame 0 of the target's own
        # timeline: core's `MiniMaxH3AddGuide` with `frame_idx = 0`, and the
        # shape the file was trained on. Both conditionings carry it — cfg 1.0
        # never reads the negative, and a negative without the guide would be
        # a different layout if it ever did.
        positive = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": [{"resolved_frame_index": 0, "latent": guide}],
            "minimax_frame_count": length,
        })
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]

        samples = latent["samples"]
        noise = comfy.sample.prepare_noise(samples, seed)
        sampled = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
            samples, denoise=1.0, seed=seed,
            callback=latent_preview.prepare_callback(model, steps),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED)
        del guide, positive, negative, noise

        # Core's decoder takes the AV pair and decodes the picture half; the
        # sound half is left where it is and never written anywhere.
        decoded = nodes.VAEDecode().decode(vae, {"samples": sampled})[0]
        del sampled
        if int(decoded.shape[0]) < count:
            raise GuidePassError(
                f"the pass decoded to {int(decoded.shape[0])} frames where "
                f"{count} went in")
        written = spill.rewrite(source, _blocks(decoded, count))
        del decoded
        comfy.model_management.soft_empty_cache()
        return written


def emit(graph, model, links, sampling, reel, request, seed):
    """The pass, wired onto the end of a render. -> the new reel link.

    `model` is the checkpoint link already wearing the stack and patched the
    way every sampler in this family is (`render.emit_finish` builds it, so the
    shift, the accelerators and the preview decoder are applied in the order
    `render.patched` fixes). The sampler row is the piece's own — under turbo
    that is the turbo row, which is the published rig; without it the pass runs
    the way the piece samples, which is the honest reading of "finish this
    piece the way it was made".
    """
    inputs = {}
    if request.picture:
        # Only when there is one: an input the graph does not write is an
        # input the node's cache key does not carry, so a sharpen keeps the
        # key it had before pictures existed.
        inputs["picture"] = request.picture
    return graph.node(
        PASS_NODE, model=model, clip=links.clip, vae=links.vae, reel=reel,
        prompt=request.prompt, seed=seed, steps=sampling.steps, cfg=sampling.cfg,
        sampler_name=sampling.sampler_name, scheduler=sampling.scheduler,
        **inputs).out(0)


NODES = [MiniMaxH3GuideModel, MiniMaxH3GuidePass]

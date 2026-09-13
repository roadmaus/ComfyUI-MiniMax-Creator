// A finished render, generated again in a look from the style atlas.
//
// The pieces already exist: the guide-LoRA pass (`pills.js`, `guidelora.py`)
// takes a file, a picture and a caption; the library's Style tab holds nine
// hundred looks with a frame each; the reel's passes are cached, so a render
// re-queued with only the pass changed samples only the pass. This module is
// the door between them, shared by the stage's chip, the pill's row and both
// nodes: open the library as a picker with this render's frame in the wipe,
// and write what comes back onto the piece's block.
//
// The style file is found rather than picked: there is one published, its
// name says what it is, and a picker for a file with one answer is a form
// nobody should fill in. When it is not installed the library's button says
// so and where to get it, and nothing is written.

import { listLoraNames } from "./api.js";
import { openPresetLibrary } from "./presetlib.js";
import { atlasRef } from "./presets/atlasref.js";
import { queueNode } from "./queue.js";
import { emptyGuideLora, guideLoraStyle, restyleCaption } from "./state.js";

/** The style file under models/loras, by the name its author gave it, or "". */
export async function styleFile(family) {
  const grammar = guideLoraStyle(family);
  if (!grammar) return "";
  try {
    const names = await listLoraNames();
    return names.find((name) => name.toLowerCase().includes(grammar.match)) ?? "";
  } catch {
    return "";
  }
}

/**
 * Open the library as a look picker for `target`, and on a pick write the
 * pass's block and, where asked, queue the node.
 *
 * @param {object} spec
 * @param {object} spec.target   the piece or timeline state, mutated in place
 * @param {string} spec.family   the family the piece renders on
 * @param {string|null} [spec.frame]  a URL of the render's own frame, for the wipe
 * @param {() => void} spec.commit
 * @param {string|number|null} [spec.nodeId]  queue this node on the pick
 */
export async function openRestyle({ target, family, frame = null, commit, nodeId = null }) {
  const file = await styleFile(family);
  return openPresetLibrary({
    restyle: {
      frame, family, file,
      onRestyle: ({ row, clip, attributes, strength }) => {
        const block = target.guide_lora ?? (target.guide_lora = emptyGuideLora());
        block.on = true;
        block.lora = file;
        block.picture = atlasRef(clip);
        block.look = row.name;
        block.prompt = restyleCaption(attributes, family);
        block.strength = strength;
        commit();
        if (nodeId != null) queueNode(nodeId);
      },
    },
  });
}

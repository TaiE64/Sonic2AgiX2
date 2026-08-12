"""Recover the obs block order from an exported policy graph.

``gear_sonic/utils/inference_helpers.py`` builds the export's slice layout with
``list(set(encoder_tokenizer_obs + decoder_tokenizer_obs))``.  Python salts str
hashes per process, so that list -- and therefore the order the 1570-vector is
sliced in -- is a coin flip per export.  Same checkpoint, re-exported, can come
out the other way (v19 s9000/s17500 are maob-first, s12000/s26000 cmf-first).

Nothing records the choice, but the Slice offsets are traced into the graph, so
the export tells us itself.  Feeding the wrong order raises no error: the policy
reads velocities as positions and the robot falls (measured: swapping cmf/maob
on v21 s12000 takes jerr 0.160 -> 0.263 and 0 -> 11 resets).

Kept dependency-light on purpose (numpy + onnx) so deploy tooling can import it
without pulling in mujoco/isaaclab.
"""

from __future__ import annotations

BLOCK_DIMS = {"cmf": 580, "maob": 60, "pro": 930}
OBS_DIM = sum(BLOCK_DIMS.values())
TOKENIZER_DIM = BLOCK_DIMS["cmf"] + BLOCK_DIMS["maob"]


def block_order_from_onnx(onnx_path):
    """Return the ("cmf"|"maob", ..., "pro") tuple this export expects.

    ``pro`` is always last: the wrapper appends proprioception after the
    tokenizer segment instead of putting it in the randomized set.  So the only
    question is which block leads the tokenizer segment, and the length of the
    first cut inside that segment answers it.  Returns None if the graph does
    not match this export shape.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.attribute:
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    lengths = []
    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue
        start, end = constants.get(node.input[1]), constants.get(node.input[2])
        if start is None or end is None:
            continue
        low = int(np.asarray(start).reshape(-1)[0])
        high = min(int(np.asarray(end).reshape(-1)[0]), OBS_DIM)
        if high > low and low < TOKENIZER_DIM and high - low < TOKENIZER_DIM:
            lengths.append(high - low)

    if not lengths:
        return None
    if lengths[0] == BLOCK_DIMS["cmf"]:
        return ("cmf", "maob", "pro")
    if lengths[0] == BLOCK_DIMS["maob"]:
        return ("maob", "cmf", "pro")
    return None

"""
Collect Llama-3.1-8B residual-stream activations for the diverse-prompt
corpus (and optionally the existing arithmetic/control corpora).

Reuses the same model loading and hook-based collection machinery as
`llama_substrate.py` (which we import from), so the residual snapshots are
captured at exactly the same points: the LAST TOKEN POSITION across all
sublayer hook points (input to each block's input_layernorm
[pre-attention] and post_attention_layernorm [pre-MLP], plus input to the
final model.norm). For Llama-3.1-8B that is 2*32 + 1 = 65 sublayers and
d_model = 4096 units per sublayer.

Both UNTRIMMED and TRIMMED versions are stored. The trimmed version drops
K sublayers from each end (default K=2, matching the substrate pipeline);
this matches what the website's existing JSONs expect. The full untrimmed
array is also kept so you can re-derive trimmed views at different K
values later without re-running the model.

Usage
-----

    # Quantized (Q8, matches the substrate script's default).
    python collect_diverse_activations.py \\
        --output diverse_activations.npz

    # Specify a different model id, output path, trim, dtype.
    python collect_diverse_activations.py \\
        --model meta-llama/Llama-3.1-8B \\
        --output out/diverse_activations.npz \\
        --trim 2 \\
        --save-dtype float16

    # Also include arithmetic + control prompts from llama_substrate.
    python collect_diverse_activations.py \\
        --output combined_activations.npz \\
        --include-arithmetic --n-arithmetic 100 \\
        --include-controls   --n-controls   100

Output NPZ keys
---------------

    residuals_untrimmed : (N, n_sub_full, d_model) float
        Full, untrimmed snapshots. n_sub_full = 65 for Llama-3.1-8B.
    residuals_trimmed   : (N, n_sub_full - 2*trim, d_model) float
        Same data with `trim` sublayers dropped from each end.
    prompts             : (N,) object array of str
    tasks               : (N,) object array of str  (task labels)
    trim                : int   (the K used for the trimmed view)
    n_sublayers_full    : int
    n_sublayers_trimmed : int
    d_model             : int
    model_id            : str
    use_q8              : bool
    save_dtype          : str
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make `llama_substrate.py` importable when the file lives in the same dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _import_heavy():
    """Lazy import of model-related machinery from llama_substrate.

    Doing this inside main() means --help works on machines that don't have
    torch/transformers installed yet. The substrate script's import chain
    pulls in torch, so importing it at module scope would block --help.
    """
    from llama_substrate import (
        get_device,
        load_llama_model,
        get_llama_hook_targets,
        collect_residual_streams,
        make_arithmetic_prompts,
        make_control_prompts,
    )
    return dict(
        get_device=get_device,
        load_llama_model=load_llama_model,
        get_llama_hook_targets=get_llama_hook_targets,
        collect_residual_streams=collect_residual_streams,
        make_arithmetic_prompts=make_arithmetic_prompts,
        make_control_prompts=make_control_prompts,
        make_diverse_prompts=make_diverse_prompts,
    )


# ---------------------------------------------------------------------------
# Diverse prompt bank (inlined so this script is self-contained; no need
# to keep diverse_prompts.py alongside it).
#
# Each entry is a (prompt_template, task_label) tuple. Some templates are
# bare strings; others contain {placeholder} fields filled in by the
# generator. Tasks are deliberately diverse within each category so that
# users scrubbing through the dropdown see "many examples of the same kind
# of computation," not just one cherry-picked instance.
# ---------------------------------------------------------------------------

_FACTUAL_RECALL = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Brazil is",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
    "The chemical symbol for gold is",
    "The author of Romeo and Juliet was",
    "The longest river in the world is the",
    "The currency of the United Kingdom is the",
    "The deepest ocean on Earth is the",
    "The tallest mountain on Earth is",
    "The first president of the United States was",
    "The smallest planet in the solar system is",
    "The Great Wall is located in",
    "The composer of the Ninth Symphony was",
]

_COMMON_SENSE = [
    "When water reaches 0 degrees Celsius, it",
    "If you drop a glass on a hard floor, it will probably",
    "Birds use their wings primarily to",
    "Fire requires fuel, heat, and",
    "Plants grow toward sources of",
    "When you exhale, your lungs release",
    "An umbrella is most useful when it is",
    "To make ice cubes, you put water in the",
    "The opposite of north is",
    "If a candle has no oxygen, the flame will",
    "A magnet will attract objects made of",
    "When you turn off a lamp, the room becomes",
]

_LANGUAGE = [
    "The plural of 'mouse' is",
    "The plural of 'child' is",
    "The past tense of 'run' is",
    "The past tense of 'go' is",
    "The opposite of 'hot' is",
    "The opposite of 'fast' is",
    "A young dog is called a",
    "A group of birds is called a",
    "The word 'enormous' means very",
    "Something that cannot be broken is",
    "A person who writes books is called an",
    "A person who flies planes is called a",
]

_REASONING = [
    "Tom has 3 apples. Mary gives him 2 more. Tom now has",
    "A square has four sides. A triangle has",
    "If today is Wednesday, two days from now will be",
    "All cats are mammals. Whiskers is a cat. Therefore Whiskers is a",
    "A car travels at 60 miles per hour for 2 hours. It covers",
    "If A is taller than B, and B is taller than C, then A is taller than",
    "Sarah is older than Ben. Ben is older than Tim. The oldest is",
    "A baker makes 12 loaves. He sells 7. He has",
    "If it rains, the ground gets wet. The ground is wet. It probably",
    "Half of 100 is",
    "Twice 25 is",
    "A week before Friday is",
]

_CODE = [
    "def add(a, b):\n    return a +",
    "def is_even(n):\n    return n % 2 ==",
    "for i in range(10):\n    print(",
    "if x > 0:\n    print('positive')\nelse:\n    print(",
    "# Reverse a string\ndef reverse(s):\n    return s[::",
    "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.",
    "SELECT * FROM users WHERE age >",
    "<html>\n  <head>\n    <title>",
    "let sum = 0;\nfor (let i = 0; i < 10; i++) {\n  sum +=",
    "function greet(name) {\n  return 'Hello, ' +",
]

_TRANSLATION = [
    "The English word 'hello' translated to Spanish is",
    "The English word 'cat' translated to French is",
    "The English word 'house' translated to German is",
    "The English word 'book' translated to Italian is",
    "The English word 'water' translated to Japanese is",
    "The Spanish word 'gracias' in English means",
    "The French word 'bonjour' in English means",
    "The German word 'Danke' in English means",
]

_CREATIVE = [
    "Once upon a time, in a kingdom far away, there lived a",
    "It was a dark and stormy night when suddenly the door",
    "The old wizard opened the dusty book and began to",
    "She had never seen the ocean before, and when she finally did, she",
    "The detective examined the room carefully, looking for any sign of",
    "Deep in the forest, beyond the river, stood an ancient",
    "He turned the key, opened the door, and saw",
    "The spaceship landed quietly in the field, and out stepped a",
]

_DIVERSE_CATEGORIES = {
    "factual_recall": _FACTUAL_RECALL,
    "common_sense":   _COMMON_SENSE,
    "language":       _LANGUAGE,
    "reasoning":      _REASONING,
    "code":           _CODE,
    "translation":    _TRANSLATION,
    "creative":       _CREATIVE,
}


def make_diverse_prompts(rng, per_category=None):
    """Return a shuffled list of {prompt, task} dicts spanning the
    seven categories above.

    Parameters
    ----------
    rng : numpy.random.Generator
        For reproducible shuffling.
    per_category : int or None
        If given, sample up to this many prompts per category. If None,
        include all available prompts.
    """
    out = []
    for task, prompts in _DIVERSE_CATEGORIES.items():
        if per_category is not None and per_category < len(prompts):
            idxs = rng.choice(len(prompts), size=per_category, replace=False)
            chosen = [prompts[i] for i in idxs]
        else:
            chosen = list(prompts)
        for p in chosen:
            out.append({"prompt": p, "task": task})
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Collect residual-stream activations across all "
                     "sublayers and units for the diverse-prompt corpus."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", "-o", required=True,
                   help="Path for output .npz file.")
    p.add_argument("--model",
                   default="meta-llama/Llama-3.1-8B",
                   help="Hugging Face model id (default: Llama-3.1-8B).")
    p.add_argument("--no-q8", action="store_true",
                   help="Disable 8-bit quantization (default: Q8 on, "
                        "matching how the substrate script was run).")
    p.add_argument("--device", default=None,
                   help="Override device (cuda / cpu / mps). Auto-detected "
                        "by default.")
    p.add_argument("--hf-home", default=None,
                   help="Override HF_HOME for model weights cache.")

    p.add_argument("--trim", type=int, default=2,
                   help="Sublayers to drop from each end for the trimmed "
                        "view (default: 2, matches substrate pipeline).")
    p.add_argument("--max-length", type=int, default=512,
                   help="Tokenizer max length per prompt (default: 512).")
    p.add_argument("--save-dtype", default="float32",
                   choices=["float32", "float16"],
                   help="Storage dtype for residuals. float16 halves file "
                        "size, with negligible loss for visualization. "
                        "Default: float32.")

    # Corpus selection -------------------------------------------------------
    p.add_argument("--per-category", type=int, default=None,
                   help="If given, cap diverse prompts per category at this "
                        "number. Default: include all bank entries.")
    p.add_argument("--include-arithmetic", action="store_true",
                   help="Also include the arithmetic corpus from "
                        "llama_substrate.make_arithmetic_prompts.")
    p.add_argument("--n-arithmetic", type=int, default=80,
                   help="If --include-arithmetic, how many to generate "
                        "(default: 80, ~20 per task family).")
    p.add_argument("--include-controls", action="store_true",
                   help="Also include the control corpus from "
                        "llama_substrate.make_control_prompts.")
    p.add_argument("--n-controls", type=int, default=80,
                   help="If --include-controls, how many to generate "
                        "(default: 80).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reproducible prompt generation "
                        "(default: 0). Note: the residual snapshots "
                        "themselves are deterministic from the prompts.")

    p.add_argument("--log-every", type=int, default=32,
                   help="Print progress every N prompts (default: 32).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompts(args, rng, mods):
    """Build the combined prompt list per the CLI flags.

    Diverse prompts always come first, then arithmetic, then controls,
    so indices in the output are predictable when only a subset is asked
    for. (Within each block, order is shuffled by its own generator.)
    """
    blocks = []

    diverse = mods["make_diverse_prompts"](rng, per_category=args.per_category)
    blocks.append(("diverse", diverse))

    if args.include_arithmetic:
        arith = mods["make_arithmetic_prompts"](args.n_arithmetic, rng)
        # The arithmetic generator returns dicts with extra keys (a, b,
        # true_sum, true_concept). We only need prompt + task here.
        arith = [{"prompt": p["prompt"], "task": p["task"]} for p in arith]
        blocks.append(("arithmetic", arith))

    if args.include_controls:
        ctrl = mods["make_control_prompts"](args.n_controls, rng)
        ctrl = [{"prompt": p["prompt"], "task": p["task"]} for p in ctrl]
        blocks.append(("controls", ctrl))

    all_prompts = []
    for _, block in blocks:
        all_prompts.extend(block)

    summary = ", ".join(f"{name}={len(block)}" for name, block in blocks)
    print(f"Assembled {len(all_prompts)} prompts ({summary}).")
    return all_prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    mods = _import_heavy()
    device = mods["get_device"](args.device)
    rng = np.random.default_rng(args.seed)

    print(f"Device: {device}")
    print(f"Model:  {args.model}  (Q8: {not args.no_q8})")
    print(f"Output: {args.output}")
    print()

    # ----- 1. Prompts -----
    prompts = assemble_prompts(args, rng, mods)
    prompt_strs = [p["prompt"] for p in prompts]
    task_strs = [p["task"] for p in prompts]

    # ----- 2. Model -----
    print(f"\nLoading model ...")
    t0 = time.time()
    model, tokenizer = mods["load_llama_model"](
        args.model, device,
        use_q8=(not args.no_q8),
        hf_home=args.hf_home,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s.")

    base = model.model if hasattr(model, "model") else model
    n_layers = len(base.layers)
    d_model = base.norm.weight.shape[0]
    n_sub_expected = 2 * n_layers + 1
    print(f"  n_layers={n_layers}, d_model={d_model}, "
          f"n_sublayers={n_sub_expected}")

    if args.trim * 2 >= n_sub_expected:
        raise ValueError(
            f"trim={args.trim} would leave <=0 sublayers in trimmed view "
            f"(n_sublayers={n_sub_expected})."
        )

    # ----- 3. Collect -----
    print(f"\nCollecting residuals for {len(prompts)} prompts ...")
    t0 = time.time()
    residuals = mods["collect_residual_streams"](
        model, tokenizer, prompt_strs, device,
        max_length=args.max_length, log_every=args.log_every,
    )
    print(f"  Done in {time.time() - t0:.1f}s.")
    print(f"  residuals shape: {residuals.shape}  "
          f"dtype={residuals.dtype}")

    # ----- 4. Cast dtype, build trimmed view -----
    if args.save_dtype == "float16":
        residuals = residuals.astype(np.float16)
    else:
        residuals = residuals.astype(np.float32)

    K = args.trim
    if K > 0:
        residuals_trim = residuals[:, K:-K, :]
    else:
        residuals_trim = residuals
    print(f"  residuals_untrimmed: {residuals.shape}")
    print(f"  residuals_trimmed:   {residuals_trim.shape}")

    # ----- 5. Save -----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_path} ...")

    np.savez(
        out_path,
        residuals_untrimmed=residuals,
        residuals_trimmed=residuals_trim,
        prompts=np.array(prompt_strs, dtype=object),
        tasks=np.array(task_strs, dtype=object),
        trim=np.int32(K),
        n_sublayers_full=np.int32(residuals.shape[1]),
        n_sublayers_trimmed=np.int32(residuals_trim.shape[1]),
        d_model=np.int32(d_model),
        model_id=np.array(args.model),
        use_q8=np.bool_(not args.no_q8),
        save_dtype=np.array(args.save_dtype),
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {size_mb:.1f} MB.")
    print(f"\nKeys in the npz:")
    for k in [
        "residuals_untrimmed", "residuals_trimmed",
        "prompts", "tasks", "trim",
        "n_sublayers_full", "n_sublayers_trimmed",
        "d_model", "model_id", "use_q8", "save_dtype",
    ]:
        print(f"  {k}")
    print()
    print("Tip: to pick activations for one unit later,")
    print("    d = np.load('...')")
    print("    unit_2300_trim = d['residuals_trimmed'][:, :, 2300]   "
          "# (N, n_sub_trim)")
    print("    unit_2300_full = d['residuals_untrimmed'][:, :, 2300] "
          "# (N, n_sub_full)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

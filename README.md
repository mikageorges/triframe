# triframe

This package helps to understand how your model uses genomic features.

Genomic prediction models often combine groups of features, such as sequence composition, motif counts, and conservation scores. triframe helps investigate which groups carry predictive information, which the model relies on, and how predictions change when their internal representations are replaced.

It combines model diagnostics, feature/label association analyses, and representation interventions in a common workflow.
`triframe` validates subgroup/token-level interpretability claims at three
progressively stronger levels:

- **Layer 1 (model-diagnostics)** — what a trained model's own internals
  (attention, ablation) directly expose about token reliance.
- **Layer 2 (associational)** — how strongly a token's feature distribution
  differs across groups, independent of any model.
- **Layer 3 (interventional)** — intervene on the model's internal token
  representation and measure the resulting change in behavior.

Association is not causation: a token a model attends to heavily may
carry no real signal, and a token it barely attends to may be exactly what
the model relies on. `triframe` lets you check.

## Motivating case study

`triframe` is a framework originating from a lncRNA/mRNA classifier ([β-LNC], not yet
public) that uses it to test whether each of 20 engineered genomic feature
groups is a real causal driver of the model's predictions, or just
associated with the outcome. The paper is in preparation; this package is
the domain-agnostic core of that work, usable on any token-structured
model.

[β-LNC]: #

## Installation

```bash
pip install triframe
```

For development (editable install + test dependencies):

```bash
git clone <repo-url>
cd triframe
pip install -e ".[dev]"
pytest tests/ -v
```

Requires Python ≥3.9. Core dependencies: numpy, pandas, scipy,
scikit-learn, matplotlib, torch.

## The two things you provide

`triframe` needs exactly two things from you to run any layer: a
**`TokenRegistry`** describing your model's token layout, and a
**`ModelAdapter`** telling `triframe` how to call your model. Everything
else — attention aggregation, ablation, Fréchet distance, activation
patching, IIA, synergy search — is generic and works the same way
regardless of your domain or architecture.

### 1. Describe your tokens: `TokenRegistry`

If your model fuses several named groups of input features into
per-group ("token") representations — the way many multimodal or
feature-engineered architectures do — describe that layout once:

```python
from triframe import TokenRegistry, BlockMeta, FeatureMeta

# A "block" is a group of tokens sharing an input source (e.g. one
# feature modality). A "token" (subgroup) is what triframe patches,
# ablates, and measures — usually built from several raw features.
features = [
    FeatureMeta(name="gc_content",   block="seq_stats", subgroup="composition"),
    FeatureMeta(name="length",       block="seq_stats", subgroup="composition"),
    FeatureMeta(name="motif_count",  block="seq_stats", subgroup="motifs"),
]

registry = TokenRegistry(blocks={
    "seq_stats": (BlockMeta(name="seq_stats", source="single"), features),
})

registry.all_subgroups        # ["composition", "motifs"]
registry.total_tokens         # 2
registry.indices_for_subgroup("composition")   # [0, 1]
```

The registry's only job is answering "which raw feature indices belong to
token X" and "what order do tokens concatenate in" — it doesn't know
anything about your model or domain.

### 2. Tell triframe how to call your model: `ModelAdapter`

Subclass `ModelAdapter` and implement three methods. Your model's
`forward()` signature is untouched — call it however it already expects
to be called.

```python
from triframe import ModelAdapter

class MyAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    def get_token_module(self):
        # The nn.Module whose output IS the token tensor,
        # shape (B, N_tokens, d_proj) — triframe hooks this module
        # to read tokens (Layer 1/2) and to patch them (Layer 3).
        return self.model.token_norm

    def run_forward(self, batch):
        # Called with whatever `batch` triframe was given — its
        # structure (tensor, dict of tensors, ...) is entirely up to you.
        return self.model(**batch)

    def get_output_value(self, model_output):
        # Extract the outcome triframe measures changes in
        # (logits, a probability, a score — your choice).
        return model_output["logits"]

adapter = MyAdapter(my_trained_model)
```

Two more adapter methods are optional, needed only if you use the
matching feature:

```python
def get_attention_weights(self, model_output):
    # (B, ..., N_tokens) — needed for triframe.layer1.aggregate_attention()
    return model_output["attn_weights"]
```

## Layer 1 — model-diagnostics

```python
from triframe.layer1 import aggregate_attention, run_ablation, pattern_alignment, representation_redundancy

# Attention: per-token weight, averaged over samples (and folds, if you
# pass multiple batches).
attn = aggregate_attention(adapter, registry, batches=[batch_fold0, batch_fold1])
attn.aggregated   # {"composition": (mean, std), "motifs": (mean, std)}

# Ablation: accuracy/score drop when a token is zeroed out. You write
# zero_fn — how "zeroing a token" works for your data representation.
# (batch shape here matches a {block_name: tensor} adapter, as in the
# example below — adapt to however your run_forward() expects batches.)
def zero_fn(batch, token_names, registry):
    modified = {k: v.clone() for k, v in batch.items()}
    for tok in token_names:
        block = registry.token_block(tok)
        idx = registry.indices_for_subgroup(tok)
        modified[block][:, idx] = 0.0
    return modified

from triframe.presets import classification_accuracy_fn
result = run_ablation(adapter, registry, batches=[batch], labels=[labels],
                       zero_fn=zero_fn, mode="feature_zero",
                       score_fn=classification_accuracy_fn)

# Pattern alignment (Haufe et al. 2014): does a token's own best
# class-separating direction agree with its covariance-corrected pattern?
# Near 1 = genuine signal; near 0 or negative = suppressor-like.
align_raw = pattern_alignment(token_representation, labels)
align_residual = pattern_alignment(token_representation, labels, controlling_for=z)

# Redundancy: how much of token A's variance does representation B
# already explain? (e.g. is a token just redundant with a shared latent z)
r2 = representation_redundancy(token_a, z)
```

`run_ablation`'s `score_fn` defaults to a task-agnostic identity function
(just averages the output) — it does **not** assume classification.
`triframe.presets` has ready classification versions if that's your task.

## Layer 2 — associational

```python
from triframe.layer2 import frechet_by_token, concentration_by_token

# Wasserstein-2 (Fréchet) distance between two groups' distributions,
# per token, decomposed into mean-shift and covariance terms.
fd = frechet_by_token(registry, block_data={"seq_stats": X}, labels=y)
fd[["token", "frechet_dist", "mean_term", "cov_term"]]

# Is a token's signal concentrated in a few features, or diffuse?
conc = concentration_by_token(registry, block_data={"seq_stats": X}, labels=y)
conc[["token", "gini", "participation_ratio", "n_features_for_threshold"]]
```

`group_a`/`group_b` (default `0`/`1`) select which label values to
compare — works for any two groups, not just binary classification
labels.

## Layer 3 — interventional

```python
from triframe.layer3 import (
    patch_and_forward, symmetry_score, interchange_intervention_accuracy,
    greedy_token_search, shuffled_null_control,
)

# One interchange intervention: substitute source's token(s) into
# target's forward pass.
baseline_out, patched_out = patch_and_forward(
    adapter, target_batch, source_batch, token_positions=[0])

# Effect size: how much does patching move the output, regardless of
# whether any decision boundary is crossed?
sym = symmetry_score(adapter, target_batches, source_batches, token_positions=[0])

# Interchange Intervention Accuracy: the stricter test — did patching
# actually flip the outcome the way you'd expect? You define "success".
# success_fn(baseline_output, patched_output, expected) -> bool, called
# once per (target_batch, source_batch, expected) triple — for a batch of
# samples, average success across the batch yourself, as below.
# (classification_success_fn in triframe.presets is the single-sample
# version; write a batch-aware success_fn like this for batched pairs.)
def success_fn(baseline_out, patched_out, expected_batch):
    preds = adapter.get_output_value(patched_out).argmax(dim=-1)
    return bool((preds == expected_batch).float().mean() > 0.5)

iia = interchange_intervention_accuracy(
    adapter, target_batches, source_batches, token_positions=[0],
    expected=expected_batches, success_fn=success_fn)

# Does a COMBINATION of tokens work where no single token does? Greedy
# search grows a token set, testing for synergy against an
# independence-null baseline at every step.
rows = greedy_token_search(
    all_tokens=registry.all_subgroups, registry=registry,
    single_token_iia=single_iia, iia_fn=my_iia_fn, max_set_size=6)
# each row: {step, added_token, iia_joint, iia_additive_expected, delta_vs_additive}
```

`token_positions` are **global token-dimension indices** (0-indexed into
`registry.all_subgroups`'s order), not per-block feature indices — use
`registry.all_subgroups.index(token_name)` to get them.

For matched (source, target) pairs, `triframe.pairing.nearest_neighbor`
generalizes "find the closest sample in another group by some similarity
embedding":

```python
from triframe.pairing import nearest_neighbor

embedding = np.column_stack([lengths, gc_content])  # any similarity space
pairs = nearest_neighbor(embedding, group_a_mask, group_b_mask, max_dist=0.2)
```

## Plotting

Generic, block-colored chart primitives — no domain-specific styling:

```python
from triframe.plotting import ranked_bar, stacked_bar, sufficiency_curve

fig, ax = ranked_bar(labels=fd["token"], values=fd["frechet_dist"],
                      blocks=fd["block"], ylabel="Fréchet distance", log_scale=True)
```

## Full working example

`triframe/examples/beta_vae_subgroup_minimal/` is a complete, runnable
reference: a small model with 3 blocks / 9 tokens, synthetic data with
known ground truth (a strong signal, a covariance-only signal, a
z-redundant token, an XOR-style synergy pair, and pure noise), exercising
every layer end to end. It's also the fixture the test suite (`tests/`)
validates against — the same ground truth described there is asserted in
code.

```python
from triframe.examples.beta_vae_subgroup_minimal.synthetic_data import build_registry, generate
from triframe.examples.beta_vae_subgroup_minimal.model import build_model, MinimalAdapter, to_block_inputs

registry = build_registry()
data = generate(n=2000, seed=0)
model = build_model(registry, d_proj=8, seed=0)   # untrained, fast, deterministic
adapter = MinimalAdapter(model)
```

`build_model` gives you an untrained model by default (sufficient for
Layer 1/2, and for testing Layer 3's mechanics). Layer 3's synergy
detection needs a model that's actually learned the relevant structure —
see `train()`/`save_checkpoint()`/`load_checkpoint()` in the same module.

The implementation is domain-independent: other applications can supply their own feature groups and model adapter.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite validates every layer against known ground
truth on the minimal example above."

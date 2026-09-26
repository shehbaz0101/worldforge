# Architecture

WorldForge is a latent world model for a scientific dynamics sandbox. The
target pipeline learns a compact state from trajectories, rolls it forward,
plans with that model, and scores prediction and control offline. The
environment, the trajectory record, an offline dataset builder, a latent
model, and a one-step training loop are in the tree. The planner, eval
harness, and HTTP API are later days.

## Components

```mermaid
flowchart LR
  env[Env] --> traj[Trajectory]
  traj --> dataset[Dataset]
  dataset --> train[Train]
  train --> encoder[Encoder]
  encoder --> dynamics[Dynamics]
  dynamics --> decoder[Decoder]
  decoder --> predict[One-step predict]
  dynamics --> planner[Planner]
  planner --> control[Control eval]
```

| Component | Shipped | Later |
| --- | --- | --- |
| Env | Shipped. Default `lotka_volterra`. | More environments can register behind the same reset/step protocol. |
| Dataset | Shipped. Corpus JSONL, per-episode JSON, manifest, train/val/test split, replay batch. | The trainer samples that replay batch. Holding out val and test is up to the caller. |
| Encoder | Shipped. Deterministic MLP, observation to latent `z`. | A stochastic encoder would belong to the reserved RSSM variant. |
| Dynamics | Shipped. Day 3 baseline: deterministic residual MLP, `(z, a) -> z_next`. | `dynamics="rssm"` is reserved and raises `NotImplementedError`. Multi-step rollout loss is later. |
| Decoder | Shipped. Small MLP, latent to reconstructed observation. | Training scores it with a weighted reconstruction MSE. |
| World model | Shipped. `encode`, `step`, `decode`, `predict`, and `forward`. Checkpoint directory with `config.json` and `weights.pt`. | Day 4 trains these weights. The checkpoint format is unchanged. |
| Train | Shipped. One-step MSE, Adam or SGD, JSONL epoch log, `worldforge train`. | Multi-step rollout loss is later. No planner. |
| Planner | Not started. | Model-based action search. |
| Eval | Not started. | Open-loop prediction error and offline control return. |

Collection and the unit tests stay offline. They do not download a dataset or a
pretrained weight file. Model and train tests need the optional `ml` extra
(`torch` and `numpy`). CI installs a CPU build of torch before that extra.
There is no FastAPI app and no paid API. There is no planner and no
multi-step rollout loss in this revision.

## Default environment

The default system is nondimensional Lotka-Volterra predator-prey kinetics
with a clipped dose on each species. The state is two positive numbers, so
a rollout is fast and a JSON trajectory stays small. With unit rates the
coexistence equilibrium is `(1, 1)`. The uncontrolled vector field is a
family of cycles around that point, which is the structure the latent
model is meant to fit. The dose is the control: positive stocks a species,
negative harvests it. The reward is the negative L1 distance from the
equilibrium. That is a placeholder regulation objective, not a learned
reward.

A Gray-Scott reaction-diffusion plate was the other candidate. It is not
the default. A plate, even a small one, has an observation made of a grid
channel for each species. Day 1 tests need a state small enough that a
seeded episode finishes well under a second. The schema stores a list of
floats, so a grid environment can be added later without a new file format.

Integration is classical RK4. One environment step is `0.2` time units,
split into four substeps. The dose is held constant across those substeps.
`reset(seed)` draws the initial population with `random.Random(seed)`. The
same seed and the same actions reproduce the same states.

`worldforge env-demo` is the smoke path. It rolls this environment and
prints the trajectory metadata (`env_id`, `seed`, `horizon`), the return,
the initial and final state, and a one-line ASCII summary.

## Trajectory record

`Observation`, `Action`, `Transition`, and `Trajectory` are frozen Pydantic
models. Extra fields are rejected. A trajectory's metadata is `env_id`,
`seed`, and `horizon`. `horizon` is the number of steps requested.
`transitions` is shorter when the episode ends first.

JSON is one document. Single-episode JSONL is one trajectory file: a `meta`
line, then one `transition` line per step. `rollout` builds a trajectory
with a zero dose, a seeded uniform dose, or an open-loop sine dose.
`sine_dose` scales a sine of the step index into the action bounds. Component
`i` has phase `i * pi / 2`. The period is 16 steps. The dose does not use
the seed.

## Corpus

`worldforge collect` writes a directory:

| File | Contents |
| --- | --- |
| `manifest.json` | Format `worldforge.corpus.v1`. `env_id`, requested `horizon`, and one record per episode: `index`, `seed`, `policy`, `steps`, and an optional relative `file`. Top-level `policy` is `mixed` when episodes differ. |
| `corpus.jsonl` | One trajectory JSON object per line (`env_id`, `seed`, `horizon`, `transitions`). Present when the jsonl format is requested. |
| `episodes/ep_XXXX.json` | One indented trajectory document per episode. Present when the json format is requested. |

`load_corpus` reads the directory (preferring `corpus.jsonl`), a corpus JSONL
file, a single-episode JSONL file, or one trajectory JSON file.
`split_trajectories` assigns episodes with the largest-remainder method and
shuffles a canonical order with `random.Random(seed)`. The same episodes and
the same seed produce the same train, val, and test tuples.
`sample_transition_batch` draws stored transitions without replacement. It
does not step the environment.

The golden set is `tests/fixtures/trajectories/`: six episodes, horizon 8,
seeds 0–5, policies `zero`, `random`, and `sine_dose`. Regenerate with
`python scripts/regenerate_fixtures.py` and commit the files. CI loads them
and does not rebuild them.

## Latent model

`src/worldforge/models/` is the Day 3 forward path. It does not step the
environment and it does not update parameters.

`ModelConfig` is a frozen Pydantic model and does not import PyTorch.
Defaults match Lotka-Volterra: `obs_dim=2`, `action_dim=2`, `latent_dim=8`,
`hidden_dim=64`. `encoder_hidden_layers` defaults to 2, `dynamics_hidden_layers`
to 2, and `decoder_hidden_layers` to 1. Those depths are stored in the
checkpoint so a later load rebuilds the same MLPs.

`WorldModel(config, seed=...)` builds three modules on CPU in float32, in
order:

| Module | Map | Default shape |
| --- | --- | --- |
| Encoder | observation to `z` | `Linear(2, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 8)` |
| Dynamics | `(z, action)` to `z_next` | residual `Linear(10, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 8)` |
| Decoder | `z` to observation | `Linear(8, 64) -> ReLU -> Linear(64, 2)` |

The dynamics baseline is

```text
z_{t+1} = z_t + mlp(concat(z_t, a_t))
```

The same latent and the same action always produce the same next latent.
There is no dropout, so `train()` and `eval()` match. `seed` draws every
linear map from a private generator. Construction restores PyTorch's global
CPU generator.

`forward(observation, action)` returns a `StepPrediction`: `latent` (`z_t`),
`next_latent` (`z_{t+1}`), `reconstructed` (`decode(z_t)`), and
`predicted_observation` (`decode(z_{t+1})`). `predict` returns only the
one-step observation. Leading tensor dimensions are a batch. The last
dimension is the feature width. Inputs are floating tensors; the modules
cast them to float32.

`build_dynamics` is the extension point. `ModelConfig.dynamics` may be
`mlp` or `rssm`. `mlp` is this baseline. `rssm` is the name reserved for a
later stochastic recurrent state-space model and raises
`NotImplementedError` at construction. A future variant should keep
`forward(latent, action) -> next_latent` so `WorldModel.step` stays the
training call site.

`save_checkpoint(path, model)` writes a directory:

| File | Contents |
| --- | --- |
| `config.json` | Format `worldforge.checkpoint.v1`, the package version, and the `ModelConfig` JSON. No optimizer state and no credentials. |
| `weights.pt` | The module `state_dict` only. |

`load_checkpoint` rebuilds the modules from that config and loads the
weights onto CPU. The init seed is not stored.

`worldforge model-info` prints the shapes and parameter counts.
`worldforge forward-smoke` runs one encode / step / decode. With no
`--fixture` it uses the coexistence state `(1, 1)` and a zero dose. With
`--fixture` it reads one stored transition from a corpus or a trajectory
file. Neither command trains. `worldforge train` does.

Importing `worldforge` or `worldforge.models.config` does not import
PyTorch. `from worldforge.models import WorldModel` does, and needs the
`ml` extra.

## Training

`src/worldforge/train/` updates the Day 3 model on stored transitions. It
does not step the environment and it does not roll the latent state past
one step.

The objective is

```text
loss = MSE(predicted_observation, next_observation)
     + reconstruction_weight * MSE(reconstructed, observation)
```

`predicted_observation` is `decode(step(encode(observation), action))`.
`reconstructed` is `decode(encode(observation))`. The default
reconstruction weight is 1. A weight of 0 leaves that term out of the
backward pass. The metrics log still records the unweighted reconstruction
MSE.

Each optimizer step calls `sample_transition_batch`. The batch seed is
`seed + epoch * 1000003 + step`, with `epoch` and `step` starting at 0.
The default steps per epoch are `ceil(transitions / batch_size)`, so one
epoch is one coverage pass over the pool. Sampling is without replacement
inside a batch and independent across steps, so a pass can repeat a
transition. `batch_size` cannot exceed the pool.

The optimizer is Adam or SGD. The device is CPU. For the length of the run
the intra-op thread count is 1 and MKLDNN is turned off when it is
available. Both are restored afterward, along with PyTorch's CPU generator
state. The same `WorldModel` seed and the same `TrainConfig` seed reproduce
the same weights in one process. `TrainConfig.seed` selects batches. The
model constructor seed selects the initial weights. `worldforge train` passes
its `--seed` to both.

`train_world_model` writes the checkpoint directory:

| File | Contents |
| --- | --- |
| `config.json` | Day 3 format `worldforge.checkpoint.v1`. No optimizer state. |
| `weights.pt` | Trained `state_dict`. `load_checkpoint` reads it and ignores the files below. |
| `metrics.jsonl` | One JSON object per epoch: `epoch`, `loss`, `prediction_loss`, `reconstruction_loss`, `steps`. |
| `train.json` | Format `worldforge.train.v1`. Hyperparameters, counts, and `final_train_loss`. |

`final_train_loss` is the last epoch's mean step loss. A step loss is the
batch-mean objective before that step's update. `worldforge train` prints
the epoch lines and `final_train_loss`.

`--data` may be a corpus directory, `corpus.jsonl`, or one trajectory file.
The command fits every loaded episode. To train on a split, call
`split_trajectories` and pass `parts.train` to `train_world_model`. The CLI
does not take a split flag. Defaults are 3 epochs, batch size 8, learning
rate `1e-3`, seed 0, device `cpu`. On the golden fixtures that is 6 batches
per epoch and finishes in a few seconds.

## Where later days attach

Day 5 can add a multi-step loss on rolled-out latents. The planner still
proposes `Action` sequences. Prediction eval still compares a rolled-out
trajectory with a held-out split. Control eval still compares return against
the zero dose already used by `env-demo`. There is still no HTTP API.

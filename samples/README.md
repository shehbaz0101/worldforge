# Demo samples

Tiny offline corpus and the caps for `worldforge demo`. CI loads these files.
It does not download them. The demo trains on `trajectories/` when that
directory is inside the data root. Otherwise it copies this directory under
the demo output, or collects the same episodes if the files are missing.

`demo.json` is format `worldforge.demo.v1`. It records the corpus name, the
episode list, and the short training and planning caps (one epoch, one replay
step, a latent width of 4, a planning horizon of 2). `worldforge demo` reads
it. A longer fit stays on `worldforge train`.

## Trajectories

`trajectories/` is format `worldforge.corpus.v1`. Two episodes, horizon 4,
on `lotka_volterra`: seed 0 with `zero`, seed 1 with `random`. `corpus.jsonl`
is one trajectory JSON object per line. `episodes/ep_XXXX.json` is one
trajectory document per episode. `manifest.json` records the seeds and
policies.

Regenerate after a change to collection, the default environment, or
`demo.json`, then commit the result:

```bash
python scripts/regenerate_samples.py
```

That script needs an installed WorldForge (`pip install -e .`). It rolls the
episodes listed in `demo.json` and overwrites `samples/trajectories/`.

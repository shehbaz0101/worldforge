# Trajectory fixtures

Checked-in golden corpus for offline tests. CI loads these files. It does not
roll the environment to recreate them and it does not use the network.

Regenerate after a change to collection or to the default environment, then
commit the result:

```bash
python scripts/regenerate_fixtures.py
```

That script needs an installed WorldForge (`pip install -e .`). It rolls six
episodes of horizon 8 on `lotka_volterra`: seeds 0 and 1 with `zero`, seeds 2
and 3 with `random`, seeds 4 and 5 with `sine_dose`.

## Layout

`manifest.json` is format `worldforge.corpus.v1`. It records `env_id`, the
requested `horizon`, and one episode entry (`index`, `seed`, `policy`,
`steps`, `file`). `policy` at the top is `mixed` when the episodes do not
share one policy. Paths in `file` are relative to this directory.

`corpus.jsonl` is one trajectory JSON object per line. A line is the same
object as `Trajectory.to_json`: `env_id`, `seed`, `horizon`, `transitions`.
This is a corpus file. A single-episode JSONL file is different: its first
line is `{"record":"meta",...}` and each following line is one transition.
`load_corpus` accepts both.

`episodes/ep_XXXX.json` is one trajectory JSON document per episode, indented,
same schema as `Trajectory.write_json`.

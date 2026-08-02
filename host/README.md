# Host run board (shared)

This folder is the **shared source of truth** for which machine is running (or will run)
which CAPO experiments. Edit **only your own** CSV; pull/rebase before push so other
hosts see an up-to-date board.

## Files

| File | Owner | Purpose |
|------|--------|---------|
| `choi.csv` | host `choi` | Live + planned cells on choi |
| `offrl.csv` | host `offrl` | Live + planned cells on offrl |
| `<alias>.csv` | that host | Same schema; one file per machine |
| `README.md` | anyone (via PR/commit) | These rules |

Add a new host by copying `choi.csv` → `<alias>.csv`, clearing rows that are not yours,
and filling your plan.

Suggested aliases (extend as needed): `choi`, `offrl`, `svcho`, `ext_csv`, `ext_csh`.

## CSV schema (required columns, TSV-friendly CSV)

Comma-separated, header required. Column order:

```text
variant,algo,env_base,dataset,seed,status,config,run_tag,run_dir,started,finished,eta,notes,updated_at
```

| Column | Values / notes |
|--------|----------------|
| `variant` | `capo` or `baseline` (matched ablation). Other tags OK if noted. |
| `algo` | `td3_bc` / `iql` / `cql` |
| `env_base` | `hopper` / `halfcheetah` / `walker2d` |
| `dataset` | `medium` / `expert` / `replay` (= medium-replay) |
| `seed` | integer; paper seeds `{0,1,2}` |
| `status` | `planned` / `queued` / `running` / `done` / `failed` / `blocked` / `cancelled` |
| `config` | path under repo, e.g. `configs/defaults.yaml` |
| `run_tag` | folder tag: `capo` / `baseline` / … |
| `run_dir` | relative path under repo when known; else empty |
| `started` / `finished` | `YYYY-MM-DD HH:MM:SS` local time, or empty |
| `eta` | free text (`~6h`, `2026-08-03 18:00`, `TBD`) |
| `notes` | short; blockers, who owns follow-up |
| `updated_at` | when **this row** was last edited |

## Status meanings

- `planned` — assigned to this host, not launched yet
- `queued` — waiting behind a live job on this host
- `running` — verified live trainer PID / active matrix cell
- `done` — finished with `summary.json` (and expected evals)
- `failed` — needs retry or diagnosis
- `blocked` — cannot proceed (GPU/data/bug); explain in `notes`
- `cancelled` — deliberately dropped; do not resume unless re-planned

Invariant for a host’s active matrix: every assigned cell should appear once.
Do not mark `done` from a checkpoint alone if the run did not finish cleanly.

## Coordination rules

1. **One owner per host file.** Never rewrite another host’s CSV except to fix an
   obvious typo with that owner’s agreement.
2. **Claim before launch.** Before starting a cell, ensure no other host has the same
   `(variant, algo, env_base, dataset, seed)` as `planned`/`queued`/`running`/`done`
   unless you are intentionally re-running (note that in `notes`).
3. **Refresh when status changes.** Update your CSV when you launch, finish, fail,
   change ETA by >30 minutes, or reassign ownership.
4. **Git share path.** Prefer committing only `host/*.csv` (+ this README). Do **not**
   commit `results/` checkpoints. Pull `--rebase` before push.
5. **Refresh + push on every queue change.** When launching a queue, finishing /
   failing a cell, changing ETA by >30 minutes, or reassigning work: update **your**
   `host/<alias>.csv` immediately (`choi` → `choi.csv`, `offrl` → `offrl.csv`), then
   `git pull --rebase` → commit → `git push`. Do not leave the board stale while a
   local matrix is live.
6. **Local queue ≠ board.** `results/queue_status.tsv` is machine-local. The board
   in `host/` is what other hosts read. Keep them roughly in sync (same statuses).
7. **Matched CAPO vs baseline.** Current default matrices (seed 0):
   - **choi / TD3+BC:** `scripts/run_matrix.sh`
     - `capo` → `configs/defaults.yaml`, `run_tag=capo`, `use_capo=true`, `n_critics=4`
     - `baseline` → `configs/baseline_td3bc.yaml`, `run_tag=baseline`, `use_capo=false`, `n_critics=4`
   - **offrl / IQL:** `scripts/run_matrix_iql.sh` (same schedule/hyperparams, `--algorithm iql`)
     - `capo` → `configs/defaults.yaml`, `run_tag=capo`, `use_capo=true`, `n_critics=4`
     - `baseline` → `configs/baseline_iql.yaml`, `run_tag=baseline`, `use_capo=false`, `n_critics=4`
   - 3 envs × 3 datasets × 2 variants = **18 cells** (sequential on one GPU)

## How to update (example)

```bash
# After pull
git pull --rebase

# Edit only your file
$EDITOR host/choi.csv

git add host/choi.csv
git commit -m "host(choi): refresh CAPO vs baseline board"
git push
```

Optional: regenerate rows from the local queue after a matrix finishes:

```bash
# manual sync is fine; keep columns stable for spreadsheet diffs
```

## Conflict checklist

Before claiming work on another machine:

```bash
git pull --rebase
grep -h ',running,\|,queued,\|,planned,' host/*.csv | sort
```

If the same cell appears on two hosts as active, stop and resolve in chat / commit
notes before launching a duplicate 1M-step run.

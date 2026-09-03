# Frozen artifacts

Codebooks, bank weights, teaching pools, generated states and results
tables. Every one of them is config-keyed: the configuration it was
produced under is stored inside the file, and a script that finds a
mismatch refits rather than trusting it. Delete a file to force it to
be rebuilt.

## What is in git

`critic.pt`, and nothing else.

That is deliberate. The critic is a fixed reference rather than a
result: it is the frozen judge every verdict table is scored by, it
never sees a generated image, and `00_train_critic.py --force`
rebuilds it in about three seconds — bit for bit on the same machine,
and to the same metrics elsewhere (see *Determinism* in the top-level
README). Everything else here is an output of a script in
`experiments/`, and shipping an output would let the script that is
supposed to recompute it skip the step instead, which is the opposite
of what this repository is for.

## What is rebuilt

The fitted codebooks and the taught banks. These are what takes hours,
and the bank weights are large. The joint bank alone is 3616 lanes
over 232 spaces of 16 symbols, held as int32 pairs.

| file | what | approximate size |
|---|---|---|
| `backbone-books.pt` | backbone codebooks | 12 MB |
| `backbone-codes.pt` | encoded backbone codes | 17 MB |
| `backbone-banks.pt` | G1 and R1 per code shape | 96 MB |
| `patch-books.pt` | residual patch codebooks | 1 MB |
| `patch-codes.pt` | encoded patch codes | 41 MB |
| `patch-banks.pt` | G2 and R2 | 93 MB |
| `joint-pool.pt` | the joint bank's teaching pool | 16 MB |
| `joint-bank.pt` | the joint repair bank | 107 MB |

```bash
make artifacts   # rebuild from the data, about 90 minutes
```

`09_prefix_schedule.py` adds one more taught bank of its own,
`prefix-schedule-bank.pt` (about 34 MB), under `make analysis`.

`make binary` builds the binary-trained arms of Sec. V C, which are a
second set of the same kind at the 0.1 threshold:

| file | what | approximate size |
|---|---|---|
| `binary-backbone-books.pt` | binary backbone codebooks | 9 MB |
| `binary-backbone-codes.pt` | encoded binary backbone codes | 13 MB |
| `binary-backbone-banks.pt` | G1 and R1 at 128 and 64 books | 85 MB |
| `binary-patch-books.pt` | binary residual patch codebooks | 104 KB |
| `binary-patch-codes.pt` | encoded binary patch codes | 6.5 MB |
| `binary-patch-banks.pt` | binary G2 and R2 | 89 MB |
| `binary-joint-bank.pt` | the binary joint repair bank | 102 MB |

Everything downstream is written here too, by the rest of
`experiments/`: generated states, judge rows, the results tables of the
sweeps and the energy model, every figure. Once the files above are in
place each of those steps takes minutes.

A rebuild on a different machine will not produce byte-identical
codebooks, for the reasons under *Determinism* in the top-level README.
The acceptance test for a rebuild is `NUMBERS.md`, not a checksum.

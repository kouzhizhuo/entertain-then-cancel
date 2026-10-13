# Data

`new_items/` holds the items created for this paper, released under CC BY 4.0:
- the speaker-knowledge scenarios (`a1ctx_*`);
- the Bott–Noveck category sentences (`bn_underinformative*`);
- the context-reversal pairs (`method_ctxrev`).

All other item sets are built from their original public releases by `code/data_builders/build_items.py`, which downloads each source and writes `data/items/*.jsonl`. They are not redistributed here; use them under their own licences:

| Source | Licence |
|---|---|
| IMPPRES | CC BY-NC 4.0 |
| PUB, Fig-QA, ImplicatureX | MIT |
| LUDWIG, Circa | CC BY-SA 4.0 (Circa) / original terms (LUDWIG) |
| PragMega (Hu et al. 2023) | original terms |
| scalar-diversity sets | as released by the authors |
| NOPE | no licence stated: build locally, do not redistribute |
| FOLIO, GSM8K | CC BY-SA 4.0 / MIT |

See `SOURCES_PROVENANCE.md` for exact versions.

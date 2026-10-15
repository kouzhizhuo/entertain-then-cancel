# Entertain, then Cancel: How Thinking Makes Reasoning Models Read  *Some* and *Or* Literally

Code, prompts and result files for an anonymous ACL Rolling Review submission.

**Summary.** We hold a reasoning model's weights fixed and vary only how much it thinks (thinking mode, token budget or reasoning effort), then record which reading it chooses for scalar implicatures such as *some* → *not all*. Across open reasoning models, thinking cancels the implicatures of *some* and *or*, while context-driven inferences improve. A trace-prefix diagnostic forces an answer after successive prefixes of the model's own reasoning; it shows the enriched reading is entertained early and cancelled late. An exploratory inference-time remedy answers scalar items from that early reading.

## Repository map
- `code/harness/`: vLLM/HF runners, prompt rendering and answer readout.
- `code/analysis/`: scoring, statistics (bootstrap CIs, Holm correction, TOST), counts, remedy scorers, citation and anonymity tools.
- `code/configs/`: experiment configurations, one per experiment block.
- `code/data_builders/`: scripts that download and build every item set.
- `code/paper_tools/`: `json2tex.py` (result JSON → `\res` macros) and the table and figure generators.
- `results/`: the small summary JSON files that every number in the paper comes from (`*/summary.json`, `DERIVED/`, `STATS/`, `M1/`, `MANIFEST/runs.json`, `PROMPTS/verbatim.json`).
- `results/human_audit/`: blind audit labels. Each is a consensus label from a panel of five or more annotators with PhD-level training in related fields, agreed through discussion and recorded with a software tool. No inter-annotator agreement is reported, and annotator identity is removed.
- `data/`: new items created for this paper, plus build instructions for public sources.

## Install
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
Model runs need a CUDA GPU and vLLM. Models are the public Hugging Face checkpoints listed in `results/MANIFEST/runs.json`. API-served models are queried through an OpenAI-compatible endpoint set by `OPENAI_BASE_URL` / `OPENAI_API_KEY` (not included).

## Reproduce tables and figures
| Paper item | Script | Input |
|---|---|---|
| Table 1 (main results) | `code/paper_tools/make_tab_results.py` | `results/E1*/summary.json`, `results/E3`, `results/E1F` |
| Sensitivity table | `code/paper_tools/make_tab_sens.py` | `results/E1`, `results/E1S`, `results/E1SF` |
| Exploratory remedy table | `code/analysis/x3_test.py` | `results/M1/x3_test.json` |
| Model-count and statistics | `code/analysis/derive_counts.py` | `results/STATS`, `results/E1*` → `results/DERIVED` |
| Figures 2 and 4 (strip, forest) | `code/paper_tools/make_fig_main.py` | `results/DERIVED/models_lists.json` + summaries |
| Size-by-family figure | `code/paper_tools/make_fig_size.py` | `results/E1*` |
| Run manifest, prompts appendix | `code/paper_tools/make_app_v2.py` | `results/MANIFEST`, `results/PROMPTS` |
| All in-text numbers | `code/paper_tools/json2tex.py` | every `results/*/summary.json` |

Raw generations (several GB) are not included. Rerun the harness with a config from `code/configs/` to regenerate them.

## Citation
Anonymous submission under review.

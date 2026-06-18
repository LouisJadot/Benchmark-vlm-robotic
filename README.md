# VLM Robotics Benchmark — From Single-Object Grasping to High-Level Task Planning

> A progressive, multi-test benchmark evaluating Vision-Language Models (VLMs) for robot manipulation tasks.  
> **Author:** Louis Jadot — University of Waikato  
> **Evaluators:** GPT-5.2 · Claude Sonnet 4.6 (dual LLM-as-judge validation)

---

## Overview

This benchmark assesses five state-of-the-art VLMs on their ability to produce structured, physically grounded JSON outputs for robot manipulation. Tasks range from identifying a single object to autonomously decomposing a high-level cleaning goal into an executable manipulation plan.

## Paper

This repository accompanies the paper:

> Louis Jadot Millet, ..., *VLM Robotics Benchmark — From Single-Object Grasping to High-Level Task Planning*, 2026.


**Models evaluated:**

| Model | Type | Parameters |
|---|---|---|
| GPT-5.4 Mini | Proprietary | — |
| Gemini 3.1 Flash Image | Proprietary | — |
| Seed 2.0 Lite | Proprietary | — |
| Qwen 3.5 9B | Open-weight | 9B |
| Ministral 8B | Open-weight | 8B |

---

## Benchmark Structure

The benchmark is composed of **5 progressive tests**, each increasing task complexity along two axes: *perceptual difficulty* (Tests 1–3) and *planning depth* (Tests 4–5).

### Test 1 — Single-Object Grasping
- **Input:** Single image of an isolated object
- **Output:** JSON with object identification, grasp strategy, manipulation plan, use-case reasoning, robustness
- **Objects:** 8 everyday items (blue pen, grey mug, computer mouse, orange balloon, blue can, USB drive, whiteboard eraser, stirrer)
- **Scoring:** Scoring System — 5 criteria (Identification, Planning, Use, Grasp, Robustness), scale 0–5
- **Evaluators:** GPT-5.2 + Claude Sonnet 4.6

### Test 2 — Multi-Object Scene Grasping
- **Input:** Scene with multiple objects + textual target hint
- **Output:** Same JSON + `position_in_scene` field (localization)
- **Key difference:** Identification criterion now evaluates localization, not name recall
- **Scoring:** Same protocol, position score replaces name score

### Test 3 — Strict Localization Grasping
- **Input:** Cluttered scene with partial occlusions + target hint
- **Output:** Same JSON with rich spatial description (orientation, neighbors, position)
- **Key difference:** Position scored +1 per valid spatial descriptor (max 3); wrong object → score 0
- **Scoring:** Same protocol, stricter localization rubric

### Test 4 — Multi-Step Execution Planning
- **Input:** Real-world domestic scene + ordered list of 5 manipulation tasks
- **Output:** Full HTN-style execution plan (scene understanding + step-by-step plan with grasp, action type, preconditions, state transitions)
- **Scenes:** 9 domestic environments (office, kitchen, bathroom, laundry room, sink, storage, etc.)
- **Runs:** 3 independent runs per (model, scene) pair
- **Scoring:** 7 criteria (OV, GD, PPC, AE, ST, SU, PC), scale 0–1

### Test 5 — High-Level Task Planning (HTN)
- **Input:** Real-world scene + single high-level goal (e.g., *"tidy the desk"*)
- **Output:** HTN decomposition (Phase 1: atomic subtask list) + grounded execution plan (Phase 2)
- **Task ambiguity:** 5 variants per scene from highly detailed (level 1) to maximally vague (level 5)
- **Scenes:** 9 domestic environments × 5 ambiguity levels = 45 scene-task combinations
- **Runs:** 3 independent runs per (model, scene, variant)
- **Scoring:** 4 criteria (SU, TD, AP, GC), scale 0–1
- **Forbidden abstract verbs:** `clean`, `tidy`, `organize`, `arrange`, `sort`, `prepare`, `fix`, `manage`, `improve`

---

## Key Results

### Overall Leaderboard (normalized to [0,1] across all tests)

| Model | T1 | T2 | T3 | T4 | T5 | **Mean** |
|---|---|---|---|---|---|---|
| GPT-5.4 Mini | 0.974 | 0.960 | 0.947 | 0.745 | 0.788 | **0.883** |
| Gemini Flash Image | 0.926 | 0.914 | 0.865 | 0.751 | 0.785 | 0.848 |
| Seed 2.0 Lite | 0.910 | 0.932 | 0.921 | 0.722 | 0.726 | 0.842 |
| Ministral 8B | 0.800 | 0.768 | 0.666 | 0.558 | 0.578 | 0.674 |
| Qwen 3.5 9B | 0.862 | 0.704 | 0.787 | 0.425 | 0.464 | 0.648 |

### Key Findings

- **GPT-5.4 Mini** leads the overall benchmark with the most consistent cross-test performance.
- **Gemini Flash Image** is the best model for planning tasks (Tests 4–5), overtaking GPT-5.4 Mini in sequential and high-level task settings.
- **Seed 2.0 Lite** is the most stable model — never placing below 3rd across all five tests.
- **Spatial localization** (Tests 2–3) and **scene understanding** (Tests 4–5) are the two hardest capability axes, consistently separating proprietary from open-weight models.
- **Qwen 3.5 9B** exhibits catastrophic run variance (σ=0.386–0.391 in Tests 4–5) and complete plan failures, disqualifying it for safety-critical deployments.
- **Task ambiguity paradox (Test 5):** performance *increases* with vague task descriptions (level 1 μ=0.652 → level 5 μ=0.714), revealing a coverage bias in LLM-judge evaluation.
- **Inter-evaluator agreement** improves with task structure: Pearson r=0.50 (Test 1) → r=0.94 (Test 2) → r=0.93 (Test 5).

---

## Repository Structure

```text
benchmark/
├── benchmarkvlm/
│   ├── data/
│   │   ├── datatest1,2,3/
│   │   └── datatest4,5/
│   ├── gen/
│   │   ├── test1gen.py
│   │   ├── test2gen.py
│   │   ├── test3gen.py
│   │   ├── test4gen.py
│   │   └── test5gen.py
│   ├── eval/
│   │   ├── test1eval.py
│   │   ├── test2eval.py
│   │   ├── test3eval.py
│   │   ├── test4eval.py
│   │   └── test5eval.py
│   ├── resultgen/
│   ├── resulteval/
│   │   ├── chat/
│   │   └── claude/
│   └── usefull/
│       ├── ingopenrouteur.py
│       └── listmodelopenrouteur.py
├── requirements.txt
├── README.md
├── VLM Robotics Benchmark — From Single-Object Grasping to High-Level Task Planning.txt
└── .gitignore

```

---

## Scoring Protocols

### Tests 1–3 — Robot-Centric Scoring V5

Each criterion is scored 0–5. Final score = mean(A, B, C, D, E).

| Criterion | Description |
|---|---|
| **A — Identification** | Name/localization match + attribute bonus |
| **B — Planning** | Approach, pre-grasp, grasp contact, lift (3 pts) + logical order, efficiency (2 pts) |
| **C — Use** | Realism, grasp-use link, physics plausibility |
| **D — Grasp** | Type correctness, contact zone precision, accessibility |
| **E — Robustness** | Hallucination detection + logic (raw ×2.5 rescaled to 0–5) |

### Test 4 — Step-Level Execution Scoring

Scored 0–1 per criterion. Final = mean(OV, GD, PPC, AE, ST, SU, PC).

| Criterion | Description |
|---|---|
| **OV** | Object validity (visibility, class, attributes) |
| **GD** | Grasp definition (type, contact zone, accessibility) |
| **PPC** | Physical properties (stability, fragility, manipulability) |
| **AE** | Action execution (task match, physical possibility, target) |
| **ST** | State transition (precondition, postcondition, consistency) |
| **SU** | Scene understanding (objects, no hallucination, layout) |
| **PC** | Plan consistency (no switch, no contradiction, temporal order) |

### Test 5 — HTN Planning Scoring

Scored 0/0.5/1 per sub-criterion. Final = mean(SU, TD, AP, GC).

| Criterion | Description |
|---|---|
| **SU** | Scene understanding |
| **TD** | Task decomposition (atomicity, grounding, coverage) |
| **AP** | Action plan (11 sub-criteria per step) |
| **GC** | Global consistency (temporal order, state tracking, task achievement) |

---

## Evaluation Setup

All VLM responses are independently evaluated by two LLM judges:
- **GPT-5.2** (OpenAI)
- **Claude Sonnet 4.6** (Anthropic)

All reported scores are means over both evaluators. Inter-evaluator agreement is reported via Pearson *r* and Mean Absolute Error (MAE).

VLMs are accessed via the [OpenRouter](https://openrouter.ai/) API. Set your API key before running:

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Requirements

```bash
pip install annotated-types==0.7.0
anyio==4.13.0
certifi==2026.2.25
defusedxml==0.7.1
distro==1.9.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.11
jiter==0.13.0
numpy==2.4.3
odfpy==1.4.1
openai==2.29.0
openrouter==0.8.0
pandas==3.0.1
pydantic==2.12.5
pydantic_core==2.41.5
python-dateutil==2.9.0.post0
six==1.17.0
sniffio==1.3.1
tqdm==4.67.3
typing-inspection==0.4.2
typing_extensions==4.15.0

```

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{jadot2025vlmbenchmark,
  title   = {Benchmarking Vision-Language Models for Robot Perception and Grasp Planning},
  author  = {Jadot Louis, Anany Dwivedi},
  year    = {2026},
  school  = {University of Waikato},
  note    = {Available at: https://github.com/LouisJadot/vlm-robotics-benchmark}
}
```

---

## License

MIT License — see `LICENSE` for details.
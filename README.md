# Viva AI

[![Viva AI — adaptive oral exam coach](assets/viva-ai-devpost-thumbnail.png)](https://youtu.be/pNTsDsS3mn8)

Viva AI is an adaptive oral exam coach grounded in the learner's own study
materials. It was built for the **Prometheus July AI Challenge 2026**.

## Video demo

Watch the [1:48 hackathon demo on YouTube](https://youtu.be/pNTsDsS3mn8).

Upload a PDF, TXT, or Markdown document, answer questions in your own words,
and receive:

- semantic assessment grounded in the source material;
- criterion-by-criterion feedback instead of an unexplained score;
- focused follow-up questions for genuine knowledge gaps;
- new-topic questions when the current concept is understood;
- a topic mastery map and personalized revision plan;
- a downloadable Markdown report.

Questions and feedback follow the selected interface language, while source
quotations remain in their original language.

## Why AI is essential

A conventional quiz compares an answer with a fixed string or a list of
keywords. Viva AI evaluates free-form explanations, recognizes equivalent
wording, identifies missing ideas, and changes the exam trajectory after every
answer.

The model is constrained by retrieved passages from the uploaded material.
Pydantic validates every structured response, while deterministic Python code
calculates the numeric score.

## Transparent scoring

The language model never assigns the final score directly. It classifies each
expected point as:

- `met` — fully and correctly covered: 1 point;
- `partial` — present but incomplete: 0.5 points;
- `missing` — absent or incorrect: 0 points.

Python calculates:

$$
\text{score} =
\frac{\sum_i w_i}{N} \times 100,
\qquad
w_i \in \{1,\ 0.5,\ 0\}.
$$

The same combination of criterion statuses therefore always produces the same
score. Learners can inspect the status, explanation, and answer evidence for
every criterion.

## Product experience

- Quick 3-question or full 5-question sessions.
- Basic, standard, and advanced difficulty levels.
- Instant first question in the built-in demo.
- Source context displayed next to every question.
- Retry, skip, and finish-early controls.
- Adaptive follow-ups or deliberate topic changes.
- Final topic mastery dashboard.
- Downloadable session report.
- Responsive interface.
- Local-first inference through Ollama.

## Quick start

Requirements:

- Python 3.10 or newer;
- [Ollama](https://ollama.com/);
- approximately 10 GB of free memory for the default `qwen3:14b` model.

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the default local model:

```powershell
ollama pull qwen3:14b
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Viva AI automatically detects available Ollama models. Select `qwen3:14b` and
the built-in machine learning notes for the fastest demo path.

An OpenAI-compatible cloud endpoint is available as an optional fallback. Its
URL, model name, and API key can be configured in the advanced sidebar settings
or through a local `.env` file.

## Recommended demo flow

1. Open Viva AI and keep the built-in study material selected.
2. Choose a quick session and standard difficulty.
3. Start the exam; the first question appears immediately.
4. Give an incomplete answer.
5. Show the deterministic score, criterion breakdown, and source evidence.
6. Answer the focused follow-up or move to a new topic.
7. Finish the session and show the mastery map and revision plan.

## Architecture

```text
PDF / TXT / MD
      |
      v
text extraction and normalization
      |
      v
passage splitting + lightweight lexical retrieval
      |
      v
LLM: question -> criterion assessment -> next question
      |
      v
Pydantic validation -> deterministic scoring -> Streamlit UI
      |
      v
mastery map + revision plan + Markdown report
```

Key files:

- `app.py` — Streamlit interface and exam session state;
- `viva_core.py` — document processing, retrieval, LLM client, schemas, and
  deterministic scoring;
- `ui_text.py` — interface copy;
- `demo_materials/` — safe built-in study material;
- `scripts/evaluate_grading.py` — real-model grading regression;
- `tests/` — deterministic tests that do not require an API call.

## Validation

Run the fast local checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile app.py viva_core.py ui_text.py
```

Run the slower golden evaluation against the configured local model:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_grading.py
```

The golden evaluation checks that scores for incorrect, vague, partial, and
complete answers are non-decreasing, that a partial answer falls within the
expected range, and that repeated complete answers both receive 100.

## Privacy and security

In local mode, Ollama processes the study material on the learner's computer.
In cloud mode, the material is sent to the configured provider, so private data
should only be uploaded after reviewing that provider's terms.

The application does not persist API keys. `.env`, virtual environments,
private Streamlit secrets, local tools, caches, and generated submission
archives are excluded through `.gitignore`.

## MVP limitations

- Uploaded PDFs must contain a text layer; OCR is not implemented yet.
- Local and cloud modes require a compatible chat-completions API.
- Very large documents use a representative subset of passages.
- Question and assessment quality still depends on the selected model.
- Important educational conclusions should be independently verified.

## Hackathon

Viva AI targets a specific educational problem: rereading notes creates an
illusion of mastery, while oral explanation exposes whether the learner can
actually reconstruct and apply a concept.

Judging categories total 100 points:

- Educational Impact — 25;
- Creative Use of AI/ML — 25;
- Technical Execution — 25;
- Pitch & Demo — 25.

Submission requirements and the final checklist are documented in
[REQUIREMENTS.md](REQUIREMENTS.md).

### Key dates

| Stage | EDT |
|---|---:|
| Project submission opened | July 17, 2026, 10:00 |
| Safe submission deadline | **July 30, 2026, 23:45** |
| Voting and judging | August 1–8, 2026 |
| Winners announced | August 9, 2026, 17:00 |

The Rules page mentions 23:59, while the official schedule and countdown show
23:45 EDT. The project uses the earlier time as the safe deadline.

### Submission status

- [x] Define a narrow audience and educational problem.
- [x] Build the end-to-end prototype.
- [x] Add deterministic scoring and regression tests.
- [x] Prepare a public repository and setup instructions.
- [x] Prepare the Devpost story and submission archive.
- [x] Join the hackathon on Devpost.
- [x] Complete the organizers' registration form.
- [x] Record and publish a video under 1:55.
- [x] Submit the project before the deadline.

## Official links

- [Hackathon overview](https://prometheus-july-ai-challenge.devpost.com/)
- [Rules](https://prometheus-july-ai-challenge.devpost.com/rules)
- [Schedule](https://prometheus-july-ai-challenge.devpost.com/details/dates)
- [Resources](https://prometheus-july-ai-challenge.devpost.com/resources)
- [Organizer registration form](https://forms.gle/DL7ye47iHE84rHQ2A)

Organizer contact: `prometheuscsinternational@gmail.com`.

Hackathon information was last verified on July 28, 2026. Recheck the Rules,
Schedule, and Updates pages before submitting.

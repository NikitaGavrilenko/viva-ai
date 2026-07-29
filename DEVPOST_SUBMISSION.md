# Devpost Submission — Viva AI

## Project name

Viva AI — Adaptive Oral Exam Coach

## Elevator pitch

Turn any study material into a grounded, adaptive oral exam with transparent
criterion-based feedback and a personalized revision plan — powered locally
for privacy.

## About the project

### Inspiration

Students often prepare for exams by rereading the same notes and highlighting
the same paragraphs. This feels productive, but it rarely tests whether they
can explain a concept without looking at the source.

We wanted to create a study experience closer to a real oral exam: a teacher
asks a meaningful question, listens to the answer, identifies what is missing,
and adapts the next question. Existing quiz generators usually expect a fixed
answer, while generic document chatbots do not provide a transparent or
consistent assessment.

That led to **Viva AI** — an adaptive oral exam coach grounded in the learner's
own materials.

### What it does

The learner uploads a PDF, TXT, or Markdown document, or starts with the built-in
machine learning demo. Viva AI then:

1. identifies important topics in the material;
2. generates an oral question with explicit assessment criteria;
3. evaluates a free-form answer criterion by criterion;
4. explains what was covered and what is still missing;
5. asks a focused follow-up for a real gap or moves to a genuinely new topic;
6. lets the learner retry, skip, or finish the session early;
7. produces a topic mastery map, personalized revision plan, and downloadable
   Markdown report.

Learners can choose a quick 3-question session or a full 5-question exam and
set the difficulty to basic, standard, or advanced. The built-in demo starts
instantly, so the core experience is easy to evaluate without uploading a file.

Questions and feedback follow the selected interface language, while source
evidence remains in the document's original language.

### Transparent assessment

The language model does **not** invent the final numeric score. It classifies
each expected point as:

- `met` — fully and correctly covered;
- `partial` — present but incomplete;
- `missing` — absent or incorrect.

Python then calculates the score deterministically:

$$
\text{score} =
\frac{\sum_i w_i}{N} \times 100,
\qquad
w_i \in \{1,\ 0.5,\ 0\}.
$$

The learner can inspect every criterion, its status, the explanation, and the
supporting fragment of their answer. This makes the assessment easier to trust
and reproduce than a single unexplained LLM-generated number.

### How we built it

Viva AI is a lightweight Streamlit application with a separate testable Python
core.

- **PyMuPDF** extracts text from uploaded PDFs.
- The material is normalized and split into manageable passages.
- Lightweight lexical retrieval selects relevant evidence for each question.
- **Ollama** runs Qwen3 locally through its native API.
- JSON Schema and **Pydantic** validate questions and criterion assessments.
- Python calculates final scores and session summaries.
- Streamlit manages the adaptive exam state and the responsive UI.
- An OpenAI-compatible cloud endpoint remains available as an optional fallback.

The default local setup uses **Qwen3 14B**. It keeps study materials on the
learner's computer and avoids API costs.

### Challenges

The hardest problem was reliable assessment. Our first version asked the model
to output a score from 0 to 100. Testing showed that even strong answers could
receive inconsistent scores. Larger models did not solve the problem.

We redesigned the evaluator around an explicit rubric. The model now performs
the semantic task it is good at — deciding whether each criterion is met —
while deterministic Python code handles the numeric calculation. A regression
set checks incorrect, vague, partial, and complete answers. In the current
golden test, the progression is `0 → 17 → 67 → 100`, and repeated complete
answers receive the same score.

Another challenge was multilingual grounding. Translating a question could
break lexical matching against the original source. We solved this by having
the model select a numbered source passage while the application preserves and
displays the original evidence.

Finally, local inference required balancing quality and latency. Enabling
reasoning improves criterion assessment, but responses take longer. Qwen3 14B
provided the best practical balance for the available hardware.

### Accomplishments that we're proud of

- A complete end-to-end adaptive oral exam experience.
- Transparent, deterministic scoring instead of arbitrary LLM grades.
- Grounded source evidence for questions and feedback.
- Local-first inference with no mandatory API key.
- A polished responsive interface with quick/full modes, difficulty controls,
  retry/skip actions, and a mastery dashboard.
- Automated unit tests and a real-model grading regression check.

### What we learned

We learned that structured output alone does not make an AI system reliable.
The structure must reflect a task that the model can perform consistently.
Breaking assessment into small auditable decisions was more effective than
asking a larger model for a better-looking final number.

We also learned that multilingual interfaces affect the retrieval and grounding
pipeline, not only visible labels. Preserving the original source passage while
translating the learning experience was essential for trustworthy feedback.

### What's next

Next improvements could include:

- OCR for scanned PDFs;
- voice answers using speech recognition;
- spaced-repetition scheduling based on weak criteria;
- teacher-authored rubrics;
- session history and progress tracking;
- a faster hosted demo model for lower latency;
- a larger evaluation set covering more subjects and languages.

## Built with

- Python
- Streamlit
- Ollama
- Qwen3
- Pydantic
- PyMuPDF
- HTTPX
- pytest
- OpenAI-compatible API
- HTML
- CSS

## Try it out links

- GitHub repository: https://github.com/NikitaGavrilenko/viva-ai
- Live demo: `TODO_LIVE_DEMO_URL` (optional)

## Project media

- Thumbnail: `assets/viva-ai-devpost-thumbnail.png`
- Add 2–4 product screenshots before submission:
  - onboarding screen;
  - adaptive question;
  - criterion-by-criterion result;
  - final revision plan.

## Video demo link

`TODO_PUBLIC_VIDEO_URL`

Recommended format: an unlisted YouTube video, 1:45–1:55 long.

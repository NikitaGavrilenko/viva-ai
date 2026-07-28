from __future__ import annotations

import html
from pathlib import Path

import httpx
import streamlit as st

from ui_text import text
from viva_core import (
    AnswerEvaluation,
    CriterionAssessment,
    LLMConfig,
    Question,
    SessionRecord,
    VivaCoach,
    build_markdown_report,
    build_summary,
    demo_question,
    extract_material,
    is_local_api,
)

APP_DIR = Path(__file__).parent
DEMO_PATH = APP_DIR / "demo_materials" / "machine_learning_basics.txt"
DEFAULT_MAX_QUESTIONS = 3

st.set_page_config(
    page_title="Viva AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --violet: #8b5cf6;
  --cyan: #22d3ee;
  --surface: rgba(23, 26, 43, .78);
  --border: rgba(148, 163, 184, .16);
  --muted: #94a3b8;
}
.stApp {
  background:
    radial-gradient(circle at 75% 4%, rgba(139, 92, 246, .16), transparent 26rem),
    radial-gradient(circle at 18% 40%, rgba(34, 211, 238, .08), transparent 25rem),
    #0b1020;
}
.stAppHeader {background: transparent;}
.block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebar"] {
  background: rgba(15, 18, 34, .94);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
.hero {
  position: relative;
  overflow: hidden;
  padding: clamp(1.6rem, 4vw, 3.5rem);
  border: 1px solid rgba(139, 92, 246, .28);
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(30, 41, 59, .94), rgba(49, 46, 129, .72)),
    #171a2b;
  box-shadow: 0 28px 80px rgba(0, 0, 0, .26);
  margin-bottom: 2rem;
}
.hero::after {
  content: "";
  position: absolute;
  width: 22rem;
  height: 22rem;
  right: -7rem;
  top: -10rem;
  border-radius: 50%;
  background: rgba(34, 211, 238, .14);
  filter: blur(5px);
}
.eyebrow {
  color: #c4b5fd;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .13em;
  text-transform: uppercase;
  margin-bottom: .9rem;
}
.hero h1 {
  max-width: 780px;
  color: #f8fafc;
  font-size: clamp(2.25rem, 5vw, 4.2rem);
  line-height: 1.02;
  letter-spacing: -.045em;
  margin: 0 0 1rem;
}
.hero-copy {
  max-width: 670px;
  color: #cbd5e1;
  font-size: 1.08rem;
  line-height: 1.7;
  margin-bottom: 1.4rem;
}
.chips {display: flex; flex-wrap: wrap; gap: .55rem;}
.chip {
  display: inline-flex;
  padding: .42rem .75rem;
  border: 1px solid rgba(196, 181, 253, .2);
  border-radius: 999px;
  background: rgba(15, 23, 42, .35);
  color: #ddd6fe;
  font-size: .8rem;
}
.section-kicker {
  color: #a78bfa;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .8rem;
  margin: .7rem 0 1.5rem;
}
.feature {
  min-height: 145px;
  padding: 1.05rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--surface);
}
.feature-index {
  color: #67e8f9;
  font-size: .72rem;
  font-weight: 700;
  margin-bottom: .7rem;
}
.feature-title {font-weight: 700; color: #f8fafc; margin-bottom: .35rem;}
.feature-copy {color: var(--muted); font-size: .88rem; line-height: 1.55;}
.question-card {
  padding: 1.5rem 1.6rem;
  border: 1px solid rgba(139, 92, 246, .3);
  border-radius: 22px;
  background: linear-gradient(145deg, rgba(30, 41, 59, .92), rgba(30, 27, 75, .65));
  box-shadow: 0 16px 45px rgba(0, 0, 0, .18);
  margin: .9rem 0 1.2rem;
}
.question-topic {
  color: #a5f3fc;
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-bottom: .65rem;
}
.question-text {color: #f8fafc; font-size: 1.35rem; line-height: 1.5;}
.score-banner {
  display: flex;
  align-items: baseline;
  gap: .55rem;
  padding: 1rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--surface);
  margin-bottom: 1rem;
}
.score-number {font-size: 2rem; font-weight: 800; color: #f8fafc;}
.score-label {color: var(--muted);}
.source-card {
  padding: 1rem 1.1rem;
  border-left: 3px solid #22d3ee;
  border-radius: 4px 14px 14px 4px;
  background: rgba(34, 211, 238, .07);
  color: #cbd5e1;
  font-size: .9rem;
  line-height: 1.6;
  margin: 1rem 0;
}
.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .85rem 1.05rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(23, 26, 43, .72);
  margin-bottom: 1rem;
}
.session-brand {
  color: #f8fafc;
  font-weight: 800;
  letter-spacing: -.02em;
}
.session-meta {
  color: var(--muted);
  font-size: .82rem;
}
.mastery-row {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  color: #cbd5e1;
  font-size: .9rem;
  margin-top: .6rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--border);
  border-radius: 20px;
  background: rgba(23, 26, 43, .45);
}
div[data-testid="stMetric"] {
  padding: 1rem 1.1rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--surface);
}
.stButton > button {
  min-height: 3rem;
  border-radius: 12px;
  font-weight: 700;
}
.stTextArea textarea, .stTextInput input {
  border-radius: 12px;
}
@media (max-width: 760px) {
  .feature-grid {grid-template-columns: 1fr;}
  .hero {border-radius: 20px;}
  .question-text {font-size: 1.12rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


def reset_session() -> None:
    for key in (
        "coach",
        "records",
        "current_question",
        "evaluation",
        "finished",
        "material_name",
        "max_questions",
        "difficulty",
    ):
        st.session_state.pop(key, None)


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models() -> list[str]:
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
    except (httpx.HTTPError, KeyError, TypeError):
        return []


if "language_toggle" not in st.session_state:
    st.session_state.language_toggle = "RU"

with st.sidebar:
    language_label = st.segmented_control(
        "Language / Язык",
        options=["RU", "EN"],
        key="language_toggle",
        label_visibility="collapsed",
        disabled="coach" in st.session_state,
    )
    lang = "en" if language_label == "EN" else "ru"
    t = lambda key, **values: text(lang, key, **values)

    env_config = LLMConfig.from_env()
    default_local = is_local_api(env_config.base_url)
    ollama_models = get_ollama_models()
    if ollama_models:
        st.success(t("ollama_online"), icon=":material/check_circle:")
    else:
        st.warning(t("ollama_offline"), icon=":material/warning:")
    st.caption(f"◈ {t('local_privacy')}")

    with st.expander(t("advanced_settings")):
        provider = st.radio(
            t("model_mode"),
            options=["local", "cloud"],
            format_func=lambda value: (
                t("local_ollama") if value == "local" else t("cloud_api")
            ),
            horizontal=True,
        )

        if provider == "local":
            base_url = "http://localhost:11434/v1"
            if ollama_models:
                preferred_model = (
                    env_config.model
                    if env_config.model in ollama_models
                    else "qwen3:14b"
                    if "qwen3:14b" in ollama_models
                    else ollama_models[0]
                )
                model = st.selectbox(
                    t("local_model"),
                    ollama_models,
                    index=ollama_models.index(preferred_model),
                )
            else:
                model = st.text_input(
                    t("local_model"), value="qwen3:14b"
                )
            api_key = ""
        else:
            api_key = st.text_input(
                t("api_key"),
                value=env_config.api_key if not default_local else "",
                type="password",
                help=t("api_key_help"),
            )
            base_url = st.text_input(
                t("base_url"),
                value=(
                    env_config.base_url
                    if not default_local
                    else "https://api.openai.com/v1"
                ),
            )
            model = st.text_input(
                t("model"),
                value=(
                    env_config.model
                    if not default_local
                    else "gpt-4.1-mini"
                ),
            )
            st.caption(t("api_compatibility"))

    if "coach" in st.session_state and st.button(
        t("restart"), use_container_width=True
    ):
        reset_session()
        st.rerun()


def render_error(error: Exception) -> None:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 401:
            st.error(t("api_bad_key"))
        elif status == 429:
            st.error(t("api_limit"))
        else:
            st.error(t("api_error", status=status))
    elif isinstance(error, httpx.TimeoutException):
        st.error(t("api_timeout"))
    else:
        st.error(str(error))


def skipped_evaluation(question: Question) -> AnswerEvaluation:
    expected_points = question.expected_points
    return AnswerEvaluation(
        score=0,
        feedback=t("skipped_feedback"),
        strengths=[],
        gaps=[],
        evidence_excerpt=question.source_excerpt,
        criterion_results=[
            CriterionAssessment(
                criterion_index=index,
                status="missing",
                explanation=t("skipped_feedback"),
            )
            for index in range(1, len(expected_points) + 1)
        ],
        follow_up_needed=False,
    )


def render_hero() -> None:
    st.markdown(
        f"""
<section class="hero">
  <div class="eyebrow">✦ {html.escape(t("eyebrow"))}</div>
  <h1>{html.escape(t("hero_title"))}</h1>
  <div class="hero-copy">{html.escape(t("hero_body"))}</div>
  <div class="chips">
    <span class="chip">◎ {html.escape(t("chip_grounded"))}</span>
    <span class="chip">↗ {html.escape(t("chip_adaptive"))}</span>
    <span class="chip">◇ {html.escape(t("chip_private"))}</span>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_session_header(max_questions: int, difficulty: str) -> None:
    difficulty_label = t(f"difficulty_{difficulty}")
    st.markdown(
        (
            '<div class="session-header">'
            '<div><div class="session-brand">✦ Viva AI</div>'
            f'<div class="session-meta">{html.escape(t("session_label"))}'
            f" · {max_questions} · {html.escape(difficulty_label)}</div></div>"
            f'<div class="chip">◇ {html.escape(t("chip_private"))}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


if "coach" not in st.session_state:
    render_hero()
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(f'<div class="section-kicker">{t("how_title")}</div>', unsafe_allow_html=True)
        features = [
            ("01", t("how_1_title"), t("how_1_body")),
            ("02", t("how_2_title"), t("how_2_body")),
            ("03", t("how_3_title"), t("how_3_body")),
        ]
        feature_html = "".join(
            (
                '<div class="feature">'
                f'<div class="feature-index">{index}</div>'
                f'<div class="feature-title">{html.escape(title)}</div>'
                f'<div class="feature-copy">{html.escape(body)}</div>'
                "</div>"
            )
            for index, title, body in features
        )
        st.markdown(
            f'<div class="feature-grid">{feature_html}</div>',
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.subheader(t("material_title"))
            st.caption(t("material_body"))
            source = st.radio(
                t("material_title"),
                options=["demo", "upload"],
                format_func=lambda value: (
                    t("demo_source") if value == "demo" else t("upload_source")
                ),
                horizontal=True,
                label_visibility="collapsed",
            )

            material: str | None = None
            material_name = ""
            if source == "demo":
                material = DEMO_PATH.read_text(encoding="utf-8")
                material_name = t("demo_name")
                with st.expander(t("demo_expander")):
                    st.write(t("demo_description"))
            else:
                uploaded = st.file_uploader(
                    t("upload_label"),
                    type=["pdf", "txt", "md"],
                    help=t("upload_help"),
                )
                if uploaded is not None:
                    if uploaded.size > 10 * 1024 * 1024:
                        st.error(t("file_too_large"))
                    else:
                        try:
                            material = extract_material(
                                uploaded.name, uploaded.getvalue()
                            )
                            material_name = uploaded.name
                            st.success(
                                t("material_loaded", chars=f"{len(material):,}")
                            )
                        except ValueError as error:
                            st.error(str(error))

            st.markdown(f"#### {t('start_title')}")
            exam_mode = st.segmented_control(
                t("exam_mode"),
                options=["quick", "full"],
                default="quick",
                format_func=lambda value: (
                    t("quick_mode") if value == "quick" else t("full_mode")
                ),
            )
            difficulty = st.segmented_control(
                t("difficulty"),
                options=["basic", "standard", "advanced"],
                default="standard",
                format_func=lambda value: t(f"difficulty_{value}"),
            )
            max_questions = 3 if exam_mode == "quick" else 5
            st.caption(
                t("demo_instant") if source == "demo" else t("start_body")
            )
            if st.button(
                t("start_button"),
                type="primary",
                use_container_width=True,
                disabled=material is None,
            ):
                if provider == "cloud" and not api_key.strip():
                    st.error(t("missing_key"))
                else:
                    config = LLMConfig(
                        api_key=api_key.strip(),
                        base_url=base_url.strip(),
                        model=model.strip(),
                    )
                    try:
                        with st.spinner(t("studying")):
                            coach = VivaCoach(
                                material or "",
                                config,
                                lang,
                                difficulty or "standard",
                            )
                            question = (
                                demo_question(
                                    lang, difficulty or "standard"
                                )
                                if source == "demo"
                                else coach.first_question()
                            )
                        st.session_state.coach = coach
                        st.session_state.records = []
                        st.session_state.current_question = question
                        st.session_state.evaluation = None
                        st.session_state.finished = False
                        st.session_state.material_name = material_name
                        st.session_state.max_questions = max_questions
                        st.session_state.difficulty = (
                            difficulty or "standard"
                        )
                        st.rerun()
                    except Exception as error:
                        render_error(error)

else:
    coach: VivaCoach = st.session_state.coach
    coach.language = lang
    coach.difficulty = st.session_state.get("difficulty", "standard")
    records: list[SessionRecord] = st.session_state.records
    max_questions = st.session_state.get(
        "max_questions", DEFAULT_MAX_QUESTIONS
    )
    render_session_header(max_questions, coach.difficulty)
    current_number = (
        len(records)
        if st.session_state.evaluation is not None
        else len(records) + 1
    )
    current_number = min(max(current_number, 1), max_questions)
    st.progress(
        len(records) / max_questions,
        text=t("progress", current=current_number, total=max_questions),
    )
    st.caption(f"{t('material')}: {st.session_state.material_name}")

    if st.session_state.finished:
        summary = build_summary(records, lang)
        st.markdown(f"## {t('completed')}")
        st.caption(t("completed_body"))
        left_metric, middle_metric, right_metric = st.columns(3)
        left_metric.metric(t("overall_score"), f"{summary.overall_score}/100")
        middle_metric.metric(t("topics_checked"), len(records))
        right_metric.metric(t("mastered"), len(summary.mastered_topics))

        if summary.mastered_topics:
            st.success(
                f"**{t('mastered')}:** " + ", ".join(summary.mastered_topics)
            )
        if summary.weak_topics:
            st.warning(
                f"**{t('needs_review')}:** " + ", ".join(summary.weak_topics)
            )

        with st.container(border=True):
            st.markdown(f"#### {t('mastery_map')}")
            for record in records:
                st.markdown(
                    (
                        '<div class="mastery-row">'
                        f"<span>{html.escape(record.question.topic)}</span>"
                        f"<strong>{record.evaluation.score}%</strong>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.progress(record.evaluation.score / 100)

        with st.container(border=True):
            st.markdown(f"#### {t('revision_plan')}")
            for index, item in enumerate(summary.revision_plan, start=1):
                st.write(f"**{index:02}.** {item}")

        with st.expander(t("details")):
            for index, record in enumerate(records, start=1):
                st.markdown(
                    f"**{index}. {record.question.topic} — "
                    f"{record.evaluation.score}/100**"
                )
                st.write(record.evaluation.feedback)
        st.info(t("repeat_tip"), icon=":material/event_repeat:")
        report = build_markdown_report(
            records, st.session_state.material_name, lang
        )
        download_col, restart_col = st.columns(2)
        download_col.download_button(
            t("download_report"),
            data=report,
            file_name="viva-ai-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if restart_col.button(
            t("new_exam"), type="primary", use_container_width=True
        ):
            reset_session()
            st.rerun()

    else:
        question = st.session_state.current_question
        question_column, source_column = st.columns(
            [0.7, 0.3], gap="large"
        )
        with question_column:
            st.markdown(
                (
                    '<div class="question-card">'
                    f'<div class="question-topic">'
                    f"{html.escape(question.topic)}</div>"
                    f'<div class="question-text">'
                    f"{html.escape(question.text)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with source_column:
            with st.container(border=True):
                st.markdown(f"**{t('source_context')}**")
                st.caption(question.source_excerpt)
                st.caption(t("source_original"))

        evaluation = st.session_state.evaluation
        if evaluation is None:
            answer = st.text_area(
                t("answer_label"),
                height=170,
                placeholder=t("answer_placeholder"),
                key=f"answer_{len(records)}",
            )
            check_col, skip_col, finish_col = st.columns(
                [0.55, 0.23, 0.22]
            )
            check_clicked = check_col.button(
                t("check_answer"),
                type="primary",
                use_container_width=True,
                disabled=len(answer.strip()) < 10,
            )
            skip_clicked = skip_col.button(
                t("skip_question"),
                use_container_width=True,
            )
            finish_clicked = finish_col.button(
                t("finish_early"),
                use_container_width=True,
                disabled=not records,
            )
            if check_clicked:
                try:
                    with st.spinner(t("evaluating")):
                        evaluation = coach.evaluate(question, answer.strip())
                    record = SessionRecord(
                        question=question,
                        answer=answer.strip(),
                        evaluation=evaluation,
                    )
                    st.session_state.records = [*records, record]
                    st.session_state.evaluation = evaluation
                    st.rerun()
                except Exception as error:
                    render_error(error)
            if skip_clicked:
                skipped = SessionRecord(
                    question=question,
                    answer="—",
                    evaluation=skipped_evaluation(question),
                )
                updated_records = [*records, skipped]
                st.session_state.records = updated_records
                if len(updated_records) >= max_questions:
                    st.session_state.finished = True
                    st.rerun()
                try:
                    with st.spinner(t("adapting")):
                        next_question = coach.next_question(updated_records)
                    st.session_state.current_question = next_question
                    st.session_state.evaluation = None
                    st.rerun()
                except Exception as error:
                    render_error(error)
            if finish_clicked:
                st.session_state.finished = True
                st.rerun()
        else:
            st.markdown(
                (
                    '<div class="score-banner">'
                    f'<span class="score-number">{evaluation.score}</span>'
                    f'<span class="score-label">/ 100 · {html.escape(t("score"))}</span>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.write(evaluation.feedback)

            result_left, result_right = st.columns(2, gap="medium")
            with result_left:
                if evaluation.strengths:
                    st.success(
                        f"**{t('strengths')}**\n\n"
                        + "\n\n".join(
                            f"✓ {item}" for item in evaluation.strengths
                        )
                    )
            with result_right:
                if evaluation.gaps:
                    st.warning(
                        f"**{t('gaps')}**\n\n"
                        + "\n\n".join(f"→ {item}" for item in evaluation.gaps)
                    )

            if evaluation.criterion_results:
                status_labels = {
                    "met": ("✅", t("status_met")),
                    "partial": ("◐", t("status_partial")),
                    "missing": ("○", t("status_missing")),
                }
                with st.expander(t("breakdown"), expanded=True):
                    for item in evaluation.criterion_results:
                        criterion = question.expected_points[
                            item.criterion_index - 1
                        ]
                        icon, label = status_labels[item.status]
                        with st.container(border=True):
                            st.markdown(
                                f"**{icon} {item.criterion_index}. "
                                f"{label}** — {criterion}"
                            )
                            st.caption(item.explanation)
                            if item.answer_evidence:
                                st.caption(
                                    f"{t('answer_fragment')}: "
                                    f"«{item.answer_evidence}»"
                                )

            st.markdown(
                (
                    '<div class="source-card">'
                    f'<strong>{html.escape(t("source"))}</strong><br>'
                    f'“{html.escape(evaluation.evidence_excerpt)}”'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            button_label = (
                t("show_summary")
                if len(st.session_state.records) >= max_questions
                else t("next_question")
            )
            retry_col, next_col, end_col = st.columns([0.25, 0.5, 0.25])
            retry_clicked = retry_col.button(
                t("retry_answer"), use_container_width=True
            )
            next_clicked = next_col.button(
                button_label,
                type="primary",
                use_container_width=True,
            )
            end_clicked = end_col.button(
                t("finish_early"),
                use_container_width=True,
                disabled=len(st.session_state.records) >= max_questions,
            )
            if retry_clicked:
                answer_key = f"answer_{len(st.session_state.records) - 1}"
                st.session_state.records = st.session_state.records[:-1]
                st.session_state.evaluation = None
                st.session_state.pop(answer_key, None)
                st.rerun()
            if next_clicked:
                if len(st.session_state.records) >= max_questions:
                    st.session_state.finished = True
                    st.rerun()
                try:
                    with st.spinner(t("adapting")):
                        next_question = coach.next_question(
                            st.session_state.records
                        )
                    st.session_state.current_question = next_question
                    st.session_state.evaluation = None
                    st.rerun()
                except Exception as error:
                    render_error(error)
            if end_clicked:
                st.session_state.finished = True
                st.rerun()

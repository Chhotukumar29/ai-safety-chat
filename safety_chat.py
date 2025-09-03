"""
Project Title: AI Safety Chat 
====================================================

What this demo shows
--------------------
- **Abuse Language Detection** (toxicity/harassment/hate/sexual/dangerous) via Gemini classification
- **Escalation Pattern Recognition** across rolling conversation window
- **Crisis Intervention** (self-harm/distress signal & escalation to human)
- **Age-appropriate Content Filtering** (guardian mode)
- **Real-time-ish**: processes each message on send; lightweight, no database
- **Privacy-by-default**: no server logging, all state is in Streamlit session
- **Interpretability**: per-message JSON scores + rule-based decisions explained

Requirements
-----------
- Python 3.10+
- Packages: `pip install streamlit google-genai pydantic python-dotenv`
- Set environment var `GEMINI_API_KEY` (or create a `.env` file with it)

Run
---
```bash
streamlit run safety_chat_poc.py
```

Notes
-----
- Uses the official **Google GenAI SDK** (`google-genai`) and **Gemini 2.5 Flash** by default.
- Uses **structured JSON output** prompting for stable parsing.
- All model calls are made with **explicit safety settings**.
- You can tweak thresholds in the `SafetyThresholds` dataclass.
"""

import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import deque

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# --- Gemini SDK ---
from google import genai
from google.genai import types

# ----------------------------
# Configuration & thresholds
# ----------------------------

@dataclass
class SafetyThresholds:
    # Base thresholds (0-1). You can tighten for minors below.
    harassment: float = 0.55
    hate: float = 0.55
    sexual: float = 0.50
    dangerous: float = 0.50
    crisis_self_harm: float = 0.50

    # Escalation rules
    window: int = 6                # rolling messages to watch
    escalate_avg_risk: float = 0.50
    escalate_growth: float = 0.12   # average increase over window


class Classification(BaseModel):
    # Scores are 0..1. Calibrated by the LLM.
    harassment: float = Field(ge=0, le=1)
    hate: float = Field(ge=0, le=1)
    sexual: float = Field(ge=0, le=1)
    dangerous: float = Field(ge=0, le=1)
    self_harm: float = Field(ge=0, le=1)
    sentiment_negativity: float = Field(ge=0, le=1)
    overall_risk: float = Field(ge=0, le=1)
    rationale: str


class Decision(BaseModel):
    abuse_flags: List[str]
    block: bool
    crisis: bool
    guardian_filtered: bool
    guardian_reasons: List[str]
    escalate_conversation: bool
    explanation: str


# ----------------------------
# Utility: Gemini client & prompt
# ----------------------------

def get_client() -> genai.Client:
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.stop()
    # The python client picks up GEMINI_API_KEY from env automatically.
    return genai.Client()


CLASSIFIER_SYSTEM_PROMPT = (
    """
    You are a safety classifier for a chat platform. For the given user message,
    return a single **valid JSON** object with these numeric fields in [0,1]:
    - harassment: aggression/abuse/profanity level
    - hate: hate speech/derogation against protected classes
    - sexual: sexual or age-inappropriate content
    - dangerous: encouragement or facilitation of harmful/illegal acts
    - self_harm: indicators of self-harm/suicide/acute crisis or severe distress
    - sentiment_negativity: general negativity/anger (0 positive -> 1 very negative)
    - overall_risk: calibrated summary risk across all axes (not average; your judgment)
    Include a short natural-language "rationale" string (<= 40 words).

    Output **only** the JSON. No markdown. No preface.
    """
)


def classify_with_gemini(client: genai.Client, text: str) -> Classification:
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        CLASSIFIER_SYSTEM_PROMPT,
        text,
    ],
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        ),
    )


    raw = response.text.strip()
    try:
        data = Classification.model_validate_json(raw)
    except ValidationError as e:
        st.warning("Classifier JSON parse failed; returning conservative default.")
        data = Classification(
            harassment=0, hate=0, sexual=0, dangerous=0, self_harm=0,
            sentiment_negativity=0.5, overall_risk=0.4, rationale="fallback"
        )
    return data


# ----------------------------
# Rule engine (deterministic, transparent)
# ----------------------------

def make_decision(
    cls: Classification,
    thresholds: SafetyThresholds,
    user_age: int,
    history_scores: deque,
) -> Decision:
    abuse_flags = []
    if cls.harassment >= thresholds.harassment:
        abuse_flags.append("harassment")
    if cls.hate >= thresholds.hate:
        abuse_flags.append("hate")
    if cls.sexual >= thresholds.sexual:
        abuse_flags.append("sexual")
    if cls.dangerous >= thresholds.dangerous:
        abuse_flags.append("dangerous")

    crisis = cls.self_harm >= thresholds.crisis_self_harm

    # Guardian filtering: stricter for minors
    guardian_filtered = False
    guardian_reasons: List[str] = []
    if user_age < 13 and (cls.sexual > 0.15 or cls.harassment > 0.35 or cls.dangerous > 0.20):
        guardian_filtered = True
        if cls.sexual > 0.15:
            guardian_reasons.append("sexual content")
        if cls.harassment > 0.35:
            guardian_reasons.append("abusive language")
        if cls.dangerous > 0.20:
            guardian_reasons.append("dangerous content")
    elif 13 <= user_age < 18 and (cls.sexual > 0.25 or cls.harassment > 0.45 or cls.dangerous > 0.30):
        guardian_filtered = True
        if cls.sexual > 0.25:
            guardian_reasons.append("sexual content")
        if cls.harassment > 0.45:
            guardian_reasons.append("abusive language")
        if cls.dangerous > 0.30:
            guardian_reasons.append("dangerous content")

    # Block if any direct abuse flag OR crisis OR guardian filter
    block = bool(abuse_flags) or crisis or guardian_filtered

    # Escalation pattern detection over the rolling window
    escalate_conversation = False
    history_scores.append(cls.overall_risk)
    if len(history_scores) > thresholds.window:
        history_scores.popleft()
    if len(history_scores) >= 3:
        avg_risk = sum(history_scores) / len(history_scores)
        growth = history_scores[-1] - history_scores[0]
        if avg_risk >= thresholds.escalate_avg_risk or growth >= thresholds.escalate_growth:
            escalate_conversation = True
    else:
        avg_risk = cls.overall_risk
        growth = 0.0

    explanation = (
        f"abuse_flags={abuse_flags}, crisis={crisis}, guardian={guardian_filtered}"
        f"(reasons={guardian_reasons}), avg_risk={avg_risk:.2f}, growth={growth:.2f}"
    )

    return Decision(
        abuse_flags=abuse_flags,
        block=block,
        crisis=crisis,
        guardian_filtered=guardian_filtered,
        guardian_reasons=guardian_reasons,
        escalate_conversation=escalate_conversation,
        explanation=explanation,
    )


# ----------------------------
# Streamlit UI
# ----------------------------

def ui_header():
    st.set_page_config(page_title="AI Safety Chat POC (Gemini)", page_icon="🛡️", layout="centered")
    st.title("🛡️ AI Safety Chat POC – Gemini")
    # st.caption(
    #     "This demo classifies each message, applies rule-based decisions, and shows why a message was blocked or escalated."
    # )


def crisis_banner():
    st.warning(
        "If you or someone you know may be in crisis or considering self-harm, seek immediate help: "
        "Call your local emergency number. In India, contact AASRA (24x7) at 91-22-27546669 or Kiran Helpline 1800-599-0019."
    )


def main():
    ui_header()
    crisis_banner()

    client = get_client()

    with st.sidebar:
        st.subheader("Guardian / Policy Settings")
        user_age = st.number_input("User age", min_value=5, max_value=120, value=18)
        st.write("Thresholds (advanced)")
        th = SafetyThresholds(
            harassment=st.slider("Harassment", 0.0, 1.0, 0.55),
            hate=st.slider("Hate", 0.0, 1.0, 0.55),
            sexual=st.slider("Sexual", 0.0, 1.0, 0.50),
            dangerous=st.slider("Dangerous", 0.0, 1.0, 0.50),
            crisis_self_harm=st.slider("Self-harm", 0.0, 1.0, 0.50),
            window=st.slider("Escalation window (messages)", 3, 12, 6),
            escalate_avg_risk=st.slider("Escalate if avg risk ≥", 0.0, 1.0, 0.50),
            escalate_growth=st.slider("Escalate if growth ≥", 0.0, 1.0, 0.12),
        )
        st.divider()
        st.caption("No data is logged. All processing is in-memory for this demo.")

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, text, cls, decision)
    if "risk_window" not in st.session_state:
        st.session_state.risk_window = deque(maxlen=th.window)
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    with st.container(border=True):
        st.subheader("Chat Simulator")
        
        # Use a different approach for clearing input
        input_value = "" if st.session_state.clear_input else st.session_state.get("current_input", "")
        
        msg = st.text_area(
            "Type a message and press Send",
            height=120,
            value=input_value,
            key="message_input",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            send = st.button("Send & Moderate", use_container_width=True)
        with col2:
            clear = st.button("Clear chat", use_container_width=True, type="secondary")

    # Reset clear flag after using it
    if st.session_state.clear_input:
        st.session_state.clear_input = False

    if clear:
        st.session_state.history.clear()
        st.session_state.risk_window.clear()
        st.session_state.clear_input = True
        if "current_input" in st.session_state:
            del st.session_state.current_input
        st.rerun()

    if send and msg.strip():
        st.session_state.current_input = msg  # Store current input
        
        with st.spinner("Classifying..."):
            cls = classify_with_gemini(client, msg)
            decision = make_decision(cls, th, user_age, st.session_state.risk_window)

        st.session_state.history.append(("user", msg, cls, decision))
        
        # Clear the input after processing
        st.session_state.clear_input = True
        st.rerun()

    # Render conversation
    for role, text, cls, decision in st.session_state.history[-50:]:
        with st.chat_message(role):
            st.markdown(text)
            with st.expander("Show safety analysis", expanded=True):
                st.json(cls.model_dump(), expanded=False)
                st.markdown(
                    f"**Decision:** {'BLOCKED' if decision.block else 'ALLOW'}"
                    f" &nbsp; | &nbsp; **Crisis:** {'YES' if decision.crisis else 'NO'}"
                    f" &nbsp; | &nbsp; **Guardian filter:** {'YES' if decision.guardian_filtered else 'NO'}"
                    f" &nbsp; | &nbsp; **Escalate thread:** {'YES' if decision.escalate_conversation else 'NO'}"
                )
                st.caption(decision.explanation)

        if decision.block:
            with st.container(border=True):
                st.error("This message was blocked by the safety system.")
                if decision.crisis:
                    st.warning(
                        "Crisis detected: Consider urgent human intervention and provide helpline resources."
                    )
                if decision.guardian_filtered:
                    st.info(
                        "Guardian filtering active: age-appropriate policy blocked this content."
                    )
                if decision.abuse_flags:
                    st.write(
                        "Abuse categories: " + ", ".join(decision.abuse_flags)
                    )

    st.divider()
    st.subheader("Moderator Console (read-only)")
    avg_risk = (
        sum(st.session_state.risk_window) / len(st.session_state.risk_window)
        if st.session_state.risk_window
        else 0.0
    )
    st.metric("Rolling average risk", f"{avg_risk:.2f}")
    st.progress(min(1.0, avg_risk))

if __name__ == "__main__":
    main()
    
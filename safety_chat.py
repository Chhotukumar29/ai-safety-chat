"""
Project Title: AI Safety Chat 
====================================================

What this demo shows
--------------------
- **Multi-category Abuse Detection** (harassment/hate/sexual/dangerous/crisis) via Gemini
- **JSON Export/Import** functionality for session data
- **Optional History Display** (expandable/collapsible)
- **Enhanced Pattern Recognition** across rolling conversation window
- **Crisis Intervention** with escalation to human moderators
- **Age-appropriate Content Filtering** (guardian mode)
- **Improved Classification** with better category detection
- **Privacy-by-default**: optional local JSON storage only

Requirements
-----------
- Python 3.10+
- Packages: `pip install streamlit google-genai pydantic python-dotenv`
- Set environment var `GEMINI_API_KEY` (or create a `.env` file with it)
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import uuid

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
    harassment: float = 0.45  # Lowered for better detection
    hate: float = 0.40        # Lowered for better detection
    sexual: float = 0.35      # Lowered for better detection
    dangerous: float = 0.40   # Lowered for better detection
    crisis_self_harm: float = 0.30  # Lowered for safety

    # Escalation rules
    window: int = 6                  # rolling messages to watch
    escalate_avg_risk: float = 0.45
    escalate_growth: float = 0.10   # average increase over window


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
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Decision(BaseModel):
    message_id: str
    abuse_flags: List[str]
    block: bool
    crisis: bool
    guardian_filtered: bool
    guardian_reasons: List[str]
    escalate_conversation: bool
    explanation: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConversationEntry(BaseModel):
    message_id: str
    role: str
    text: str
    classification: Classification
    decision: Decision
    user_age: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ----------------------------
# Enhanced Gemini client & prompt
# ----------------------------

def get_client() -> genai.Client:
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set your API key in the .env file.")
        st.stop()
    return genai.Client()


# Enhanced prompt for better category detection
ENHANCED_CLASSIFIER_PROMPT = """
You are an advanced safety classifier for chat platforms. Analyze the given message and return a JSON object with these scores (0.0 to 1.0):

**Category Definitions:**
- harassment: Personal attacks, bullying, aggressive language, insults, threats
- hate: Discrimination, prejudice against race/religion/gender/orientation/nationality
- sexual: Sexual content, adult themes, inappropriate relationships, suggestive material  
- dangerous: Instructions for harm, illegal activities, violence, self-endangerment
- self_harm: Suicide ideation, self-injury, depression, crisis language, hopelessness
- sentiment_negativity: Overall emotional negativity, anger, frustration (0=positive, 1=very negative)
- overall_risk: Your professional judgment of total safety risk (not just average)

**Scoring Guidelines:**
- 0.0-0.2: Safe/minimal concern
- 0.3-0.5: Moderate concern, monitor
- 0.6-0.8: High concern, likely block
- 0.9-1.0: Severe concern, definite block

**Examples:**
- "I hate you, you're stupid!" → harassment: 0.75, hate: 0.1, overall_risk: 0.7
- "Kill all [group]" → hate: 0.95, dangerous: 0.8, overall_risk: 0.95
- "I want to die" → self_harm: 0.85, sentiment_negativity: 0.9, overall_risk: 0.85
- "Let's have sex" → sexual: 0.8, overall_risk: 0.6

Include a brief "rationale" (≤50 words) explaining your scoring.

Return ONLY valid JSON. No markdown, no explanation outside JSON.
"""


def classify_with_gemini(client: genai.Client, text: str) -> Classification:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                ENHANCED_CLASSIFIER_PROMPT,
                f"Message to classify: {text}",
            ],
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,  # Let our system handle it
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                ],
                temperature=0.1,  # Low temperature for consistent classification
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        data = Classification.model_validate_json(raw)
        return data
        
    except Exception as e:
        st.warning(f"Classifier error: {str(e)}. Using fallback scoring.")
        # More nuanced fallback based on text content
        text_lower = text.lower()
        fallback_harassment = 0.7 if any(word in text_lower for word in ['hate', 'stupid', 'idiot', 'kill']) else 0.1
        fallback_sexual = 0.6 if any(word in text_lower for word in ['sex', 'porn', 'adult']) else 0.1
        fallback_dangerous = 0.8 if any(word in text_lower for word in ['bomb', 'weapon', 'hurt']) else 0.1
        fallback_self_harm = 0.8 if any(word in text_lower for word in ['die', 'suicide', 'kill myself']) else 0.1
        
        return Classification(
            harassment=fallback_harassment,
            hate=0.2,
            sexual=fallback_sexual,
            dangerous=fallback_dangerous,
            self_harm=fallback_self_harm,
            sentiment_negativity=0.6,
            overall_risk=max(fallback_harassment, fallback_sexual, fallback_dangerous, fallback_self_harm),
            rationale="Fallback classification due to API error"
        )


# ----------------------------
# Enhanced rule engine
# ----------------------------

def make_decision(
    cls: Classification,
    thresholds: SafetyThresholds,
    user_age: int,
    history_scores: deque,
    message_id: str,
) -> Decision:
    abuse_flags = []
    
    # Check each category against thresholds
    if cls.harassment >= thresholds.harassment:
        abuse_flags.append("harassment")
    if cls.hate >= thresholds.hate:
        abuse_flags.append("hate")
    if cls.sexual >= thresholds.sexual:
        abuse_flags.append("sexual")
    if cls.dangerous >= thresholds.dangerous:
        abuse_flags.append("dangerous")

    crisis = cls.self_harm >= thresholds.crisis_self_harm

    # Enhanced guardian filtering with age-specific rules
    guardian_filtered = False
    guardian_reasons: List[str] = []
    
    if user_age < 13:
        strict_thresholds = {"sexual": 0.15, "harassment": 0.30, "dangerous": 0.20, "hate": 0.25}
        if cls.sexual > strict_thresholds["sexual"]:
            guardian_filtered = True
            guardian_reasons.append("sexual content")
        if cls.harassment > strict_thresholds["harassment"]:
            guardian_filtered = True
            guardian_reasons.append("abusive language")
        if cls.dangerous > strict_thresholds["dangerous"]:
            guardian_filtered = True
            guardian_reasons.append("dangerous content")
        if cls.hate > strict_thresholds["hate"]:
            guardian_filtered = True
            guardian_reasons.append("hate speech")
            
    elif 13 <= user_age < 18:
        teen_thresholds = {"sexual": 0.25, "harassment": 0.40, "dangerous": 0.30, "hate": 0.35}
        if cls.sexual > teen_thresholds["sexual"]:
            guardian_filtered = True
            guardian_reasons.append("sexual content")
        if cls.harassment > teen_thresholds["harassment"]:
            guardian_filtered = True
            guardian_reasons.append("abusive language")
        if cls.dangerous > teen_thresholds["dangerous"]:
            guardian_filtered = True
            guardian_reasons.append("dangerous content")
        if cls.hate > teen_thresholds["hate"]:
            guardian_filtered = True
            guardian_reasons.append("hate speech")

    # Block if any direct abuse flag OR crisis OR guardian filter
    block = bool(abuse_flags) or crisis or guardian_filtered

    # Enhanced escalation pattern detection
    escalate_conversation = False
    history_scores.append(cls.overall_risk)
    if len(history_scores) > thresholds.window:
        history_scores.popleft()
        
    if len(history_scores) >= 3:
        avg_risk = sum(history_scores) / len(history_scores)
        growth = history_scores[-1] - history_scores[0]
        
        # Multiple escalation triggers
        if (avg_risk >= thresholds.escalate_avg_risk or 
            growth >= thresholds.escalate_growth or
            cls.overall_risk > 0.8):  # High single-message risk
            escalate_conversation = True
    else:
        avg_risk = cls.overall_risk
        growth = 0.0

    explanation = (
        f"Categories: {abuse_flags} | Crisis: {crisis} | Guardian: {guardian_filtered} "
        f"({guardian_reasons}) | Avg Risk: {avg_risk:.2f} | Growth: {growth:.2f}"
    )

    return Decision(
        message_id=message_id,
        abuse_flags=abuse_flags,
        block=block,
        crisis=crisis,
        guardian_filtered=guardian_filtered,
        guardian_reasons=guardian_reasons,
        escalate_conversation=escalate_conversation,
        explanation=explanation,
    )


# ----------------------------
# JSON Export/Import functions
# ----------------------------

def export_conversation_history(history: List[ConversationEntry]) -> str:
    """Export conversation history to JSON string"""
    return json.dumps([entry.model_dump() for entry in history], indent=2, ensure_ascii=False)


def import_conversation_history(json_str: str) -> List[ConversationEntry]:
    """Import conversation history from JSON string"""
    try:
        data = json.loads(json_str)
        return [ConversationEntry(**entry) for entry in data]
    except Exception as e:
        st.error(f"Failed to import history: {e}")
        return []


def save_to_file(data: str, filename: str):
    """Save data to downloadable file"""
    return data.encode('utf-8')


# ----------------------------
# Enhanced Streamlit UI
# ----------------------------

def ui_header():
    st.set_page_config(
        page_title="AI Safety Chat", 
        page_icon="🛡️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🛡️ AI Safety Chat")
    st.caption("Advanced content moderation with multi-category detection and JSON export capabilities")


def crisis_banner():
    st.warning(
        "🆘 **Crisis Resources**: If you or someone you know may be in crisis, seek immediate help\n"
        "Emergency services:\n"
        "AASRA (India): 91-22-27546669\n"
        "Kiran: 1800-599-0019\n"
    )

def main():
    ui_header()
    crisis_banner()

    client = get_client()

    # Enhanced sidebar with better organization
    with st.sidebar:
        st.subheader("🔧 Configuration")
        
        # User settings
        user_age = st.number_input("👤 User age", min_value=5, max_value=120, value=18)
        
        st.subheader("⚖️ Safety Thresholds")
        with st.expander("Category Thresholds", expanded=False):
            th = SafetyThresholds(
                harassment=st.slider("Harassment", 0.0, 1.0, 0.45, step=0.05),
                hate=st.slider("Hate Speech", 0.0, 1.0, 0.40, step=0.05),
                sexual=st.slider("Sexual Content", 0.0, 1.0, 0.35, step=0.05),
                dangerous=st.slider("Dangerous Content", 0.0, 1.0, 0.40, step=0.05),
                crisis_self_harm=st.slider("Self-harm/Crisis", 0.0, 1.0, 0.30, step=0.05),
            )
        
        with st.expander("Escalation Settings", expanded=False):
            th.window = st.slider("Rolling window (messages)", 3, 12, 6)
            th.escalate_avg_risk = st.slider("Escalate if avg risk ≥", 0.0, 1.0, 0.45, step=0.05)
            th.escalate_growth = st.slider("Escalate if growth ≥", 0.0, 1.0, 0.10, step=0.02)
        
        st.divider()
        
        # JSON Export/Import section
        st.subheader("💾 Data Management")
        
        if st.button("📥 Export Chat History", use_container_width=True):
            if "conversation_history" in st.session_state and st.session_state.conversation_history:
                json_data = export_conversation_history(st.session_state.conversation_history)
                st.download_button(
                    label="💾 Download JSON",
                    data=save_to_file(json_data, f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No chat history to export")
        
        # File uploader for importing
        # uploaded_file = st.file_uploader("📤 Import Chat History", type=['json'])
        # if uploaded_file is not None:
            # try:
                # json_content = uploaded_file.read().decode('utf-8')
                # imported_history = import_conversation_history(json_content)
                # st.session_state.conversation_history = imported_history
                # st.success(f"✅ Imported {len(imported_history)} messages")
                # st.rerun()
            # except Exception as e:
                # st.error(f"❌ Import failed: {e}")
        # 
        # st.caption("🔒 All data stays local - no server logging")

    # Initialize enhanced session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "risk_window" not in st.session_state:
        st.session_state.risk_window = deque(maxlen=th.window)
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    if "show_history" not in st.session_state:
        st.session_state.show_history = True
    if "show_analytics" not in st.session_state:
        st.session_state.show_analytics = False

    # If a previous action requested clearing the input, set it BEFORE widgets are created
    if st.session_state.clear_input:
        st.session_state["message_input"] = ""
        st.session_state.clear_input = False

    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("💬 Chat Interface")
            
            # Use a form for message input to get reliable submit behavior and auto-clear on submit
            with st.form("message_form", clear_on_submit=True):
                # Do NOT assign to st.session_state.message_input after widget creation.
                # Initialize default value (if missing) BEFORE creating the widget (done above).
                msg = st.text_area(
                    "Type your message here...",
                    height=120,
                    key="message_input",
                    placeholder="Try different types of content: harassment, hate speech, sexual content, dangerous instructions, crisis language..."
                )
                
                # Form buttons: send is the form submit, others can stay outside or be regular buttons
                send_submitted = st.form_submit_button("🚀 Send & Analyze")
            
            # Enhanced button layout for other actions (outside the form)
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 1, 1, 1])
            # with btn_col1:
            #     clear_chat = st.button("Clear", use_container_width=True)
            # with btn_col3:
            #     toggle_history = st.button(
            #         f"{'Hide' if st.session_state.show_history else 'Show'} History", 
            #         use_container_width=True
            #     )
            with btn_col2:
                # capture the click and toggle session state so the panel can be shown/hidden
                analytics_clicked = st.button("Analytics", use_container_width=True)
            if analytics_clicked:
                st.session_state.show_analytics = not st.session_state.show_analytics
            with btn_col4:
                # placeholder or other action
                pass
            with btn_col1:
                clear_chat = st.button("Clear", use_container_width=True)
            with btn_col3:
                toggle_history = st.button(
                    f"{'Hide' if st.session_state.show_history else 'Show'} History", 
                    use_container_width=True
                )


    with col2:
        # Real-time stats
        st.subheader("📊 Live Stats")
        total_messages = len(st.session_state.conversation_history)
        blocked_messages = sum(1 for entry in st.session_state.conversation_history if entry.decision.block)
        crisis_messages = sum(1 for entry in st.session_state.conversation_history if entry.decision.crisis)
        
        st.metric("Total Messages", total_messages)
        st.metric("Blocked", blocked_messages)
        st.metric("Crisis Detected", crisis_messages)
        
        if st.session_state.risk_window:
            avg_risk = sum(st.session_state.risk_window) / len(st.session_state.risk_window)
            st.metric("Avg Risk", f"{avg_risk:.2f}")
            st.progress(min(1.0, avg_risk))

        # Analytics panel (toggleable)
        if st.session_state.show_analytics:
            st.divider()
            st.subheader("📈 Analytics")
            # simple aggregate: counts per abuse flag
            flag_counts: Dict[str, int] = {}
            for e in st.session_state.conversation_history:
                for f in e.decision.abuse_flags:
                    flag_counts[f] = flag_counts.get(f, 0) + 1

            if flag_counts:
                st.write("Flag counts (all time):")
                st.bar_chart({k: v for k, v in flag_counts.items()})
            else:
                st.info("No flagged messages yet.")

    # Handle clear chat / toggle_history actions
    if clear_chat:
        st.session_state.conversation_history.clear()
        st.session_state.risk_window.clear()
        # Request the input widget be cleared on the next run (do NOT modify message_input now)
        st.session_state.clear_input = True
        if "current_input" in st.session_state:
            del st.session_state.current_input
        st.experimental_rerun()

    if toggle_history:
        st.session_state.show_history = not st.session_state.show_history

    # Process message when form submitted
    if send_submitted and msg and msg.strip():
        # save current input to session_state (optional)
        st.session_state.current_input = msg
        message_id = str(uuid.uuid4())[:8]
        
        with st.spinner("🔍 Analyzing message..."):
            try:
                cls = classify_with_gemini(client, msg)
                decision = make_decision(cls, th, user_age, st.session_state.risk_window, message_id)
                
                # Create conversation entry
                entry = ConversationEntry(
                    message_id=message_id,
                    role="user",
                    text=msg,
                    classification=cls,
                    decision=decision,
                    user_age=user_age
                )
                
                st.session_state.conversation_history.append(entry)
                # NO direct assignment to st.session_state["message_input"] here — form.clear_on_submit clears it
                st.success("✅ Message analyzed and saved")
            except Exception as e:
                st.error(f"❌ Error processing message: {e}")

    # Display conversation history (optional)
    if st.session_state.show_history and st.session_state.conversation_history:
        st.divider()
        st.subheader(f"💬 Conversation History ({len(st.session_state.conversation_history)} messages)")
        
        for entry in st.session_state.conversation_history[-20:]:  # Show last 20 messages
            with st.chat_message("user"):
                st.markdown(f"**Message:** {entry.text}")
                st.caption(f"🆔 ID: {entry.message_id} | 📅 {entry.timestamp[:19]} | 👤 Age: {entry.user_age}")
                
                # Enhanced safety analysis display
                with st.expander("🔍 Detailed Safety Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Classification Scores")
                        cls_dict = entry.classification.model_dump()
                        for key, value in cls_dict.items():
                            if key not in ['rationale', 'timestamp']:
                                if isinstance(value, (int, float)):
                                    color = "🔴" if value > 0.6 else "🟡" if value > 0.3 else "🟢"
                                    st.write(f"{color} **{key.title()}**: {value:.3f}")
                        st.write(f"💭 **Rationale**: {cls_dict['rationale']}")
                    
                    with col2:
                        st.subheader("⚖️ Decision Details")
                        decision_status = "🚫 BLOCKED" if entry.decision.block else "✅ ALLOWED"
                        st.write(f"**Status**: {decision_status}")
                        st.write(f"**Crisis**: {'🆘 YES' if entry.decision.crisis else '✅ NO'}")
                        st.write(f"**Guardian Filter**: {'👶 YES' if entry.decision.guardian_filtered else '✅ NO'}")
                        st.write(f"**Escalation**: {'⬆️ YES' if entry.decision.escalate_conversation else '✅ NO'}")
                        if entry.decision.abuse_flags:
                            st.write(f"**Flags**: {', '.join(entry.decision.abuse_flags)}")
                        st.caption(entry.decision.explanation)

            # Show blocking notifications
            if entry.decision.block:
                alert_type = "error" if entry.decision.crisis else "warning"
                with st.container(border=True):
                    if alert_type == "error":
                        st.error("🚨 **CRISIS DETECTED** - Immediate human intervention recommended")
                    else:
                        st.warning("⚠️ **MESSAGE BLOCKED** by safety system")
                    
                    if entry.decision.guardian_filtered:
                        st.info(f"👶 **Guardian Mode**: Content blocked for user age {entry.user_age}")
                        if entry.decision.guardian_reasons:
                            st.write("**Reasons**: " + ", ".join(entry.decision.guardian_reasons))
                    
                    if entry.decision.abuse_flags:
                        st.write("**Detected Categories**: " + ", ".join(f"🏷️ {flag}" for flag in entry.decision.abuse_flags))

    elif not st.session_state.show_history:
        st.info("History is hidden. Click 'Show History' to view past messages.")

if __name__ == "__main__":
    main()
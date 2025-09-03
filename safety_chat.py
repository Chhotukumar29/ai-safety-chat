"""
Project Title: AI Safety Chat - FastAPI Backend
====================================================

What this API provides
----------------------
- **Multi-category Abuse Detection** (harassment/hate/sexual/dangerous/crisis) via Gemini
- **RESTful endpoints** for message analysis and conversation management
- **JSON-based** request/response format
- **Enhanced Pattern Recognition** across rolling conversation window
- **Crisis Intervention** with escalation detection
- **Age-appropriate Content Filtering** (guardian mode)
- **Conversation History Management** with export/import
- **Privacy-by-default**: no persistent storage, session-based

Requirements
-----------
- Python 3.10+
- Packages: `pip install fastapi uvicorn google-genai pydantic python-dotenv`
- Set environment var `GEMINI_API_KEY` (or create a `.env` file with it)

Usage
-----
Run the server: `uvicorn safety_chat:app --reload --host 0.0.0.0 --port 8000`
API docs: http://localhost:8000/docs
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# --- Gemini SDK ---
# from google import genai
# from google.genai import types
import google.generativeai as genai
import google.generativeai.types as types

load_dotenv(override=False)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-2.5-flash')

# ----------------------------
# Configuration & Models
# ----------------------------

@dataclass
class SafetyThresholds:
    harassment: float = 0.45
    hate: float = 0.40
    sexual: float = 0.35
    dangerous: float = 0.40
    crisis_self_harm: float = 0.30
    window: int = 6
    escalate_avg_risk: float = 0.45
    escalate_growth: float = 0.10

class Classification(BaseModel):
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

# Request/Response Models
class MessageAnalysisRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Message text to analyze")
    user_age: int = Field(18, ge=5, le=120, description="User age for appropriate filtering")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Custom safety thresholds")

class MessageAnalysisResponse(BaseModel):
    message_id: str
    session_id: str
    classification: Classification
    decision: Decision
    processing_time_ms: float
    status: str

class ConversationHistoryResponse(BaseModel):
    session_id: str
    total_messages: int
    messages: List[ConversationEntry]
    statistics: Dict[str, Any]

class SessionStatsResponse(BaseModel):
    session_id: str
    total_messages: int
    blocked_messages: int
    crisis_messages: int
    escalated_messages: int
    guardian_filtered: int
    average_risk: float
    category_stats: Dict[str, Dict[str, float]]
    risk_trend: List[float]

class ThresholdUpdateRequest(BaseModel):
    harassment: Optional[float] = Field(None, ge=0, le=1)
    hate: Optional[float] = Field(None, ge=0, le=1)
    sexual: Optional[float] = Field(None, ge=0, le=1)
    dangerous: Optional[float] = Field(None, ge=0, le=1)
    crisis_self_harm: Optional[float] = Field(None, ge=0, le=1)
    window: Optional[int] = Field(None, ge=3, le=20)
    escalate_avg_risk: Optional[float] = Field(None, ge=0, le=1)
    escalate_growth: Optional[float] = Field(None, ge=0, le=1)

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    gemini_api_status: str

# ----------------------------
# Global Variables and Setup
# ----------------------------

# In-memory session storage (use Redis/Database in production)
sessions: Dict[str, Dict] = {}

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

Include a brief "rationale" (≤50 words) explaining your scoring.
Return ONLY valid JSON. No markdown, no explanation outside JSON.
"""

# ----------------------------
# Core Logic Functions
# ----------------------------

async def classify_with_gemini(model, text: str) -> Classification:
    try:
        response = model.generate_content(
            [
                ENHANCED_CLASSIFIER_PROMPT,
                f"Message to classify: {text}",
            ],
            generation_config=types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
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
        )
        raw = response.text.strip()
        data = Classification.model_validate_json(raw)
        return data
    except Exception as e:
        # Fallback classification
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
            rationale=f"Fallback classification due to API error: {str(e)[:30]}"
        )

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
        
        if (avg_risk >= thresholds.escalate_avg_risk or 
            growth >= thresholds.escalate_growth or
            cls.overall_risk > 0.8):
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

def get_or_create_session(session_id: str) -> Dict:
    """Get or create a session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "conversation_history": [],
            "risk_window": deque(maxlen=6),
            "thresholds": SafetyThresholds(),
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    else:
        sessions[session_id]["last_activity"] = datetime.now().isoformat()
    
    return sessions[session_id]

def calculate_session_stats(session: Dict) -> Dict[str, Any]:
    """Calculate statistics for a session"""
    history = session["conversation_history"]
    if not history:
        return {
            "total_messages": 0,
            "blocked_messages": 0,
            "crisis_messages": 0,
            "escalated_messages": 0,
            "guardian_filtered": 0,
            "average_risk": 0.0,
            "category_averages": {},
            "risk_trend": []
        }
    
    total_messages = len(history)
    blocked_messages = sum(1 for entry in history if entry["decision"]["block"])
    crisis_messages = sum(1 for entry in history if entry["decision"]["crisis"])
    escalated_messages = sum(1 for entry in history if entry["decision"]["escalate_conversation"])
    guardian_filtered = sum(1 for entry in history if entry["decision"]["guardian_filtered"])
    
    # Calculate category averages
    categories = ['harassment', 'hate', 'sexual', 'dangerous', 'self_harm', 'sentiment_negativity', 'overall_risk']
    category_averages = {}
    
    for category in categories:
        scores = [entry["classification"][category] for entry in history]
        category_averages[category] = {
            'average': sum(scores) / len(scores) if scores else 0,
            'maximum': max(scores) if scores else 0,
            'minimum': min(scores) if scores else 0
        }
    
    # Risk trend
    risk_trend = [entry["classification"]["overall_risk"] for entry in history]
    
    return {
        "total_messages": total_messages,
        "blocked_messages": blocked_messages,
        "crisis_messages": crisis_messages,
        "escalated_messages": escalated_messages,
        "guardian_filtered": guardian_filtered,
        "average_risk": sum(risk_trend) / len(risk_trend) if risk_trend else 0,
        "category_averages": category_averages,
        "risk_trend": risk_trend
    }

# ----------------------------
# FastAPI Application Setup
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        print("Gemini client initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")
    
    yield
    
    # Cleanup
    sessions.clear()
    print("Application shutdown complete")

app = FastAPI(
    title="AI Safety Chat API",
    description="Advanced content moderation API with multi-category abuse detection",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (optional)
security = HTTPBearer(auto_error=False)

async def get_optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement authentication if needed
    return credentials

# ----------------------------
# API Endpoints
# ----------------------------

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI Safety Chat API",
        "version": "2.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        _ = model  # Just check model is initialized
        gemini_status = "healthy"
    except Exception as e:
        gemini_status = f"error: {str(e)[:50]}"
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        gemini_api_status=gemini_status
    )

@app.post("/analyze", response_model=MessageAnalysisResponse)
async def analyze_message(
    request: MessageAnalysisRequest,
    background_tasks: BackgroundTasks,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Analyze a message for safety violations"""
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())[:8]
    
    # Get or create session
    session = get_or_create_session(session_id)
    
    # Update thresholds if provided
    thresholds = session["thresholds"]
    if request.thresholds:
        for key, value in request.thresholds.items():
            if hasattr(thresholds, key) and value is not None:
                setattr(thresholds, key, value)
    
    try:
        # Get Gemini client and classify message
        client = get_gemini_client()
        classification = await classify_with_gemini(model, request.text)
        
        # Make decision
        decision = make_decision(
            classification, 
            thresholds, 
            request.user_age, 
            session["risk_window"], 
            message_id
        )
        
        # Create conversation entry
        entry = {
            "message_id": message_id,
            "role": "user",
            "text": request.text,
            "classification": classification.model_dump(),
            "decision": decision.model_dump(),
            "user_age": request.user_age,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to session history
        session["conversation_history"].append(entry)
        
        # Handle crisis intervention in background
        if decision.crisis:
            background_tasks.add_task(handle_crisis_intervention, session_id, message_id, request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return MessageAnalysisResponse(
            message_id=message_id,
            session_id=session_id,
            classification=classification,
            decision=decision,
            processing_time_ms=processing_time,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    history = session["conversation_history"]
    
    # Apply pagination
    total_messages = len(history)
    paginated_history = history[offset:offset + limit]
    
    # Convert to ConversationEntry objects
    messages = []
    for entry in paginated_history:
        messages.append(ConversationEntry(
            message_id=entry["message_id"],
            role=entry["role"],
            text=entry["text"],
            classification=Classification(**entry["classification"]),
            decision=Decision(**entry["decision"]),
            user_age=entry["user_age"],
            timestamp=entry["timestamp"]
        ))
    
    statistics = calculate_session_stats(session)
    
    return ConversationHistoryResponse(
        session_id=session_id,
        total_messages=total_messages,
        messages=messages,
        statistics=statistics
    )

@app.get("/sessions/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(
    session_id: str,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Get detailed statistics for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    stats = calculate_session_stats(session)
    
    return SessionStatsResponse(
        session_id=session_id,
        total_messages=stats["total_messages"],
        blocked_messages=stats["blocked_messages"],
        crisis_messages=stats["crisis_messages"],
        escalated_messages=stats["escalated_messages"],
        guardian_filtered=stats["guardian_filtered"],
        average_risk=stats["average_risk"],
        category_stats=stats["category_averages"],
        risk_trend=stats["risk_trend"]
    )

@app.post("/sessions/{session_id}/export")
async def export_session_data(
    session_id: str,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Export session data as JSON"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    export_data = {
        "session_id": session_id,
        "export_timestamp": datetime.now().isoformat(),
        "conversation_history": session["conversation_history"],
        "statistics": calculate_session_stats(session)
    }
    
    filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/json"
        }
    )

# @app.post("/sessions/{session_id}/import")
# async def import_session_data(
#     session_id: str,
#     import_data: Dict[str, Any],
#     auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
# ):
#     """Import session data from JSON"""
#     try:
#         # Validate import data structure
#         if "conversation_history" not in import_data:
#             raise HTTPException(status_code=400, detail="Invalid import data: missing conversation_history")
        
#         # Create or update session
#         session = get_or_create_session(session_id)
#         session["conversation_history"] = import_data["conversation_history"]
        
#         # Rebuild risk window from history
#         session["risk_window"] = deque(maxlen=session["thresholds"].window)
#         for entry in session["conversation_history"]:
#             session["risk_window"].append(entry["classification"]["overall_risk"])
        
#         return {
#             "status": "success",
#             "message": f"Imported {len(import_data['conversation_history'])} messages",
#             "session_id": session_id
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.put("/sessions/{session_id}/thresholds", response_model=Dict[str, str])
async def update_session_thresholds(
    session_id: str,
    thresholds: ThresholdUpdateRequest,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Update safety thresholds for a session"""
    session = get_or_create_session(session_id)
    session_thresholds = session["thresholds"]
    
    # Update provided thresholds
    update_data = thresholds.model_dump(exclude_none=True)
    for key, value in update_data.items():
        if hasattr(session_thresholds, key):
            setattr(session_thresholds, key, value)
    
    # Update risk window size if changed
    if "window" in update_data:
        old_window = list(session["risk_window"])
        session["risk_window"] = deque(old_window, maxlen=update_data["window"])
    
    return {
        "status": "success",
        "message": "Thresholds updated successfully",
        "session_id": session_id
    }

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """Delete a session and all its data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    
    return {
        "status": "success",
        "message": f"Session {session_id} deleted successfully"
    }

@app.get("/sessions", response_model=Dict[str, Any])
async def list_sessions(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_optional_auth)
):
    """List all active sessions"""
    session_list = []
    
    for session_id, session in sessions.items():
        session_list.append({
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "message_count": len(session["conversation_history"])
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": session_list
    }

# ----------------------------
# Background Tasks
# ----------------------------

async def handle_crisis_intervention(session_id: str, message_id: str, message_text: str):
    """Handle crisis intervention - log and potentially notify moderators"""
    crisis_log = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "message_id": message_id,
        "message_preview": message_text[:100] + "..." if len(message_text) > 100 else message_text,
        "action": "crisis_detected",
        "intervention_triggered": True
    }
    
    # In production, you might:
    # - Log to a secure crisis intervention system
    # - Send notifications to human moderators
    # - Create support tickets
    # - Integrate with mental health resources
    
    print(f"CRISIS INTERVENTION: {json.dumps(crisis_log, indent=2)}")

# ----------------------------
# Error Handlers
# ----------------------------

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Request validation failed"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "message": exc.detail
        }
    )

# ----------------------------
# Development Server
# ----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "safety_chat:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
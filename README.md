# AI Safety Chat POC 🛡️

A real-time content moderation system using Google's Gemini AI for detecting harmful content, crisis situations, and implementing age-appropriate filtering in chat applications.

## 🎯 Overview

This proof-of-concept demonstrates advanced AI safety measures including:
- **Multi-category Content Classification**: Harassment, hate speech, sexual content, dangerous content, and self-harm detection
- **Crisis Intervention System**: Real-time detection and escalation of mental health crises
- **Age-Appropriate Guardian Mode**: Dynamic filtering based on user age
- **Escalation Pattern Recognition**: Rolling window analysis to detect conversation deterioration
- **Transparent Decision Making**: Explainable AI with detailed reasoning for each moderation decision

## 🏗️ Architecture & Approach

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Gemini AI      │───▶│  Rule Engine    │
│                 │    │  Classifier     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │  JSON Scores    │    │  Decision +     │
                    │  (0.0 - 1.0)    │    │  Explanation    │
                    └─────────────────┘    └─────────────────┘
```

### Design Decisions

1. **Hybrid AI + Rule-Based Approach**: 
   - Gemini AI provides nuanced content understanding
   - Deterministic rules ensure consistent, auditable decisions
   - Combines flexibility with transparency

2. **Structured JSON Output**: 
   - Reliable parsing with Pydantic validation
   - Consistent scoring across categories (0.0-1.0 scale)
   - Built-in fallback for parsing failures

3. **Multi-Dimensional Safety Assessment**:
   - Individual category thresholds
   - Composite risk scoring
   - Age-specific guardian filtering
   - Temporal pattern analysis

4. **Privacy-First Design**:
   - No server-side logging
   - In-memory processing only
   - Local session state management

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Google AI API key (Gemini)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-safety-chat
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install streamlit google-genai pydantic python-dotenv
```

4. **Set up environment variables**:
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

5. **Run the application**:
```bash
streamlit run safety_chat_poc.py
```

6. **Open your browser** to `http://localhost:8501`

## 🎮 How to Use the Application

### Basic Usage

1. **Set User Age**: Use the sidebar to set the user's age (affects guardian filtering)
2. **Type Message**: Enter text in the chat input area
3. **Send & Moderate**: Click to process the message through safety filters
4. **Review Analysis**: Expand "Show safety analysis" to see detailed scores and reasoning

### Understanding the Interface

#### Main Chat Area
- **User Messages**: Your input messages with safety analysis
- **Safety Scores**: JSON output showing classification scores (0.0-1.0)
- **Decision Summary**: BLOCKED/ALLOW status with crisis and escalation indicators
- **Detailed Explanation**: Rule-based reasoning for the decision

#### Sidebar Controls
- **User Age**: Affects guardian filtering thresholds
- **Safety Thresholds**: Fine-tune sensitivity for each category
- **Escalation Settings**: Configure pattern detection parameters

#### Moderator Console
- **Rolling Average Risk**: Real-time risk trend analysis
- **Progress Bar**: Visual representation of conversation risk level

### Sample Test Cases

Try these inputs to see the system in action:

**Safe Content**:
```
"Hello, how are you today?"
"I'm working on a school project about renewable energy"
```

**Harassment Detection**:
```
"You're such an idiot, I hate you"
"Go kill yourself, nobody likes you"
```

**Crisis Detection**:
```
"I don't want to live anymore"
"I'm thinking about ending it all"
```

**Age-Inappropriate Content**:
```
"Let's talk about adult topics" (with age set to 12)
```

## ⚙️ Threshold Management

### Understanding Thresholds

Each category has a threshold value (0.0-1.0):
- **0.0**: Most permissive (allows almost everything)
- **1.0**: Most restrictive (blocks almost everything)
- **Default values**: Balanced for general use

### Category-Specific Thresholds

#### Harassment (Default: 0.55)
- Detects aggressive language, bullying, personal attacks
- **Lower values**: Catch mild aggression
- **Higher values**: Only severe harassment

#### Hate Speech (Default: 0.55)
- Identifies discrimination against protected groups
- **Lower values**: Sensitive to subtle bias
- **Higher values**: Only explicit hate speech

#### Sexual Content (Default: 0.50)
- Flags sexual or adult content
- **Lower values**: Conservative filtering
- **Higher values**: Permissive approach

#### Dangerous Content (Default: 0.50)
- Catches instructions for harmful activities
- **Lower values**: Broad safety net
- **Higher values**: Only explicit danger

#### Self-Harm/Crisis (Default: 0.50)
- Detects suicide ideation and crisis situations
- **Lower values**: Sensitive crisis detection
- **Higher values**: Only explicit self-harm

### Escalation Settings

#### Window Size (Default: 6 messages)
- Number of recent messages to analyze for patterns
- **Smaller windows**: React quickly to changes
- **Larger windows**: More stable, less reactive

#### Average Risk Threshold (Default: 0.50)
- Triggers escalation when average risk exceeds threshold
- **Lower values**: Earlier escalation
- **Higher values**: Later escalation

#### Growth Threshold (Default: 0.12)
- Escalates when risk increases significantly across window
- **Lower values**: Sensitive to small increases
- **Higher values**: Only major escalations

### Age-Based Guardian Filtering

The system automatically applies stricter thresholds for minors:

**Ages 5-12**:
- Sexual content: > 0.15
- Harassment: > 0.35  
- Dangerous: > 0.20

**Ages 13-17**:
- Sexual content: > 0.25
- Harassment: > 0.45
- Dangerous: > 0.30

**Ages 18+**:
- Uses standard thresholds

## 🔧 Technical Implementation

### Model Integration

```python
# Gemini API call with safety settings
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[CLASSIFIER_SYSTEM_PROMPT, text],
    config=types.GenerateContentConfig(
        safety_settings=[...],  # Explicit safety configuration
        response_mime_type="application/json"  # Structured output
    )
)
```

### Decision Flow

```python
def make_decision(classification, thresholds, user_age, history):
    # 1. Check category thresholds
    abuse_flags = check_abuse_categories(classification, thresholds)
    
    # 2. Crisis detection
    crisis = classification.self_harm >= thresholds.crisis_self_harm
    
    # 3. Age-appropriate filtering
    guardian_filtered = apply_guardian_rules(classification, user_age)
    
    # 4. Escalation pattern analysis
    escalate = analyze_conversation_trend(history, thresholds)
    
    # 5. Final blocking decision
    block = bool(abuse_flags) or crisis or guardian_filtered
    
    return Decision(...)
```

### Data Flow

1. **Input**: User message string
2. **Classification**: Gemini API returns JSON scores
3. **Validation**: Pydantic ensures data integrity
4. **Decision**: Rule engine applies thresholds and policies
5. **History**: Rolling window tracks conversation patterns
6. **Output**: Block/allow decision with detailed explanation

## 📊 Evaluation & Metrics

### Classification Accuracy
- **Precision**: Correctly identified harmful content / Total flagged content
- **Recall**: Correctly identified harmful content / Total harmful content
- **F1-Score**: Harmonic mean of precision and recall

### Response Time
- **Target**: < 2 seconds for real-time chat
- **Measurement**: End-to-end processing time

### False Positive/Negative Rates
- **False Positives**: Safe content incorrectly blocked
- **False Negatives**: Harmful content incorrectly allowed

## 🔒 Privacy & Security

### Data Protection
- **No Persistent Storage**: All data exists only in memory
- **No Logging**: Messages are not saved or transmitted to servers
- **Local Processing**: Streamlit session state keeps data client-side

### API Security
- **Environment Variables**: API keys stored securely
- **Rate Limiting**: Built-in protection against abuse
- **Error Handling**: Graceful degradation on API failures

## 🚀 Production Considerations

### Scalability
- **Microservices Architecture**: Separate classification and decision services
- **Caching**: Redis for frequent classifications
- **Load Balancing**: Distribute API calls across regions

### Monitoring
- **Metrics Dashboard**: Real-time classification statistics
- **Alert System**: Automated escalation for crisis detection
- **Audit Logs**: Compliance and debugging capabilities

### Continuous Improvement
- **Feedback Loop**: Human moderator corrections
- **Model Fine-tuning**: Regular updates based on new data
- **Threshold Optimization**: A/B testing for optimal settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Crisis Resources

If you or someone you know is in crisis:
- **Emergency**: Call your local emergency number
- **India**: AASRA 91-22-27546669 (24/7), Kiran Helpline 1800-599-0019
- **US**: National Suicide Prevention Lifeline 988
- **UK**: Samaritans 116 123

## 📞 Support

For technical issues or questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error logs and reproduction steps

---

**Built with ❤️ for safer online communities**
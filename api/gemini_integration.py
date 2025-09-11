"""Gemini AI Integration for MPS Connect System.

This module provides integration with Google's Gemini AI models for:
1. Case Analysis & Classification
2. Letter Generation
3. Approval Workflow & Recommendations

Uses Gemini Flash for quick previews and Gemini Pro for final outputs.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)


@dataclass
class CaseAnalysis:
    """Structured case analysis result."""

    categories: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    key_facts: List[str]
    priority_level: str
    recommended_actions: List[str]
    reasoning: str


@dataclass
class LetterDraft:
    """Letter generation result."""

    subject: str
    content: str
    tone: str
    key_points: List[str]
    suggested_improvements: List[str]
    confidence: float


@dataclass
class ApprovalRecommendation:
    """Approval workflow recommendation."""

    recommendation: str  # "approve", "manual_review", "reject"
    confidence: float
    reasoning: str
    risk_factors: List[str]
    suggested_conditions: List[str]
    next_steps: List[str]


class GeminiIntegration:
    """Main class for Gemini AI integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini integration."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not available. Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=self.api_key)

        # Initialize models
        self.flash_model = genai.GenerativeModel("gemini-1.5-flash")
        self.pro_model = genai.GenerativeModel("gemini-1.5-pro")

        # Safety settings for both models
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logger.info("Gemini integration initialized successfully")

        # Lightweight in-memory cache (TTL-based)
        # Key: str, Value: (expires_at_ts, data)
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl_seconds: int = 120
        self._cache_max_entries: int = 256

    def _cache_make_key(self, scope: str, payload: Any) -> str:
        try:
            ser = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            ser = str(payload)
        return f"{scope}:{hash(ser)}"

    def _cache_get(self, key: str) -> Optional[Any]:
        import time

        item = self._cache.get(key)
        if not item:
            return None
        expires_at, data = item
        if time.time() > expires_at:
            self._cache.pop(key, None)
            return None
        return data

    def _cache_set(self, key: str, data: Any) -> None:
        import time

        if len(self._cache) >= self._cache_max_entries:
            items = sorted(self._cache.items(), key=lambda kv: kv[1][0])
            for k, _ in items[: len(items) // 2 or 1]:
                self._cache.pop(k, None)
        self._cache[key] = (time.time() + self._cache_ttl_seconds, data)

    async def analyze_case_preview(self, case_text: str) -> CaseAnalysis:
        """Quick case analysis using Gemini Flash."""
        prompt = f"""
        Analyze this MP case for quick preview. Extract key information and provide initial classification.
        
        Case Text: {case_text}
        
        Please provide:
        1. Main categories (max 3)
        2. Key facts extracted
        3. Initial priority level (LOW/MEDIUM/HIGH/URGENT)
        4. Brief reasoning
        
        Format as JSON with keys: categories, key_facts, priority_level, reasoning
        """

        try:
            response = await self._generate_async(self.flash_model, prompt)
            return self._parse_case_analysis(response)
        except Exception as e:
            logger.error(f"Error in case analysis preview: {e}")
            return self._fallback_case_analysis()

    async def analyze_case_final(
        self, case_text: str, feedback: Optional[str] = None
    ) -> CaseAnalysis:
        """Detailed case analysis using Gemini Pro."""
        prompt = f"""
        Perform detailed analysis of this MP case for final classification.
        
        Case Text: {case_text}
        
        {f"Additional context/feedback: {feedback}" if feedback else ""}
        
        Please provide comprehensive analysis including:
        1. Detailed categories with confidence scores
        2. All key facts and details
        3. Priority assessment with reasoning
        4. Recommended actions
        5. Risk factors and considerations
        
        Format as JSON with keys: categories, confidence_scores, key_facts, priority_level, recommended_actions, reasoning
        """

        try:
            response = await self._generate_async(self.pro_model, prompt)
            return self._parse_case_analysis(response)
        except Exception as e:
            logger.error(f"Error in case analysis final: {e}")
            return self._fallback_case_analysis()

    async def generate_letter_preview(self, case_data: Dict[str, Any]) -> LetterDraft:
        """Quick letter draft using Gemini Flash."""
        prompt = f"""
        Generate a quick draft of an MP letter for this case.
        
        Case Data: {json.dumps(case_data, indent=2)}
        
        Create a brief, professional letter draft including:
        1. Appropriate subject line
        2. Key points to address
        3. Professional tone
        4. Basic structure
        
        Format as JSON with keys: subject, content, tone, key_points, suggested_improvements, confidence
        """

        try:
            response = await self._generate_async(self.flash_model, prompt)
            return self._parse_letter_draft(response)
        except Exception as e:
            logger.error(f"Error in letter generation preview: {e}")
            return self._fallback_letter_draft()

    async def generate_letter_final(
        self, case_data: Dict[str, Any], feedback: Optional[str] = None
    ) -> LetterDraft:
        """Polished letter using Gemini Pro."""
        prompt = f"""
        Generate a professional, polished MP letter for this case.
        
        Case Data: {json.dumps(case_data, indent=2)}
        
        {f"User feedback and requirements: {feedback}" if feedback else ""}
        
        Create a comprehensive, professional letter including:
        1. Compelling subject line
        2. Well-structured content with proper sections
        3. Appropriate tone for the situation
        4. All key points clearly addressed
        5. Professional formatting
        6. Call to action
        
        Ensure accuracy in all details (especially numbers, dates, amounts).
        
        Format as JSON with keys: subject, content, tone, key_points, suggested_improvements, confidence
        """

        try:
            response = await self._generate_async(self.pro_model, prompt)
            return self._parse_letter_draft(response)
        except Exception as e:
            logger.error(f"Error in letter generation final: {e}")
            return self._fallback_letter_draft()

    async def recommend_approval_preview(
        self, case_analysis: CaseAnalysis
    ) -> ApprovalRecommendation:
        """Quick approval recommendation using Gemini Flash."""
        prompt = f"""
        Provide quick approval recommendation for this case analysis.
        
        Case Analysis: {json.dumps(case_analysis.__dict__, indent=2)}
        
        Recommend: APPROVE, MANUAL_REVIEW, or REJECT
        Provide brief reasoning and key factors.
        
        Format as JSON with keys: recommendation, confidence, reasoning, risk_factors, suggested_conditions, next_steps
        """

        try:
            response = await self._generate_async(self.flash_model, prompt)
            return self._parse_approval_recommendation(response)
        except Exception as e:
            logger.error(f"Error in approval recommendation preview: {e}")
            return self._fallback_approval_recommendation()

    async def recommend_approval_final(
        self, case_analysis: CaseAnalysis, feedback: Optional[str] = None
    ) -> ApprovalRecommendation:
        """Detailed approval recommendation using Gemini Pro."""
        prompt = f"""
        Provide comprehensive approval recommendation for this case analysis.
        
        Case Analysis: {json.dumps(case_analysis.__dict__, indent=2)}
        
        {f"Additional context/feedback: {feedback}" if feedback else ""}
        
        Provide detailed analysis including:
        1. Clear recommendation (APPROVE/MANUAL_REVIEW/REJECT)
        2. Confidence level and reasoning
        3. Risk factors and considerations
        4. Suggested conditions if applicable
        5. Next steps and follow-up actions
        
        Format as JSON with keys: recommendation, confidence, reasoning, risk_factors, suggested_conditions, next_steps
        """

        try:
            response = await self._generate_async(self.pro_model, prompt)
            return self._parse_approval_recommendation(response)
        except Exception as e:
            logger.error(f"Error in approval recommendation final: {e}")
            return self._fallback_approval_recommendation()

    async def _generate_async(self, model, prompt: str) -> str:
        """Generate content asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt, safety_settings=self.safety_settings
            ),
        )
        return str(response.text)

    def _parse_case_analysis(self, response: str) -> CaseAnalysis:
        """Parse case analysis response."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            return CaseAnalysis(
                categories=data.get("categories", []),
                confidence_scores=data.get("confidence_scores", {}),
                key_facts=data.get("key_facts", []),
                priority_level=data.get("priority_level", "MEDIUM"),
                recommended_actions=data.get("recommended_actions", []),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Error parsing case analysis: {e}")
            return self._fallback_case_analysis()

    def _parse_letter_draft(self, response: str) -> LetterDraft:
        """Parse letter draft response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            return LetterDraft(
                subject=data.get("subject", ""),
                content=data.get("content", ""),
                tone=data.get("tone", "professional"),
                key_points=data.get("key_points", []),
                suggested_improvements=data.get("suggested_improvements", []),
                confidence=data.get("confidence", 0.5),
            )
        except Exception as e:
            logger.error(f"Error parsing letter draft: {e}")
            return self._fallback_letter_draft()

    def _parse_approval_recommendation(self, response: str) -> ApprovalRecommendation:
        """Parse approval recommendation response."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            return ApprovalRecommendation(
                recommendation=data.get("recommendation", "MANUAL_REVIEW"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                risk_factors=data.get("risk_factors", []),
                suggested_conditions=data.get("suggested_conditions", []),
                next_steps=data.get("next_steps", []),
            )
        except Exception as e:
            logger.error(f"Error parsing approval recommendation: {e}")
            return self._fallback_approval_recommendation()

    def _fallback_case_analysis(self) -> CaseAnalysis:
        """Fallback case analysis when Gemini fails."""
        return CaseAnalysis(
            categories=[{"label": "General Inquiry", "score": 0.5}],
            confidence_scores={"General Inquiry": 0.5},
            key_facts=["Case requires manual review"],
            priority_level="MEDIUM",
            recommended_actions=["Manual review required"],
            reasoning="Fallback analysis due to processing error",
        )

    # LLM-Guided Chat Methods
    async def start_guided_chat(
        self,
        session_id: str,
        initial_message: str,
        language: str = "en",
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Start a new LLM-guided chat session with adaptive questioning."""
        try:
            # Detect language if not specified
            detected_lang = await self._detect_language(initial_message)
            if not language or language == "auto":
                language = detected_lang

            # Analyze initial message for intent and context
            intent_analysis = await self._analyze_intent(initial_message, language)
            # Municipal nuisance hinting: bias toward municipal/noise if keywords present
            low = initial_message.lower()
            if any(
                k in low
                for k in [
                    "noise",
                    "pickleball",
                    "pickle ball",
                    "badminton",
                    "court",
                    "town council",
                    "tc ",
                    " nea ",
                    "disturb",
                    "nuisance",
                ]
            ):
                intent_analysis.setdefault("category", "municipal")
                intent_analysis.setdefault("intent", "complaint")

            # Generate first adaptive question
            first_question = await self._generate_adaptive_question(
                initial_message, intent_analysis, language
            )

            # Extract initial facts
            facts = await self._extract_facts(initial_message, language)

            # Suggest initial agencies
            agencies = await self._suggest_agencies(intent_analysis, facts)

            # Produce archival English summary regardless of input language
            archival_english = await self._to_archival_english(initial_message)

            # Try to anticipate missing facts at start (lightweight heuristic prompt)
            missing = await self._anticipate_missing_facts(intent_analysis, facts)

            return {
                "success": True,
                "message": first_question,
                "language": language,
                "confidence": intent_analysis.get("confidence", 0.7),
                "facts_extracted": facts,
                "suggested_agencies": agencies,
                "next_question": None,
                "is_complete": False,
                "archival_english": archival_english,
                "missing_facts": missing,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error starting guided chat: {e}")
            return {
                "success": False,
                "message": "I'm sorry, I couldn't start our conversation. Please try again.",
                "language": language,
                "confidence": 0.0,
                "facts_extracted": [],
                "suggested_agencies": [],
                "next_question": None,
                "is_complete": False,
                "error": str(e),
            }

    async def continue_guided_chat(
        self,
        session_id: str,
        message: str,
        language: str = "en",
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Continue chat with adaptive questioning based on conversation flow."""
        try:
            # Analyze current message
            intent_analysis = await self._analyze_intent(message, language)
            low = message.lower()
            if any(
                k in low
                for k in [
                    "noise",
                    "pickleball",
                    "pickle ball",
                    "badminton",
                    "court",
                    "town council",
                    "tc ",
                    " nea ",
                    "disturb",
                    "nuisance",
                ]
            ):
                intent_analysis.setdefault("category", "municipal")
                intent_analysis.setdefault("intent", "complaint")

            # Extract new facts from current message
            new_facts = await self._extract_facts(message, language)

            # Update context with new information
            updated_context = context or {}
            updated_context.update(intent_analysis)

            # Determine if we have enough information
            completeness = await self._assess_completeness(updated_context, new_facts)

            archival_english = await self._to_archival_english(message)

            if completeness["is_complete"]:
                # Generate final summary and next steps
                final_message = await self._generate_completion_message(
                    updated_context, new_facts, language
                )
                return {
                    "success": True,
                    "message": final_message,
                    "language": language,
                    "confidence": completeness["confidence"],
                    "facts_extracted": new_facts,
                    "suggested_agencies": completeness["agencies"],
                    "next_question": None,
                    "is_complete": True,
                    "archival_english": archival_english,
                    "missing_facts": completeness.get("missing", []),
                    "error": None,
                }
            else:
                # Generate next adaptive question
                next_question = await self._generate_next_question(
                    updated_context, new_facts, language
                )
                return {
                    "success": True,
                    "message": next_question,
                    "language": language,
                    "confidence": completeness["confidence"],
                    "facts_extracted": new_facts,
                    "suggested_agencies": completeness["agencies"],
                    "next_question": next_question,
                    "is_complete": False,
                    "archival_english": archival_english,
                    "missing_facts": completeness.get("missing", []),
                    "error": None,
                }
        except Exception as e:
            logger.error(f"Error continuing guided chat: {e}")
            return {
                "success": False,
                "message": "I'm sorry, I couldn't process your message. Please try again.",
                "language": language,
                "confidence": 0.0,
                "facts_extracted": [],
                "suggested_agencies": [],
                "next_question": None,
                "is_complete": False,
                "error": str(e),
            }

    async def review_facts_checklist(
        self,
        session_id: str,
        facts: List[Dict[str, Any]],
        corrections: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Review and validate extracted facts before letter generation."""
        try:
            # Apply corrections if provided
            corrected_facts = facts.copy()
            if corrections:
                for fact_id, correction in corrections.items():
                    for fact in corrected_facts:
                        if fact.get("id") == fact_id:
                            fact["content"] = correction
                            fact["corrected"] = True

            # Validate facts with LLM
            validation = await self._validate_facts(corrected_facts)

            # Check if ready for letter generation
            readiness = await self._assess_letter_readiness(corrected_facts, validation)

            return {
                "success": True,
                "reviewed_facts": corrected_facts,
                "confidence": validation["confidence"],
                "ready_for_letter": readiness["ready"],
                "missing_facts": readiness["missing"],
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error reviewing facts checklist: {e}")
            return {
                "success": False,
                "reviewed_facts": facts,
                "confidence": 0.0,
                "ready_for_letter": False,
                "missing_facts": [],
                "error": str(e),
            }

    async def generate_letter_from_chat(
        self, session_id: str, language: str = "en", context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate professional letter from completed chat session."""
        try:
            # Generate letter using the pro model for final output
            letter = await self.generate_letter_final(
                case_data=context or {}, feedback=None
            )

            return {
                "subject": letter.subject,
                "content": letter.content,
                "tone": letter.tone,
                "key_points": letter.key_points,
                "suggested_improvements": letter.suggested_improvements,
                "confidence": letter.confidence,
                "language": language,
            }
        except Exception as e:
            logger.error(f"Error generating letter from chat: {e}")
            return {
                "subject": "Error",
                "content": "Unable to generate letter. Please try again.",
                "tone": "neutral",
                "key_points": [],
                "suggested_improvements": [],
                "confidence": 0.0,
                "language": language,
            }

    # Helper methods for LLM-guided chat
    async def _detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            prompt = f"""
            Detect the language of this text and return only the language code (en, zh, ms, ta):
            Text: {text}
            
            Return only the language code.
            """

            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            lang_code = response.text.strip().lower()
            if lang_code in ["en", "zh", "ms", "ta"]:
                return lang_code
            return "en"  # Default to English
        except Exception:
            return "en"

    async def _analyze_intent(self, message: str, language: str) -> Dict[str, Any]:
        """Analyze user intent and extract context."""
        try:
            lang_prompts = {
                "en": f"""
                Analyze this message for intent and context. Return JSON with:
                - intent: main purpose (complaint, request, inquiry, etc.)
                - urgency: low/medium/high
                - category: municipal, traffic, housing, etc.
                - confidence: 0.0-1.0
                - key_concerns: list of main issues
                
                Message: {message}
                """,
                "zh": f"""
                分析这条消息的意图和上下文。返回JSON格式：
                - intent: 主要目的（投诉、请求、询问等）
                - urgency: low/medium/high
                - category: municipal, traffic, housing等
                - confidence: 0.0-1.0
                - key_concerns: 主要问题列表
                
                消息: {message}
                """,
                "ms": f"""
                Analisis mesej ini untuk niat dan konteks. Kembalikan JSON dengan:
                - intent: tujuan utama (aduan, permintaan, pertanyaan, dll)
                - urgency: low/medium/high
                - category: municipal, traffic, housing, dll
                - confidence: 0.0-1.0
                - key_concerns: senarai isu utama
                
                Mesej: {message}
                """,
                "ta": f"""
                இந்த செய்தியின் நோக்கம் மற்றும் சூழலை பகுப்பாய்வு செய்யுங்கள். JSON வடிவத்தில் திரும்பவும்:
                - intent: முக்கிய நோக்கம் (புகார், கோரிக்கை, வினா, முதலியன)
                - urgency: low/medium/high
                - category: municipal, traffic, housing, முதலியன
                - confidence: 0.0-1.0
                - key_concerns: முக்கிய பிரச்சினைகளின் பட்டியல்
                
                செய்தி: {message}
                """,
            }

            prompt = lang_prompts.get(language, lang_prompts["en"])
            cache_key = self._cache_make_key("intent", {"m": message, "l": language})
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            # Parse JSON response
            import json

            data = json.loads(response.text)
            self._cache_set(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return {
                "intent": "unknown",
                "urgency": "medium",
                "category": "general",
                "confidence": 0.5,
                "key_concerns": [],
            }

    async def _extract_facts(self, message: str, language: str) -> List[Dict[str, Any]]:
        """Extract structured facts from message."""
        try:
            lang_prompts = {
                "en": f"""
                Extract structured facts from this message. Return JSON array with:
                - id: unique identifier
                - type: fact type (location, time, person, issue, etc.)
                - content: fact content
                - confidence: 0.0-1.0
                
                Message: {message}
                """,
                "zh": f"""
                从这条消息中提取结构化事实。返回JSON数组：
                - id: 唯一标识符
                - type: 事实类型（地点、时间、人员、问题等）
                - content: 事实内容
                - confidence: 0.0-1.0
                
                消息: {message}
                """,
                "ms": f"""
                Ekstrak fakta berstruktur dari mesej ini. Kembalikan array JSON dengan:
                - id: pengenal unik
                - type: jenis fakta (lokasi, masa, orang, isu, dll)
                - content: kandungan fakta
                - confidence: 0.0-1.0
                
                Mesej: {message}
                """,
                "ta": f"""
                இந்த செய்தியிலிருந்து கட்டமைக்கப்பட்ட உண்மைகளை பிரித்தெடுக்கவும். JSON வரிசையை திரும்பவும்:
                - id: தனித்துவமான அடையாளம்
                - type: உண்மை வகை (இடம், நேரம், நபர், பிரச்சினை, முதலியன)
                - content: உண்மை உள்ளடக்கம்
                - confidence: 0.0-1.0
                
                செய்தி: {message}
                """,
            }

            prompt = lang_prompts.get(language, lang_prompts["en"])
            cache_key = self._cache_make_key("facts", {"m": message, "l": language})
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            import json

            data = json.loads(response.text)
            if isinstance(data, list):
                self._cache_set(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []

    async def _suggest_agencies(
        self, intent_analysis: Dict[str, Any], facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest appropriate government agencies based on intent and facts."""
        try:
            prompt = f"""
            Based on this intent analysis and facts, suggest appropriate Singapore government agencies:
            
            Intent: {intent_analysis.get('intent', 'unknown')}
            Category: {intent_analysis.get('category', 'general')}
            Urgency: {intent_analysis.get('urgency', 'medium')}
            Facts: {facts}
            
            Return JSON array with:
            - agency: agency name
            - reason: why this agency is relevant
            - priority: high/medium/low
            - contact_info: basic contact information
            - note: concise routing note if municipal noise (e.g., Town Council vs NEA)
            """

            cache_key = self._cache_make_key(
                "agencies", {"i": intent_analysis, "f": facts}
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            import json

            data = json.loads(response.text)
            if isinstance(data, list):
                self._cache_set(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error suggesting agencies: {e}")
            return []

    async def _to_archival_english(self, text: str) -> str:
        """Create a concise formal English archival summary of user text."""
        try:
            prompt = f"""
            Rephrase the following into concise, formal English suitable for audit archiving. Preserve facts and intent.
            Text: {text}
            """
            cache_key = self._cache_make_key("archival", {"t": text})
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
            response = await self.flash_model.generate_content_async(
                prompt,
                safety_settings=self.safety_settings,
            )
            out = response.text.strip()
            self._cache_set(cache_key, out)
            return out
        except Exception:
            return text

    async def _generate_adaptive_question(
        self, initial_message: str, intent_analysis: Dict[str, Any], language: str
    ) -> str:
        """Generate first adaptive question based on initial message analysis."""
        try:
            lang_prompts = {
                "en": f"""
                Generate a warm, empathetic first question for a Singaporean MP case assistant.
                Based on this initial message: "{initial_message}"
                Intent: {intent_analysis.get('intent', 'unknown')}
                Category: {intent_analysis.get('category', 'general')}
                
                Be respectful, use appropriate Singaporean addressing, and ask for the most important missing information.
                """,
                "zh": f"""
                为新加坡议员案例助手生成一个温暖、富有同理心的第一个问题。
                基于这个初始消息: "{initial_message}"
                意图: {intent_analysis.get('intent', 'unknown')}
                类别: {intent_analysis.get('category', 'general')}
                
                要尊重，使用适当的新加坡称呼方式，询问最重要的缺失信息。
                """,
                "ms": f"""
                Hasilkan soalan pertama yang mesra dan empati untuk pembantu kes MP Singapura.
                Berdasarkan mesej awal ini: "{initial_message}"
                Niat: {intent_analysis.get('intent', 'unknown')}
                Kategori: {intent_analysis.get('category', 'general')}
                
                Bersikap hormat, gunakan panggilan Singapura yang sesuai, dan tanya maklumat penting yang hilang.
                """,
                "ta": f"""
                சிங்கப்பூர் எம்பி வழக்கு உதவியாளருக்கு வெப்பமான, பச்சாதாபமான முதல் கேள்வியை உருவாக்குங்கள்.
                இந்த ஆரம்ப செய்தியின் அடிப்படையில்: "{initial_message}"
                நோக்கம்: {intent_analysis.get('intent', 'unknown')}
                வகை: {intent_analysis.get('category', 'general')}
                
                மரியாதையாக இருங்கள், பொருத்தமான சிங்கப்பூர் முகவரியிடல் பயன்படுத்துங்கள், மற்றும் மிக முக்கியமான காணாமல் போன தகவலைக் கேளுங்கள்.
                """,
            }

            prompt = lang_prompts.get(language, lang_prompts["en"])
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating adaptive question: {e}")
            return "I understand you need help. Could you tell me more about your situation?"

    async def _generate_next_question(
        self, context: Dict[str, Any], facts: List[Dict[str, Any]], language: str
    ) -> str:
        """Generate next adaptive question based on conversation context."""
        try:
            lang_prompts = {
                "en": f"""
                Generate the next question for a Singaporean MP case assistant.
                Context: {context}
                Current facts: {facts}
                
                Ask for the most important missing information to help with their case.
                Be warm, empathetic, and use appropriate Singaporean addressing.
                """,
                "zh": f"""
                为新加坡议员案例助手生成下一个问题。
                上下文: {context}
                当前事实: {facts}
                
                询问最重要的缺失信息来帮助他们的案例。
                要温暖、富有同理心，使用适当的新加坡称呼方式。
                """,
                "ms": f"""
                Hasilkan soalan seterusnya untuk pembantu kes MP Singapura.
                Konteks: {context}
                Fakta semasa: {facts}
                
                Tanya maklumat penting yang hilang untuk membantu kes mereka.
                Bersikap mesra, empati, dan gunakan panggilan Singapura yang sesuai.
                """,
                "ta": f"""
                சிங்கப்பூர் எம்பி வழக்கு உதவியாளருக்கு அடுத்த கேள்வியை உருவாக்குங்கள்.
                சூழல்: {context}
                தற்போதைய உண்மைகள்: {facts}
                
                அவர்களின் வழக்குக்கு உதவ மிக முக்கியமான காணாமல் போன தகவலைக் கேளுங்கள்.
                வெப்பமாக, பச்சாதாபமாக இருங்கள், மற்றும் பொருத்தமான சிங்கப்பூர் முகவரியிடல் பயன்படுத்துங்கள்.
                """,
            }

            prompt = lang_prompts.get(language, lang_prompts["en"])
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating next question: {e}")
            return "Could you provide more details about your situation?"

    async def _assess_completeness(
        self, context: Dict[str, Any], facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess if we have enough information to proceed with letter generation."""
        try:
            prompt = f"""
            Assess if we have enough information to generate a proper MP letter.
            Context: {context}
            Facts: {facts}
            
            Return JSON with:
            - is_complete: true/false
            - confidence: 0.0-1.0
            - missing: list of missing information
            - agencies: suggested agencies with priorities
            """

            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            import json

            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error assessing completeness: {e}")
            return {
                "is_complete": False,
                "confidence": 0.5,
                "missing": ["More information needed"],
                "agencies": [],
            }

    async def _generate_completion_message(
        self, context: Dict[str, Any], facts: List[Dict[str, Any]], language: str
    ) -> str:
        """Generate completion message when enough information is gathered."""
        try:
            lang_prompts = {
                "en": f"""
                Generate a completion message for a Singaporean MP case assistant.
                We have gathered enough information. Thank the user and explain next steps.
                Context: {context}
                Facts: {facts}
                
                Be warm, professional, and reassuring.
                """,
                "zh": f"""
                为新加坡议员案例助手生成完成消息。
                我们已经收集了足够的信息。感谢用户并解释下一步。
                上下文: {context}
                事实: {facts}
                
                要温暖、专业、令人安心。
                """,
                "ms": f"""
                Hasilkan mesej penyiapan untuk pembantu kes MP Singapura.
                Kami telah mengumpul maklumat yang mencukupi. Ucapkan terima kasih kepada pengguna dan jelaskan langkah seterusnya.
                Konteks: {context}
                Fakta: {facts}
                
                Bersikap mesra, profesional, dan menenangkan.
                """,
                "ta": f"""
                சிங்கப்பூர் எம்பி வழக்கு உதவியாளருக்கு முடிவு செய்தியை உருவாக்குங்கள்.
                நாங்கள் போதுமான தகவல்களை சேகரித்துள்ளோம். பயனருக்கு நன்றி தெரிவித்து அடுத்த படிகளை விளக்குங்கள்.
                சூழல்: {context}
                உண்மைகள்: {facts}
                
                வெப்பமாக, தொழில்முறையாக, மற்றும் நம்பிக்கையூட்டும் வகையில் இருங்கள்.
                """,
            }

            prompt = lang_prompts.get(language, lang_prompts["en"])
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating completion message: {e}")
            return "Thank you for providing the information. We have enough details to proceed with your case."

    async def _validate_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted facts for accuracy and completeness."""
        try:
            prompt = f"""
            Validate these extracted facts for accuracy and completeness:
            {facts}
            
            Return JSON with:
            - confidence: overall confidence 0.0-1.0
            - validated_facts: list of validated facts
            - issues: list of any issues found
            """

            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            import json

            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error validating facts: {e}")
            return {"confidence": 0.5, "validated_facts": facts, "issues": []}

    async def _assess_letter_readiness(
        self, facts: List[Dict[str, Any]], validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess if facts are ready for letter generation."""
        try:
            prompt = f"""
            Assess if these facts are ready for MP letter generation:
            Facts: {facts}
            Validation: {validation}
            
            Return JSON with:
            - ready: true/false
            - missing: list of missing required information
            - quality_score: 0.0-1.0
            """

            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )

            import json

            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error assessing letter readiness: {e}")
            return {
                "ready": False,
                "missing": ["More information needed"],
                "quality_score": 0.5,
            }

    async def _anticipate_missing_facts(
        self, intent: Dict[str, Any], facts: List[Dict[str, Any]]
    ) -> List[str]:
        """Lightweight anticipation of likely missing facts to guide quick chips."""
        try:
            prompt = f"""
            Given this intent and the current extracted facts, list the top 3 missing pieces of information that would be most helpful to resolve the case. Provide a JSON array of short phrases.
            Intent: {intent}
            Facts: {facts}
            """
            response = await self.flash_model.generate_content_async(
                prompt, safety_settings=self.safety_settings
            )
            import json

            data = json.loads(response.text)
            if isinstance(data, list):
                return data[:3]
            return []
        except Exception:
            return []

    def _fallback_letter_draft(self) -> LetterDraft:
        """Fallback letter draft when Gemini fails."""
        return LetterDraft(
            subject="MP Letter - Manual Review Required",
            content="This case requires manual review and letter generation.",
            tone="professional",
            key_points=["Manual review needed"],
            suggested_improvements=["Please review case details manually"],
            confidence=0.1,
        )

    def _fallback_approval_recommendation(self) -> ApprovalRecommendation:
        """Fallback approval recommendation when Gemini fails."""
        return ApprovalRecommendation(
            recommendation="MANUAL_REVIEW",
            confidence=0.1,
            reasoning="Fallback recommendation due to processing error",
            risk_factors=["Processing error occurred"],
            suggested_conditions=["Manual review required"],
            next_steps=["Review case manually"],
        )


# Global instance
gemini_integration = None


def get_gemini_integration() -> Optional[GeminiIntegration]:
    """Get the global Gemini integration instance."""
    global gemini_integration
    if gemini_integration is None and GEMINI_AVAILABLE:
        try:
            gemini_integration = GeminiIntegration()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini integration: {e}")
    return gemini_integration

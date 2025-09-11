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

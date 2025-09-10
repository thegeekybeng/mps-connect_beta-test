"""Main explainability engine for MPS Connect AI system."""

import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .feature_importance import FeatureImportanceAnalyzer
from database.connection import get_db

logger = logging.getLogger(__name__)


class MPSExplainabilityEngine:
    """Main explainability engine that coordinates all explanation methods."""

    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.db = get_db()

    def explain_classification(
        self, text: str, model_output: Dict[str, Any], model_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for text classification.

        Args:
            text: Input text to explain
            model_output: Model prediction results
            model_version: Version of the model used

        Returns:
            Dictionary containing all explanations
        """
        try:
            explanation_id = str(uuid.uuid4())
            start_time = datetime.now()

            # Generate SHAP explanations
            shap_explanations = self.shap_explainer.explain(
                text, model_output, model_version
            )

            # Generate LIME explanations
            lime_explanations = self.lime_explainer.explain(
                text, model_output, model_version
            )

            # Analyze feature importance
            feature_importance = self.feature_analyzer.analyze(text, model_output)

            # Generate decision path
            decision_path = self._generate_decision_path(
                text, model_output, feature_importance
            )

            # Calculate confidence breakdown
            confidence_breakdown = self._calculate_confidence_breakdown(
                model_output, feature_importance
            )

            # Generate counterfactual analysis
            counterfactuals = self._generate_counterfactuals(
                text, model_output, feature_importance
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Compile comprehensive explanation
            explanation = {
                "explanation_id": explanation_id,
                "model_version": model_version,
                "shap_explanations": shap_explanations,
                "lime_explanations": lime_explanations,
                "feature_importance": feature_importance,
                "decision_path": decision_path,
                "confidence_breakdown": confidence_breakdown,
                "counterfactuals": counterfactuals,
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat(),
            }

            # Store explanation in database
            self._store_explanation(explanation)

            logger.info(
                f"Generated explanation {explanation_id} in {processing_time:.2f}s"
            )
            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise

    def generate_explanation_report(self, case_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report for a case.

        Args:
            case_id: Unique case identifier

        Returns:
            Comprehensive explanation report
        """
        try:
            # Retrieve case data
            case_data = self._get_case_data(case_id)
            if not case_data:
                raise ValueError(f"Case {case_id} not found")

            # Get all explanations for this case
            explanations = self._get_case_explanations(case_id)

            # Generate summary report
            report = {
                "case_id": case_id,
                "case_data": case_data,
                "explanations": explanations,
                "summary": self._generate_summary(explanations),
                "recommendations": self._generate_recommendations(explanations),
                "generated_at": datetime.now().isoformat(),
            }

            return report

        except Exception as e:
            logger.error(f"Error generating explanation report: {str(e)}")
            raise

    def _generate_decision_path(
        self,
        text: str,
        model_output: Dict[str, Any],
        feature_importance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate step-by-step decision path."""
        decision_steps = []

        # Extract key features and their contributions
        for feature, importance in feature_importance.get("top_features", {}).items():
            decision_steps.append(
                {
                    "step": len(decision_steps) + 1,
                    "feature": feature,
                    "importance": importance,
                    "contribution": "positive" if importance > 0 else "negative",
                    "reasoning": self._get_feature_reasoning(feature, importance),
                }
            )

        return decision_steps

    def _calculate_confidence_breakdown(
        self, model_output: Dict[str, Any], feature_importance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed confidence breakdown."""
        confidence_scores = model_output.get("confidence_scores", {})

        breakdown = {
            "overall_confidence": model_output.get("confidence", 0.0),
            "category_confidence": confidence_scores,
            "feature_contribution": {},
            "uncertainty_factors": [],
        }

        # Analyze feature contributions to confidence
        for feature, importance in feature_importance.get("top_features", {}).items():
            contribution = abs(importance) * 0.1  # Scale factor
            breakdown["feature_contribution"][feature] = contribution

        # Identify uncertainty factors
        if model_output.get("confidence", 0) < 0.7:
            breakdown["uncertainty_factors"].append("Low overall confidence")

        if len(confidence_scores) > 1:
            max_conf = max(confidence_scores.values())
            min_conf = min(confidence_scores.values())
            if max_conf - min_conf < 0.2:
                breakdown["uncertainty_factors"].append(
                    "Close confidence scores between categories"
                )

        return breakdown

    def _generate_counterfactuals(
        self,
        text: str,
        model_output: Dict[str, Any],
        feature_importance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations."""
        counterfactuals = []

        # Generate counterfactuals by modifying key features
        top_features = feature_importance.get("top_features", {})

        for feature, importance in list(top_features.items())[:3]:  # Top 3 features
            if importance > 0:  # Positive contribution
                # Try removing or reducing this feature
                counterfactual_text = self._modify_text_remove_feature(text, feature)
                counterfactuals.append(
                    {
                        "original_text": text,
                        "modified_text": counterfactual_text,
                        "modification": f"Removed/reduced: {feature}",
                        "expected_impact": f"Would decrease confidence by ~{abs(importance):.2f}",
                    }
                )
            else:  # Negative contribution
                # Try adding or strengthening this feature
                counterfactual_text = self._modify_text_add_feature(text, feature)
                counterfactuals.append(
                    {
                        "original_text": text,
                        "modified_text": counterfactual_text,
                        "modification": f"Added/strengthened: {feature}",
                        "expected_impact": f"Would increase confidence by ~{abs(importance):.2f}",
                    }
                )

        return counterfactuals

    def _modify_text_remove_feature(self, text: str, feature: str) -> str:
        """Modify text to remove or reduce a feature."""
        # Simple implementation - in practice, this would be more sophisticated
        feature_words = feature.split()
        modified_text = text
        for word in feature_words:
            if word.lower() in modified_text.lower():
                modified_text = modified_text.replace(word, "")
        return modified_text.strip()

    def _modify_text_add_feature(self, text: str, feature: str) -> str:
        """Modify text to add or strengthen a feature."""
        # Simple implementation - in practice, this would be more sophisticated
        return f"{text} {feature}"

    def _get_feature_reasoning(self, feature: str, importance: float) -> str:
        """Get human-readable reasoning for feature importance."""
        if importance > 0.1:
            return f"Strong positive indicator for this category"
        elif importance > 0.05:
            return f"Moderate positive indicator for this category"
        elif importance < -0.1:
            return f"Strong negative indicator for this category"
        elif importance < -0.05:
            return f"Moderate negative indicator for this category"
        else:
            return f"Neutral indicator with minimal impact"

    def _generate_summary(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of all explanations."""
        if not explanations:
            return {"message": "No explanations available"}

        # Aggregate key insights
        all_features = []
        all_confidences = []

        for explanation in explanations:
            if "feature_importance" in explanation:
                features = explanation["feature_importance"].get("top_features", {})
                all_features.extend(features.keys())

            if "confidence_breakdown" in explanation:
                conf = explanation["confidence_breakdown"].get("overall_confidence", 0)
                all_confidences.append(conf)

        return {
            "total_explanations": len(explanations),
            "key_features": list(set(all_features))[:10],
            "average_confidence": np.mean(all_confidences) if all_confidences else 0,
            "confidence_range": {
                "min": min(all_confidences) if all_confidences else 0,
                "max": max(all_confidences) if all_confidences else 0,
            },
        }

    def _generate_recommendations(
        self, explanations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on explanations."""
        recommendations = []

        for explanation in explanations:
            confidence = explanation.get("confidence_breakdown", {}).get(
                "overall_confidence", 0
            )

            if confidence < 0.6:
                recommendations.append(
                    "Consider gathering more information to improve classification confidence"
                )

            uncertainty_factors = explanation.get("confidence_breakdown", {}).get(
                "uncertainty_factors", []
            )
            if uncertainty_factors:
                recommendations.append(
                    f"Address uncertainty factors: {', '.join(uncertainty_factors)}"
                )

            counterfactuals = explanation.get("counterfactuals", [])
            if counterfactuals:
                recommendations.append(
                    "Review counterfactual scenarios to understand decision boundaries"
                )

        return list(set(recommendations))  # Remove duplicates

    def _store_explanation(self, explanation: Dict[str, Any]) -> None:
        """Store explanation in database."""
        try:
            # This would integrate with your database
            # For now, we'll just log it
            logger.info(f"Storing explanation {explanation['explanation_id']}")
        except Exception as e:
            logger.error(f"Error storing explanation: {str(e)}")

    def _get_case_data(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve case data from database."""
        # This would query your database
        # For now, return None
        return None

    def _get_case_explanations(self, case_id: str) -> List[Dict[str, Any]]:
        """Retrieve all explanations for a case."""
        # This would query your database
        # For now, return empty list
        return []

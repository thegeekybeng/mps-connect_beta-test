"""LIME-based explainer for MPS Connect AI system."""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME-based explainer for text classification."""

    def __init__(self):
        self.explainer = None
        self.vectorizer = None
        self.model = None

    def explain(
        self, text: str, model_output: Dict[str, Any], model_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for text classification.

        Args:
            text: Input text to explain
            model_output: Model prediction results
            model_version: Version of the model used

        Returns:
            LIME explanation results
        """
        try:
            # Initialize explainer if not already done
            if self.explainer is None:
                self._initialize_explainer()

            # Generate LIME explanation
            explanation = self.explainer.explain_instance(
                text, self._dummy_predict_proba, num_features=10, top_labels=3
            )

            # Process explanation
            processed_explanation = self._process_lime_explanation(
                explanation, model_output
            )

            logger.info(f"Generated LIME explanation for model {model_version}")
            return processed_explanation

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return self._generate_fallback_explanation(text, model_output)

    def _initialize_explainer(self):
        """Initialize LIME explainer."""
        try:
            # Initialize LIME text explainer
            self.explainer = LimeTextExplainer(
                class_names=[
                    "transport",
                    "housing",
                    "social_support",
                    "employment",
                    "tax_finance",
                    "utilities_comms",
                ]
            )

            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )

        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {str(e)}")
            self.explainer = None

    def _dummy_predict_proba(self, texts):
        """Dummy prediction function for LIME."""
        # This is a placeholder - in practice, use your actual model
        # Return probabilities for each class
        num_classes = 6
        num_texts = len(texts)

        # Generate random probabilities that sum to 1
        probabilities = np.random.random((num_texts, num_classes))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def _process_lime_explanation(
        self, explanation, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process LIME explanation into interpretable format."""
        try:
            # Get explanation for the top predicted class
            top_label = explanation.top_labels[0] if explanation.top_labels else 0

            # Get explanation as list of (feature, weight) tuples
            exp_list = explanation.as_list(label=top_label)

            # Separate positive and negative contributions
            positive_features = [(f, w) for f, w in exp_list if w > 0]
            negative_features = [(f, w) for f, w in exp_list if w < 0]

            # Sort by absolute weight
            positive_features.sort(key=lambda x: abs(x[1]), reverse=True)
            negative_features.sort(key=lambda x: abs(x[1]), reverse=True)

            # Get local prediction
            local_prediction = (
                explanation.local_pred[top_label]
                if hasattr(explanation, "local_pred")
                else 0.0
            )

            # Get feature importance scores
            feature_importance = dict(exp_list)

            return {
                "lime_explanation": exp_list,
                "top_label": top_label,
                "local_prediction": float(local_prediction),
                "positive_features": positive_features,
                "negative_features": negative_features,
                "feature_importance": feature_importance,
                "summary": {
                    "total_features": len(exp_list),
                    "positive_contributors": len(positive_features),
                    "negative_contributors": len(negative_features),
                    "max_importance": (
                        max([abs(w) for _, w in exp_list]) if exp_list else 0
                    ),
                    "confidence": float(local_prediction),
                },
            }

        except Exception as e:
            logger.error(f"Error processing LIME explanation: {str(e)}")
            return self._generate_fallback_explanation("", model_output)

    def _generate_fallback_explanation(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback explanation when LIME fails."""
        return {
            "lime_explanation": [],
            "top_label": 0,
            "local_prediction": 0.0,
            "positive_features": [],
            "negative_features": [],
            "feature_importance": {},
            "summary": {
                "total_features": 0,
                "positive_contributors": 0,
                "negative_contributors": 0,
                "max_importance": 0,
                "confidence": 0.0,
            },
            "fallback": True,
            "message": "LIME explanation unavailable - using fallback",
        }

"""SHAP-based explainer for MPS Connect AI system."""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import shap
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainer for text classification."""

    def __init__(self):
        self.explainer = None
        self.vectorizer = None
        self.model = None

    def explain(
        self, text: str, model_output: Dict[str, Any], model_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for text classification.

        Args:
            text: Input text to explain
            model_output: Model prediction results
            model_version: Version of the model used

        Returns:
            SHAP explanation results
        """
        try:
            # Initialize explainer if not already done
            if self.explainer is None:
                self._initialize_explainer()

            # Prepare text for SHAP
            text_vectorized = self._vectorize_text(text)

            # Generate SHAP values
            shap_values = self.explainer.shap_values(text_vectorized)

            # Extract feature names
            feature_names = self._get_feature_names(text)

            # Process SHAP values
            explanation = self._process_shap_values(
                shap_values, feature_names, model_output
            )

            logger.info(f"Generated SHAP explanation for model {model_version}")
            return explanation

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            # Return fallback explanation
            return self._generate_fallback_explanation(text, model_output)

    def _initialize_explainer(self):
        """Initialize SHAP explainer."""
        try:
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )

            # Create dummy data for initialization
            dummy_texts = ["dummy text for initialization"]
            self.vectorizer.fit(dummy_texts)

            # Initialize SHAP explainer
            # Note: In practice, you would use your actual trained model
            self.explainer = shap.Explainer(self._dummy_model)

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            self.explainer = None

    def _dummy_model(self, texts):
        """Dummy model for SHAP initialization."""
        # This is a placeholder - in practice, use your actual model
        return np.random.random((len(texts), 5))

    def _vectorize_text(self, text: str) -> np.ndarray:
        """Vectorize text for SHAP analysis."""
        try:
            if self.vectorizer is None:
                self._initialize_explainer()

            return self.vectorizer.transform([text]).toarray()
        except Exception as e:
            logger.error(f"Error vectorizing text: {str(e)}")
            return np.zeros((1, 1000))  # Fallback

    def _get_feature_names(self, text: str) -> List[str]:
        """Extract feature names from text."""
        try:
            if self.vectorizer is None:
                return []

            # Get feature names from vectorizer
            feature_names = self.vectorizer.get_feature_names_out()

            # For text, we can also extract n-grams
            words = text.lower().split()
            ngrams = []

            # Unigrams
            ngrams.extend(words)

            # Bigrams
            for i in range(len(words) - 1):
                ngrams.append(f"{words[i]} {words[i+1]}")

            # Filter to only include features that exist in our vectorizer
            valid_features = [f for f in ngrams if f in feature_names]

            return valid_features[:20]  # Limit to top 20 features

        except Exception as e:
            logger.error(f"Error extracting feature names: {str(e)}")
            return []

    def _process_shap_values(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process SHAP values into interpretable format."""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values = np.array(shap_values)

            if len(shap_values.shape) > 2:
                # Reshape if necessary
                shap_values = shap_values.reshape(shap_values.shape[0], -1)

            # Get the most important features
            if len(shap_values) > 0:
                values = shap_values[0]  # First (and likely only) sample
            else:
                values = np.array([])

            # Create feature importance mapping
            feature_importance = {}
            if len(values) > 0 and len(feature_names) > 0:
                # Map values to feature names
                min_len = min(len(values), len(feature_names))
                for i in range(min_len):
                    feature_importance[feature_names[i]] = float(values[i])

            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )

            # Extract top positive and negative features
            positive_features = [(f, v) for f, v in sorted_features if v > 0][:10]
            negative_features = [(f, v) for f, v in sorted_features if v < 0][:10]

            return {
                "shap_values": shap_values.tolist() if len(shap_values) > 0 else [],
                "feature_importance": dict(sorted_features[:20]),
                "top_positive_features": positive_features,
                "top_negative_features": negative_features,
                "feature_names": feature_names,
                "summary": {
                    "total_features": len(feature_names),
                    "positive_contributors": len(positive_features),
                    "negative_contributors": len(negative_features),
                    "max_importance": (
                        max([abs(v) for v in feature_importance.values()])
                        if feature_importance
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error processing SHAP values: {str(e)}")
            return self._generate_fallback_explanation("", model_output)

    def _generate_fallback_explanation(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback explanation when SHAP fails."""
        return {
            "shap_values": [],
            "feature_importance": {},
            "top_positive_features": [],
            "top_negative_features": [],
            "feature_names": [],
            "summary": {
                "total_features": 0,
                "positive_contributors": 0,
                "negative_contributors": 0,
                "max_importance": 0,
            },
            "fallback": True,
            "message": "SHAP explanation unavailable - using fallback",
        }

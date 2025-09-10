"""Feature importance analyzer for MPS Connect AI system."""

import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyzes feature importance for text classification."""

    def __init__(self):
        self.vectorizer = None
        self.keyword_patterns = self._load_keyword_patterns()
        self.category_keywords = self._load_category_keywords()

    def analyze(self, text: str, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feature importance for text classification.

        Args:
            text: Input text to analyze
            model_output: Model prediction results

        Returns:
            Feature importance analysis
        """
        try:
            # Extract text features
            text_features = self._extract_text_features(text)

            # Analyze keyword importance
            keyword_importance = self._analyze_keyword_importance(text, model_output)

            # Analyze n-gram importance
            ngram_importance = self._analyze_ngram_importance(text, model_output)

            # Analyze semantic importance
            semantic_importance = self._analyze_semantic_importance(text, model_output)

            # Combine all importance scores
            combined_importance = self._combine_importance_scores(
                keyword_importance, ngram_importance, semantic_importance
            )

            # Generate top features
            top_features = self._get_top_features(combined_importance)

            return {
                "text_features": text_features,
                "keyword_importance": keyword_importance,
                "ngram_importance": ngram_importance,
                "semantic_importance": semantic_importance,
                "combined_importance": combined_importance,
                "top_features": top_features,
                "summary": self._generate_summary(top_features, model_output),
            }

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return self._generate_fallback_analysis(text, model_output)

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        try:
            words = text.lower().split()

            # Basic statistics
            features = {
                "word_count": len(words),
                "char_count": len(text),
                "sentence_count": len(re.split(r"[.!?]+", text)),
                "avg_word_length": (
                    np.mean([len(word) for word in words]) if words else 0
                ),
                "unique_words": len(set(words)),
                "vocabulary_diversity": len(set(words)) / len(words) if words else 0,
            }

            # Extract numbers and amounts
            numbers = re.findall(r"\$?[\d,]+\.?\d*", text)
            features["numbers"] = numbers
            features["has_amounts"] = any("$" in num for num in numbers)

            # Extract dates
            dates = re.findall(
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text
            )
            features["dates"] = dates
            features["has_dates"] = len(dates) > 0

            # Extract reference numbers
            refs = re.findall(r"\b[A-Z]{2,}\s*\d{3,}\b|\b[A-Z0-9]{6,}\b", text)
            features["reference_numbers"] = refs
            features["has_references"] = len(refs) > 0

            return features

        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return {}

    def _analyze_keyword_importance(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze importance of keywords."""
        try:
            text_lower = text.lower()
            importance_scores = {}

            # Get predicted category
            predicted_category = model_output.get("predicted_category", "")

            # Check category-specific keywords
            if predicted_category in self.category_keywords:
                category_keywords = self.category_keywords[predicted_category]

                for keyword, weight in category_keywords.items():
                    if keyword.lower() in text_lower:
                        # Calculate importance based on frequency and weight
                        frequency = text_lower.count(keyword.lower())
                        importance_scores[keyword] = frequency * weight

            # Check general keyword patterns
            for pattern, weight in self.keyword_patterns.items():
                if re.search(pattern, text_lower):
                    importance_scores[f"pattern_{pattern}"] = weight

            return importance_scores

        except Exception as e:
            logger.error(f"Error analyzing keyword importance: {str(e)}")
            return {}

    def _analyze_ngram_importance(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze importance of n-grams."""
        try:
            words = text.lower().split()
            ngram_importance = {}

            # Unigrams
            unigrams = words
            unigram_counts = Counter(unigrams)

            # Bigrams
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)

            # Trigrams
            trigrams = [
                f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)
            ]
            trigram_counts = Counter(trigrams)

            # Calculate importance scores
            for ngram, count in unigram_counts.items():
                if len(ngram) > 2:  # Skip very short words
                    ngram_importance[ngram] = count * 0.1

            for ngram, count in bigram_counts.items():
                ngram_importance[ngram] = count * 0.2

            for ngram, count in trigram_counts.items():
                ngram_importance[ngram] = count * 0.3

            return ngram_importance

        except Exception as e:
            logger.error(f"Error analyzing n-gram importance: {str(e)}")
            return {}

    def _analyze_semantic_importance(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze semantic importance using TF-IDF."""
        try:
            if self.vectorizer is None:
                self._initialize_vectorizer()

            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()

            # Get TF-IDF scores
            tfidf_scores = text_vector.toarray()[0]

            # Create importance mapping
            semantic_importance = {}
            for i, score in enumerate(tfidf_scores):
                if score > 0:
                    semantic_importance[feature_names[i]] = float(score)

            return semantic_importance

        except Exception as e:
            logger.error(f"Error analyzing semantic importance: {str(e)}")
            return {}

    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer."""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 3)
            )

            # Fit with dummy data
            dummy_texts = ["dummy text for initialization"]
            self.vectorizer.fit(dummy_texts)

        except Exception as e:
            logger.error(f"Error initializing vectorizer: {str(e)}")
            self.vectorizer = None

    def _combine_importance_scores(
        self,
        keyword_importance: Dict[str, float],
        ngram_importance: Dict[str, float],
        semantic_importance: Dict[str, float],
    ) -> Dict[str, float]:
        """Combine different importance scores."""
        try:
            combined = {}

            # Add keyword importance
            for feature, score in keyword_importance.items():
                combined[feature] = combined.get(feature, 0) + score * 0.4

            # Add n-gram importance
            for feature, score in ngram_importance.items():
                combined[feature] = combined.get(feature, 0) + score * 0.3

            # Add semantic importance
            for feature, score in semantic_importance.items():
                combined[feature] = combined.get(feature, 0) + score * 0.3

            return combined

        except Exception as e:
            logger.error(f"Error combining importance scores: {str(e)}")
            return {}

    def _get_top_features(
        self, combined_importance: Dict[str, float], top_k: int = 20
    ) -> Dict[str, float]:
        """Get top K most important features."""
        try:
            # Sort by absolute importance
            sorted_features = sorted(
                combined_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )

            return dict(sorted_features[:top_k])

        except Exception as e:
            logger.error(f"Error getting top features: {str(e)}")
            return {}

    def _generate_summary(
        self, top_features: Dict[str, float], model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of feature importance analysis."""
        try:
            if not top_features:
                return {"message": "No features analyzed"}

            # Calculate statistics
            scores = list(top_features.values())

            return {
                "total_features": len(top_features),
                "max_importance": max(scores) if scores else 0,
                "min_importance": min(scores) if scores else 0,
                "avg_importance": np.mean(scores) if scores else 0,
                "positive_features": len([s for s in scores if s > 0]),
                "negative_features": len([s for s in scores if s < 0]),
                "top_feature": (
                    max(top_features.items(), key=lambda x: abs(x[1]))[0]
                    if top_features
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"message": "Error generating summary"}

    def _load_keyword_patterns(self) -> Dict[str, float]:
        """Load keyword patterns and their importance weights."""
        return {
            r"\$[\d,]+": 0.8,  # Money amounts
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b": 0.6,  # Dates
            r"\b[A-Z]{2,}\s*\d{3,}\b": 0.7,  # Reference numbers
            r"\b(urgent|asap|immediately)\b": 0.9,  # Urgency indicators
            r"\b(fine|penalty|summon)\b": 0.8,  # Legal terms
            r"\b(appeal|request|help)\b": 0.7,  # Action words
            r"\b(family|children|kids|wife|husband)\b": 0.6,  # Family terms
            r"\b(job|work|employment|salary)\b": 0.6,  # Employment terms
            r"\b(hdb|housing|rental|purchase)\b": 0.8,  # Housing terms
            r"\b(comcare|financial|assistance)\b": 0.7,  # Social support terms
        }

    def _load_category_keywords(self) -> Dict[str, Dict[str, float]]:
        """Load category-specific keywords and weights."""
        return {
            "transport": {
                "traffic": 0.9,
                "fine": 0.8,
                "demerit": 0.8,
                "points": 0.7,
                "offence": 0.8,
                "summon": 0.8,
                "police": 0.7,
                "driving": 0.6,
                "license": 0.6,
            },
            "housing": {
                "hdb": 0.9,
                "housing": 0.8,
                "rental": 0.7,
                "purchase": 0.7,
                "flat": 0.6,
                "bto": 0.8,
                "resale": 0.6,
                "appeal": 0.6,
            },
            "social_support": {
                "comcare": 0.9,
                "financial": 0.7,
                "assistance": 0.8,
                "support": 0.6,
                "welfare": 0.7,
                "help": 0.5,
                "family": 0.6,
                "children": 0.6,
            },
            "employment": {
                "job": 0.8,
                "work": 0.7,
                "employment": 0.8,
                "salary": 0.7,
                "unemployed": 0.8,
                "resume": 0.6,
                "interview": 0.6,
                "career": 0.5,
            },
            "tax_finance": {
                "tax": 0.9,
                "iras": 0.8,
                "cpf": 0.7,
                "income": 0.6,
                "finance": 0.6,
                "payment": 0.5,
                "bill": 0.5,
                "debt": 0.6,
            },
            "utilities_comms": {
                "utilities": 0.8,
                "electricity": 0.7,
                "water": 0.7,
                "gas": 0.7,
                "internet": 0.6,
                "mobile": 0.6,
                "phone": 0.6,
                "broadband": 0.6,
            },
        }

    def _generate_fallback_analysis(
        self, text: str, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback analysis when main analysis fails."""
        return {
            "text_features": {},
            "keyword_importance": {},
            "ngram_importance": {},
            "semantic_importance": {},
            "combined_importance": {},
            "top_features": {},
            "summary": {
                "total_features": 0,
                "max_importance": 0,
                "min_importance": 0,
                "avg_importance": 0,
                "positive_features": 0,
                "negative_features": 0,
                "top_feature": None,
            },
            "fallback": True,
            "message": "Feature importance analysis unavailable - using fallback",
        }

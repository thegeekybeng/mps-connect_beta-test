"""Model registry for transparency in MPS Connect AI system."""

import uuid
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Data class for model information."""

    model_id: str
    model_name: str
    model_version: str
    model_type: str
    training_data_hash: str
    performance_metrics: Dict[str, Any]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0


class ModelRegistry:
    """Registry for tracking AI models and their usage."""

    def __init__(self):
        self.models = {}
        self.usage_history = []
        self.performance_metrics = {}

    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        training_data_hash: str = None,
        performance_metrics: Dict[str, Any] = None,
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_type: Type of model (classification, generation, etc.)
            training_data_hash: Hash of training data
            performance_metrics: Model performance metrics

        Returns:
            Model ID
        """
        try:
            model_id = str(uuid.uuid4())

            # Generate training data hash if not provided
            if training_data_hash is None:
                training_data_hash = self._generate_training_data_hash(
                    model_name, model_version
                )

            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                training_data_hash=training_data_hash,
                performance_metrics=performance_metrics or {},
                created_at=datetime.now(),
            )

            # Register model
            self.models[model_id] = model_info

            # Store performance metrics
            if performance_metrics:
                self.performance_metrics[model_id] = performance_metrics

            logger.info(
                f"Registered model {model_name} v{model_version} with ID {model_id}"
            )
            return model_id

        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise

    def track_usage(self, usage_entry: Dict[str, Any]) -> None:
        """
        Track model usage.

        Args:
            usage_entry: Model usage entry
        """
        try:
            model_name = usage_entry.get("model_name")
            model_version = usage_entry.get("model_version")

            # Find model by name and version
            model_id = self._find_model_by_name_version(model_name, model_version)

            if model_id:
                # Update model usage
                model_info = self.models[model_id]
                model_info.usage_count += 1
                model_info.last_used = datetime.now()

                # Add to usage history
                usage_record = {
                    "usage_id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "timestamp": usage_entry.get(
                        "timestamp", datetime.now().isoformat()
                    ),
                    "session_id": usage_entry.get("session_id"),
                    "model_type": usage_entry.get("model_type", "unknown"),
                }

                self.usage_history.append(usage_record)

                logger.debug(f"Tracked usage for model {model_name} v{model_version}")
            else:
                logger.warning(
                    f"Model {model_name} v{model_version} not found in registry"
                )

        except Exception as e:
            logger.error(f"Error tracking model usage: {str(e)}")

    def get_model_info(self, model_versions: Dict[str, str]) -> Dict[str, Any]:
        """
        Get model information for given model versions.

        Args:
            model_versions: Dictionary of model names to versions

        Returns:
            Model information
        """
        try:
            model_info = {}

            for model_name, version in model_versions.items():
                model_id = self._find_model_by_name_version(model_name, version)

                if model_id:
                    model = self.models[model_id]
                    model_info[model_name] = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_version": model.model_version,
                        "model_type": model.model_type,
                        "training_data_hash": model.training_data_hash,
                        "performance_metrics": model.performance_metrics,
                        "created_at": model.created_at.isoformat(),
                        "last_used": (
                            model.last_used.isoformat() if model.last_used else None
                        ),
                        "usage_count": model.usage_count,
                    }
                else:
                    model_info[model_name] = {
                        "error": f"Model {model_name} v{version} not found in registry"
                    }

            return model_info

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}

    def update_performance_metrics(
        self, model_id: str, metrics: Dict[str, Any]
    ) -> None:
        """
        Update performance metrics for a model.

        Args:
            model_id: Model ID
            metrics: Performance metrics
        """
        try:
            if model_id in self.models:
                self.models[model_id].performance_metrics.update(metrics)
                self.performance_metrics[model_id] = metrics
                logger.info(f"Updated performance metrics for model {model_id}")
            else:
                logger.warning(f"Model {model_id} not found for metrics update")

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.

        Args:
            model_id: Model ID

        Returns:
            Performance metrics
        """
        try:
            if model_id in self.models:
                return self.models[model_id].performance_metrics
            else:
                return {"error": f"Model {model_id} not found"}

        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {"error": str(e)}

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all models."""
        try:
            total_models = len(self.models)
            total_usage = sum(model.usage_count for model in self.models.values())

            # Most used models
            most_used = sorted(
                self.models.items(), key=lambda x: x[1].usage_count, reverse=True
            )[:10]

            # Usage by model type
            usage_by_type = {}
            for model in self.models.values():
                model_type = model.model_type
                usage_by_type[model_type] = (
                    usage_by_type.get(model_type, 0) + model.usage_count
                )

            # Recent usage
            recent_usage = [
                {
                    "model_name": model.model_name,
                    "model_version": model.model_version,
                    "last_used": (
                        model.last_used.isoformat() if model.last_used else None
                    ),
                    "usage_count": model.usage_count,
                }
                for model in self.models.values()
                if model.last_used
            ]
            recent_usage.sort(key=lambda x: x["last_used"] or "", reverse=True)

            return {
                "total_models": total_models,
                "total_usage": total_usage,
                "most_used_models": [
                    {
                        "model_name": model.model_name,
                        "model_version": model.model_version,
                        "usage_count": model.usage_count,
                    }
                    for model_id, model in most_used
                ],
                "usage_by_type": usage_by_type,
                "recent_usage": recent_usage[:10],
            }

        except Exception as e:
            logger.error(f"Error getting usage statistics: {str(e)}")
            return {"error": str(e)}

    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get model lineage and version history.

        Args:
            model_id: Model ID

        Returns:
            Model lineage information
        """
        try:
            if model_id not in self.models:
                return {"error": f"Model {model_id} not found"}

            model = self.models[model_id]
            model_name = model.model_name

            # Find all versions of this model
            versions = [
                {
                    "model_id": m.model_id,
                    "model_version": m.model_version,
                    "created_at": m.created_at.isoformat(),
                    "last_used": m.last_used.isoformat() if m.last_used else None,
                    "usage_count": m.usage_count,
                }
                for m in self.models.values()
                if m.model_name == model_name
            ]

            # Sort by version
            versions.sort(key=lambda x: x["model_version"])

            return {
                "model_name": model_name,
                "total_versions": len(versions),
                "versions": versions,
                "current_version": model.model_version,
                "training_data_hash": model.training_data_hash,
            }

        except Exception as e:
            logger.error(f"Error getting model lineage: {str(e)}")
            return {"error": str(e)}

    def validate_model_integrity(self, model_id: str) -> Dict[str, Any]:
        """
        Validate model integrity.

        Args:
            model_id: Model ID

        Returns:
            Integrity validation results
        """
        try:
            if model_id not in self.models:
                return {"error": f"Model {model_id} not found"}

            model = self.models[model_id]

            # Check if model has been used recently
            recently_used = (
                model.last_used and (datetime.now() - model.last_used).days < 30
            )

            # Check performance metrics
            has_metrics = bool(model.performance_metrics)

            # Check training data hash
            has_training_hash = bool(model.training_data_hash)

            # Calculate integrity score
            integrity_score = 0
            if recently_used:
                integrity_score += 25
            if has_metrics:
                integrity_score += 25
            if has_training_hash:
                integrity_score += 25
            if model.usage_count > 0:
                integrity_score += 25

            return {
                "model_id": model_id,
                "model_name": model.model_name,
                "model_version": model.model_version,
                "integrity_score": integrity_score,
                "recently_used": recently_used,
                "has_metrics": has_metrics,
                "has_training_hash": has_training_hash,
                "usage_count": model.usage_count,
                "status": "healthy" if integrity_score >= 75 else "needs_attention",
            }

        except Exception as e:
            logger.error(f"Error validating model integrity: {str(e)}")
            return {"error": str(e)}

    def _find_model_by_name_version(
        self, model_name: str, version: str
    ) -> Optional[str]:
        """Find model ID by name and version."""
        try:
            for model_id, model in self.models.items():
                if model.model_name == model_name and model.model_version == version:
                    return model_id
            return None

        except Exception as e:
            logger.error(f"Error finding model by name and version: {str(e)}")
            return None

    def _generate_training_data_hash(self, model_name: str, version: str) -> str:
        """Generate a hash for training data."""
        try:
            # Create a hash based on model name, version, and timestamp
            data_string = f"{model_name}_{version}_{datetime.now().isoformat()}"
            return hashlib.sha256(data_string.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Error generating training data hash: {str(e)}")
            return "unknown"

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of the model registry."""
        try:
            total_models = len(self.models)
            total_usage = sum(model.usage_count for model in self.models.values())

            # Model types
            model_types = {}
            for model in self.models.values():
                model_type = model.model_type
                model_types[model_type] = model_types.get(model_type, 0) + 1

            # Recent registrations
            recent_models = sorted(
                self.models.values(), key=lambda x: x.created_at, reverse=True
            )[:10]

            return {
                "total_models": total_models,
                "total_usage": total_usage,
                "model_types": model_types,
                "recent_models": [
                    {
                        "model_name": model.model_name,
                        "model_version": model.model_version,
                        "model_type": model.model_type,
                        "created_at": model.created_at.isoformat(),
                    }
                    for model in recent_models
                ],
                "registry_health": self._assess_registry_health(),
            }

        except Exception as e:
            logger.error(f"Error getting registry summary: {str(e)}")
            return {"error": str(e)}

    def _assess_registry_health(self) -> Dict[str, Any]:
        """Assess the health of the model registry."""
        try:
            total_models = len(self.models)
            if total_models == 0:
                return {"status": "empty", "score": 0}

            # Check various health indicators
            models_with_metrics = sum(
                1 for model in self.models.values() if model.performance_metrics
            )
            models_recently_used = sum(
                1
                for model in self.models.values()
                if model.last_used and (datetime.now() - model.last_used).days < 30
            )
            models_with_hash = sum(
                1 for model in self.models.values() if model.training_data_hash
            )

            # Calculate health score
            health_score = (
                (models_with_metrics / total_models) * 25
                + (models_recently_used / total_models) * 25
                + (models_with_hash / total_models) * 25
                + (min(total_usage / total_models, 10) / 10) * 25  # Usage factor
            )

            if health_score >= 80:
                status = "excellent"
            elif health_score >= 60:
                status = "good"
            elif health_score >= 40:
                status = "fair"
            else:
                status = "poor"

            return {
                "status": status,
                "score": health_score,
                "models_with_metrics": models_with_metrics,
                "models_recently_used": models_recently_used,
                "models_with_hash": models_with_hash,
                "total_models": total_models,
            }

        except Exception as e:
            logger.error(f"Error assessing registry health: {str(e)}")
            return {"status": "error", "score": 0}

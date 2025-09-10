"""Content tracking for transparency in MPS Connect AI system."""

import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContentSource:
    """Data class for content sources."""

    source_id: str
    source_type: str
    source_name: str
    content_description: str
    confidence_score: float
    attribution: Dict[str, Any]
    timestamp: datetime


class ContentTracker:
    """Tracks content generation and sources for transparency."""

    def __init__(self):
        self.tracked_content = {}
        self.source_registry = {}

    def track_content(self, session_id: str, content_entry: Dict[str, Any]) -> str:
        """
        Track content generation and its sources.

        Args:
            session_id: Tracking session ID
            content_entry: Content entry to track

        Returns:
            Content tracking ID
        """
        try:
            content_id = str(uuid.uuid4())

            # Process content sources
            processed_sources = []
            for source in content_entry.get("sources", []):
                processed_source = self._process_content_source(source)
                processed_sources.append(processed_source)

            # Create content tracking entry
            tracking_entry = {
                "content_id": content_id,
                "session_id": session_id,
                "content_type": content_entry.get("content_type", "unknown"),
                "sources": processed_sources,
                "timestamp": content_entry.get("timestamp", datetime.now().isoformat()),
                "tracking_id": content_entry.get("tracking_id", str(uuid.uuid4())),
            }

            # Store tracking entry
            if session_id not in self.tracked_content:
                self.tracked_content[session_id] = []

            self.tracked_content[session_id].append(tracking_entry)

            # Update source registry
            self._update_source_registry(processed_sources)

            logger.debug(f"Tracked content {content_id} for session {session_id}")
            return content_id

        except Exception as e:
            logger.error(f"Error tracking content: {str(e)}")
            raise

    def _process_content_source(self, source: Dict[str, Any]) -> ContentSource:
        """Process a content source into structured format."""
        try:
            source_id = str(uuid.uuid4())

            # Extract source information
            source_type = source.get("type", "unknown")
            source_name = source.get("name", "Unknown Source")
            content_description = source.get("description", "No description available")
            confidence_score = source.get("confidence", 0.0)
            attribution = source.get("attribution", {})

            return ContentSource(
                source_id=source_id,
                source_type=source_type,
                source_name=source_name,
                content_description=content_description,
                confidence_score=confidence_score,
                attribution=attribution,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error processing content source: {str(e)}")
            # Return fallback source
            return ContentSource(
                source_id=str(uuid.uuid4()),
                source_type="unknown",
                source_name="Unknown Source",
                content_description="Error processing source",
                confidence_score=0.0,
                attribution={},
                timestamp=datetime.now(),
            )

    def _update_source_registry(self, sources: List[ContentSource]) -> None:
        """Update the source registry with new sources."""
        try:
            for source in sources:
                source_key = f"{source.source_type}:{source.source_name}"

                if source_key not in self.source_registry:
                    self.source_registry[source_key] = {
                        "source_type": source.source_type,
                        "source_name": source.source_name,
                        "usage_count": 0,
                        "total_confidence": 0.0,
                        "first_used": source.timestamp.isoformat(),
                        "last_used": source.timestamp.isoformat(),
                    }

                # Update usage statistics
                registry_entry = self.source_registry[source_key]
                registry_entry["usage_count"] += 1
                registry_entry["total_confidence"] += source.confidence_score
                registry_entry["last_used"] = source.timestamp.isoformat()

        except Exception as e:
            logger.error(f"Error updating source registry: {str(e)}")

    def get_content_attribution(self, session_id: str) -> Dict[str, Any]:
        """
        Get content attribution for a session.

        Args:
            session_id: Tracking session ID

        Returns:
            Content attribution information
        """
        try:
            if session_id not in self.tracked_content:
                return {"message": f"No content tracked for session {session_id}"}

            content_entries = self.tracked_content[session_id]

            # Analyze content sources
            source_analysis = self._analyze_content_sources(content_entries)

            # Generate attribution report
            attribution = {
                "session_id": session_id,
                "total_content_items": len(content_entries),
                "source_analysis": source_analysis,
                "content_breakdown": self._breakdown_content_types(content_entries),
                "confidence_analysis": self._analyze_confidence_scores(content_entries),
                "attribution_chain": self._build_attribution_chain(content_entries),
            }

            return attribution

        except Exception as e:
            logger.error(f"Error getting content attribution: {str(e)}")
            return {"error": str(e)}

    def _analyze_content_sources(
        self, content_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze content sources across all entries."""
        try:
            all_sources = []
            for entry in content_entries:
                all_sources.extend(entry.get("sources", []))

            if not all_sources:
                return {"message": "No sources found"}

            # Analyze source types
            source_types = {}
            source_names = {}
            confidence_scores = []

            for source in all_sources:
                source_type = source.source_type
                source_name = source.source_name
                confidence = source.confidence_score

                source_types[source_type] = source_types.get(source_type, 0) + 1
                source_names[source_name] = source_names.get(source_name, 0) + 1
                confidence_scores.append(confidence)

            return {
                "total_sources": len(all_sources),
                "unique_source_types": len(source_types),
                "unique_source_names": len(source_names),
                "source_type_distribution": source_types,
                "source_name_distribution": source_names,
                "average_confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0
                ),
                "confidence_range": {
                    "min": min(confidence_scores) if confidence_scores else 0,
                    "max": max(confidence_scores) if confidence_scores else 0,
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing content sources: {str(e)}")
            return {"error": str(e)}

    def _breakdown_content_types(
        self, content_entries: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Break down content by type."""
        try:
            content_types = {}
            for entry in content_entries:
                content_type = entry.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

            return content_types

        except Exception as e:
            logger.error(f"Error breaking down content types: {str(e)}")
            return {}

    def _analyze_confidence_scores(
        self, content_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze confidence scores across content entries."""
        try:
            all_confidences = []
            for entry in content_entries:
                for source in entry.get("sources", []):
                    all_confidences.append(source.confidence_score)

            if not all_confidences:
                return {"message": "No confidence scores available"}

            return {
                "total_scores": len(all_confidences),
                "average_confidence": sum(all_confidences) / len(all_confidences),
                "min_confidence": min(all_confidences),
                "max_confidence": max(all_confidences),
                "high_confidence_count": len([c for c in all_confidences if c > 0.8]),
                "low_confidence_count": len([c for c in all_confidences if c < 0.5]),
            }

        except Exception as e:
            logger.error(f"Error analyzing confidence scores: {str(e)}")
            return {"error": str(e)}

    def _build_attribution_chain(
        self, content_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build attribution chain for content generation."""
        try:
            attribution_chain = []

            for entry in content_entries:
                chain_entry = {
                    "content_type": entry.get("content_type", "unknown"),
                    "timestamp": entry.get("timestamp", ""),
                    "sources": [
                        {
                            "source_name": source.source_name,
                            "source_type": source.source_type,
                            "confidence": source.confidence_score,
                            "attribution": source.attribution,
                        }
                        for source in entry.get("sources", [])
                    ],
                }
                attribution_chain.append(chain_entry)

            return attribution_chain

        except Exception as e:
            logger.error(f"Error building attribution chain: {str(e)}")
            return []

    def get_source_registry(self) -> Dict[str, Any]:
        """Get the complete source registry."""
        try:
            # Calculate additional statistics
            registry_stats = {
                "total_sources": len(self.source_registry),
                "source_types": list(
                    set(entry["source_type"] for entry in self.source_registry.values())
                ),
                "most_used_sources": sorted(
                    self.source_registry.items(),
                    key=lambda x: x[1]["usage_count"],
                    reverse=True,
                )[:10],
                "average_confidence_by_source": self._calculate_average_confidence_by_source(),
            }

            return {"registry": self.source_registry, "statistics": registry_stats}

        except Exception as e:
            logger.error(f"Error getting source registry: {str(e)}")
            return {"error": str(e)}

    def _calculate_average_confidence_by_source(self) -> Dict[str, float]:
        """Calculate average confidence by source."""
        try:
            avg_confidence = {}

            for source_key, entry in self.source_registry.items():
                if entry["usage_count"] > 0:
                    avg_confidence[source_key] = (
                        entry["total_confidence"] / entry["usage_count"]
                    )
                else:
                    avg_confidence[source_key] = 0.0

            return avg_confidence

        except Exception as e:
            logger.error(f"Error calculating average confidence: {str(e)}")
            return {}

    def clear_session_data(self, session_id: str) -> None:
        """Clear tracking data for a session."""
        try:
            if session_id in self.tracked_content:
                del self.tracked_content[session_id]
                logger.info(f"Cleared content tracking data for session {session_id}")

        except Exception as e:
            logger.error(f"Error clearing session data: {str(e)}")

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracking data."""
        try:
            total_sessions = len(self.tracked_content)
            total_content_items = sum(
                len(entries) for entries in self.tracked_content.values()
            )
            total_sources = len(self.source_registry)

            return {
                "total_sessions": total_sessions,
                "total_content_items": total_content_items,
                "total_sources": total_sources,
                "active_sessions": list(self.tracked_content.keys()),
                "source_registry_size": total_sources,
            }

        except Exception as e:
            logger.error(f"Error getting tracking summary: {str(e)}")
            return {"error": str(e)}

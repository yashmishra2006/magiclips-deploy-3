"""
Pub/Sub Publisher for Clip Processing Jobs

Publishes clip processing jobs to Pub/Sub topic for asynchronous processing.
"""

import json
import logging
import os
import time
from typing import List, Dict, Optional
from concurrent import futures

from google.cloud import pubsub_v1

import obs

logger = logging.getLogger(__name__)


class ClipJobPublisher:
    """Publishes clip processing jobs to Pub/Sub."""

    def __init__(
        self,
        project_id: str = None,
        topic_name: str = "clip-processing-jobs",
    ):
        """Initialize publisher.

        Args:
            project_id: GCP project ID
            topic_name: Pub/Sub topic name
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "new-backend-475414")
        self.topic_name = topic_name
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(self.project_id, self.topic_name)

        logger.debug("Initialized Pub/Sub publisher for topic: %s", self.topic_path)

    def _publish(self, message_data: Dict) -> futures.Future:
        message_bytes = json.dumps(message_data).encode("utf-8")
        return self.publisher.publish(
            self.topic_path,
            message_bytes,
            clip_id=message_data["clip_id"],
            project_id=message_data["project_id"],
            type="clip_processing",
            # Correlation: the consumer reads attribute `request_id` back (obs convention).
            **obs.pubsub_attributes(),
        )

    def publish_clips_batch(
        self,
        clips: List[Dict],
        input_gcs_uri: str,
        audio_gcs_uri: Optional[str],
        project_id: str,
    ) -> List[Optional[str]]:
        """Publish one clip job per clip (contract C7).

        Each clip dict must already carry a server-generated ``clip_id``
        (uuid4 hex) plus ``start``/``end``/``reason``. The job references the
        source by ``gs://`` path (read with the processor's service account), not
        by a signed URL that could expire while the job waits in the queue (SC-2).

        Returns the Pub/Sub message IDs (None for a publish that failed).
        """
        if not isinstance(input_gcs_uri, str) or not input_gcs_uri.startswith("gs://"):
            raise ValueError("input_gcs_uri must be a gs:// URI")

        started = time.perf_counter()
        publish_futures = []
        for clip_meta in clips:
            message_data = {
                "clip_id": clip_meta["clip_id"],
                "input_gcs_uri": input_gcs_uri,
                "audio_gcs_uri": audio_gcs_uri,
                "start": clip_meta.get("start"),
                "end": clip_meta.get("end"),
                "project_id": project_id,
                "reason": clip_meta.get("reason", ""),
                "duration": clip_meta.get("end", 0) - clip_meta.get("start", 0),
            }
            publish_futures.append(self._publish(message_data))

        message_ids: List[Optional[str]] = []
        for future in publish_futures:
            try:
                message_ids.append(future.result(timeout=60))
            except Exception as e:
                logger.error("Error publishing message: %s", e)
                message_ids.append(None)

        successful = len([mid for mid in message_ids if mid is not None])
        failed = len(message_ids) - successful
        obs.log_event(
            "pubsub_publish_done",
            logging.WARNING if failed else logging.INFO,
            f"Published {successful}/{len(clips)} clip jobs to Pub/Sub for project {project_id}",
            topic=self.topic_name,
            messages=successful,
            failed=failed,
            duration_ms=obs.ms_since(started),
            status="ok" if not failed else "partial" if successful else "error",
        )
        return message_ids


# ============================================================
# Helper Functions
# ============================================================


def create_clip_publisher(
    project_id: str = None,
    topic_name: str = None,
) -> ClipJobPublisher:
    """Factory function to create publisher.

    Args:
        project_id: GCP project ID
        topic_name: Pub/Sub topic name

    Returns:
        ClipJobPublisher instance
    """
    return ClipJobPublisher(
        project_id=project_id or os.getenv("GCP_PROJECT_ID", "new-backend-475414"),
        topic_name=topic_name or os.getenv("CLIP_TOPIC", "clip-processing-jobs"),
    )

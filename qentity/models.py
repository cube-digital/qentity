from typing import Dict, List, Optional, Union
import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Unpack
from qdrant_client.http.models import PointStruct, Record


class BasePointModel(BaseModel):
    """Base class for point model definitions"""

    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    vector: Optional[Dict[str, List[float]]] = None

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    class Settings:
        point_type: str

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        """Ensure subclasses define point_type in Settings"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "Settings") and not hasattr(cls.Settings, "point_type"):
            raise TypeError(f"{cls.__name__} must define Settings.point_type")

    @classmethod
    def from_qdrant(cls, point: Union[PointStruct, Record, Dict]) -> "BasePointModel":
        """Convert Qdrant point to model instance"""

        if isinstance(point, dict):
            point_id = point.get("id")
            payload = point.get("payload", {})
            vector = point.get("vector")

        elif hasattr(point, "id") and hasattr(point, "payload"):
            point_id = point.id
            payload = point.payload or {}
            vector = getattr(point, "vector", None)

        else:
            raise ValueError(f"Unknown point format: {type(point)}")

        payload_copy = payload.copy()
        payload_copy.pop("point_type", None)

        instance = cls(id=uuid.UUID(point_id), **payload_copy)

        if vector is not None:
            instance.vector = vector

        return instance

    def to_qdrant(self, include_vector: bool = True) -> Dict[str, Any]:
        """Convert model instance to Qdrant point format"""

        # Use mode="json" to ensure datetime objects are serialized to ISO strings
        payload = self.model_dump(exclude={"id", "vector"}, mode="json")
        if hasattr(self, "Settings") and hasattr(self.Settings, "point_type"):
            payload["point_type"] = self.Settings.point_type

        point_data = {"id": str(self.id), "payload": payload}

        if include_vector and self.vector is not None:
            point_data["vector"] = self.vector

        return point_data
    

class TimestampMixin:
    """Mixin for timestamp fields"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class EmbeddingMixin:
    """Mixin to auto-generate embeddings from content"""

    @property
    def embedding_config(self) -> Dict[str, str]:
        return {}

    @property
    def embedding_content(self) -> str:
        """Override to define what content to embed"""
        raise NotImplementedError

    async def generate_embedding(
        self, embedding_services: Dict[str, EmbeddingModel]
    ) -> Dict[str, List[float]]:
        """Generate embedding for the content"""
        content = self.embedding_content
        vectors = {}
        for vector_name, service in embedding_services.items():
            try:
                # TODO: check for embedding services to not be async
                vector = service.embed(content)
                vectors[vector_name] = vector
            except Exception as e:
                logger.error(f"Failed to embed {vector_name}: {e}")
                continue
        return vectors
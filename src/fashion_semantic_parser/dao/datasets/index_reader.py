"""Read lightweight dataset JSONL indexes."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.dao.datasets.indexes import DatasetIndexManifest


class DatasetIndexReader:
    """Reader for generated dataset index manifests and JSONL files."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize the reader from a manifest path.

        Args:
            manifest_path: Path to the generated ``manifest.json`` file.
        """
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest(manifest_path)

    def list_indexes(self) -> list[str]:
        """List available index names from the manifest.

        Returns:
            Index names in manifest order.
        """
        return [index_file.name for index_file in self.manifest.files]

    def record_counts(self) -> dict[str, int]:
        """Return record counts declared by the manifest.

        Returns:
            Mapping from index name to record count.
        """
        return {
            index_file.name: index_file.record_count
            for index_file in self.manifest.files
        }

    def iter_records(
        self,
        index_name: str,
        category_name: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate records from one index file.

        Args:
            index_name: Index file name from the manifest.
            category_name: Optional DeepFashion2 category filter.
            limit: Optional maximum number of matching records to return.

        Yields:
            JSONL index records.
        """
        output_count = 0
        with self._index_path(index_name).open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if category_name is not None and not _has_category(
                    record,
                    category_name,
                ):
                    continue

                yield record
                output_count += 1
                if limit is not None and output_count >= limit:
                    return

    def _index_path(self, index_name: str) -> Path:
        """Resolve one index path by name.

        Args:
            index_name: Index file name from the manifest.

        Returns:
            Absolute path to the JSONL index file.

        Raises:
            KeyError: If the index name is not present in the manifest.
        """
        for index_file in self.manifest.files:
            if index_file.name == index_name:
                index_path = Path(index_file.path)
                if index_path.is_absolute():
                    return index_path
                return Path(resolve_project_path(index_path))
        raise KeyError(f"Unknown dataset index: {index_name}")

    @staticmethod
    def _load_manifest(manifest_path: Path) -> DatasetIndexManifest:
        """Load a dataset index manifest.

        Args:
            manifest_path: Manifest file path.

        Returns:
            Parsed manifest model.
        """
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest_data: dict[str, Any] = json.load(file)
        return DatasetIndexManifest.model_validate(manifest_data)


def _has_category(record: dict[str, Any], category_name: str) -> bool:
    """Check whether an index record contains the requested category."""
    items = record.get("items", [])
    if not isinstance(items, list):
        return False
    for item in items:
        if isinstance(item, dict) and item.get("category_name") == category_name:
            return True
    return False

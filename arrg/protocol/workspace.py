"""Shared workspace for agents to store and retrieve large artifacts."""

from typing import Any, Dict, Optional
import json
from pathlib import Path


class SharedWorkspace:
    """
    Shared workspace for passing references to large artifacts between agents.
    Prevents context window overflow by storing data and passing only references.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        """
        Initialize the shared workspace.
        
        Args:
            workspace_dir: Directory to store workspace artifacts. 
                          If None, uses in-memory storage.
        """
        self._storage: Dict[str, Any] = {}
        self.workspace_dir = workspace_dir
        if workspace_dir:
            workspace_dir.mkdir(parents=True, exist_ok=True)

    def store(self, key: str, data: Any, persist: bool = False) -> str:
        """
        Store data in the workspace.
        
        Args:
            key: Identifier for the data
            data: Data to store
            persist: If True and workspace_dir exists, persist to disk
            
        Returns:
            Reference key for retrieving the data
        """
        self._storage[key] = data
        
        if persist and self.workspace_dir:
            file_path = self.workspace_dir / f"{key}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        return key

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data from the workspace.
        
        Args:
            key: Identifier for the data
            
        Returns:
            Stored data or None if not found
        """
        # Try in-memory first
        if key in self._storage:
            return self._storage[key]
        
        # Try disk if workspace_dir exists
        if self.workspace_dir:
            file_path = self.workspace_dir / f"{key}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._storage[key] = data
                    return data
        
        return None

    def delete(self, key: str) -> bool:
        """
        Delete data from the workspace.
        
        Args:
            key: Identifier for the data
            
        Returns:
            True if data was deleted, False if not found
        """
        deleted = False
        
        if key in self._storage:
            del self._storage[key]
            deleted = True
        
        if self.workspace_dir:
            file_path = self.workspace_dir / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
                deleted = True
        
        return deleted

    def list_keys(self) -> list[str]:
        """List all keys in the workspace."""
        keys = set(self._storage.keys())
        
        if self.workspace_dir:
            for file_path in self.workspace_dir.glob("*.json"):
                keys.add(file_path.stem)
        
        return sorted(keys)

    def clear(self):
        """Clear all data from the workspace."""
        self._storage.clear()
        
        if self.workspace_dir:
            for file_path in self.workspace_dir.glob("*.json"):
                file_path.unlink()

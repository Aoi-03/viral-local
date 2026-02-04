"""
Caching and resource management utilities for the Viral-Local system.

This module provides caching mechanisms for intermediate results and resource
management for large video files to optimize performance and avoid redundant processing.
"""

import os
import json
import pickle
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Dict, List, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager

from ..models import VideoFile, AudioFile, Transcription, TranslatedSegment
from .errors import ProcessingError

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    data: Any
    created_at: float
    accessed_at: float
    size_bytes: int
    ttl: Optional[float] = None  # Time to live in seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at


class CacheManager:
    """Manages caching of intermediate processing results."""
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_size_mb: int = 1000,
        default_ttl: Optional[float] = None,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default time-to-live for cache entries in seconds
            cleanup_interval: Interval between automatic cleanup runs in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self.logger = logging.getLogger(__name__)
        self._last_cleanup = time.time()
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = {
                        key: CacheEntry(**entry_data) 
                        for key, entry_data in data.items()
                    }
            else:
                self.metadata = {}
        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                key: asdict(entry) 
                for key, entry in self.metadata.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a deterministic string from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        
        # Generate hash
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.cache"
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get the size of a file in bytes."""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0
    
    def _cleanup_if_needed(self):
        """Run cleanup if enough time has passed."""
        if time.time() - self._last_cleanup > self.cleanup_interval:
            self.cleanup()
            self._last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        self._cleanup_if_needed()
        
        if key not in self.metadata:
            return None
        
        entry = self.metadata[key]
        
        # Check if expired
        if entry.is_expired:
            self.delete(key)
            return None
        
        # Load data from file
        file_path = self._get_file_path(key)
        if not file_path.exists():
            # File missing, remove from metadata
            del self.metadata[key]
            self._save_metadata()
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            entry.accessed_at = time.time()
            self._save_metadata()
            
            self.logger.debug(f"Cache hit for key: {key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load cached data for key {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, data: Any, ttl: Optional[float] = None) -> bool:
        """Store an item in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (overrides default)
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            file_path = self._get_file_path(key)
            
            # Serialize data to file
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Get file size
            size_bytes = self._get_file_size(file_path)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=None,  # Don't store data in memory
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            self.metadata[key] = entry
            self._save_metadata()
            
            # Check if we need to free up space
            self._ensure_space()
            
            self.logger.debug(f"Cached data for key: {key} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache data for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            # Remove file
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            
            # Remove from metadata
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
            
            self.logger.debug(f"Deleted cache entry: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry {key}: {e}")
            return False
    
    def _ensure_space(self):
        """Ensure cache doesn't exceed size limit by removing old entries."""
        total_size = sum(entry.size_bytes for entry in self.metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        self.logger.info(f"Cache size ({total_size} bytes) exceeds limit ({self.max_size_bytes} bytes), cleaning up...")
        
        # Sort entries by last access time (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].accessed_at
        )
        
        # Remove entries until we're under the limit
        for key, entry in sorted_entries:
            if total_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                break
            
            self.delete(key)
            total_size -= entry.size_bytes
            self.logger.debug(f"Removed cache entry {key} to free space")
    
    def cleanup(self):
        """Remove expired and orphaned cache entries."""
        self.logger.info("Running cache cleanup...")
        
        expired_keys = []
        orphaned_keys = []
        
        # Find expired entries
        for key, entry in self.metadata.items():
            if entry.is_expired:
                expired_keys.append(key)
            elif not self._get_file_path(key).exists():
                orphaned_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self.delete(key)
            self.logger.debug(f"Removed expired cache entry: {key}")
        
        # Remove orphaned entries
        for key in orphaned_keys:
            del self.metadata[key]
            self.logger.debug(f"Removed orphaned cache entry: {key}")
        
        if expired_keys or orphaned_keys:
            self._save_metadata()
        
        # Remove orphaned files
        cache_files = set(f.stem for f in self.cache_dir.glob("*.cache"))
        metadata_keys = set(self.metadata.keys())
        orphaned_files = cache_files - metadata_keys
        
        for file_key in orphaned_files:
            try:
                (self.cache_dir / f"{file_key}.cache").unlink()
                self.logger.debug(f"Removed orphaned cache file: {file_key}")
            except OSError:
                pass
        
        self.logger.info(f"Cache cleanup completed. Removed {len(expired_keys)} expired and {len(orphaned_keys)} orphaned entries")
    
    def clear(self):
        """Clear all cache entries."""
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            self.logger.info("Cache cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = sum(entry.size_bytes for entry in self.metadata.values())
        total_entries = len(self.metadata)
        
        if total_entries > 0:
            avg_size = total_size / total_entries
            avg_age = sum(entry.age_seconds for entry in self.metadata.values()) / total_entries
        else:
            avg_size = 0
            avg_age = 0
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_percent": (total_size / self.max_size_bytes) * 100,
            "average_entry_size_bytes": avg_size,
            "average_age_seconds": avg_age
        }


def cached(
    cache_manager: CacheManager,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results.
    
    Args:
        cache_manager: CacheManager instance
        ttl: Time-to-live for cached results
        key_func: Optional function to generate cache key from arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class ResourceManager:
    """Manages system resources and temporary files."""
    
    def __init__(
        self,
        temp_dir: str = "temp",
        max_memory_mb: int = 2000,
        cleanup_on_exit: bool = True
    ):
        """Initialize resource manager.
        
        Args:
            temp_dir: Directory for temporary files
            max_memory_mb: Maximum memory usage in megabytes
            cleanup_on_exit: Whether to cleanup temp files on exit
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_on_exit = cleanup_on_exit
        
        self.logger = logging.getLogger(__name__)
        self.temp_files: List[Path] = []
        
        # Register cleanup on exit
        if cleanup_on_exit:
            import atexit
            atexit.register(self.cleanup_temp_files)
    
    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "viral_local_"):
        """Context manager for temporary files.
        
        Args:
            suffix: File suffix/extension
            prefix: File prefix
            
        Yields:
            Path to temporary file
        """
        import tempfile
        
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(self.temp_dir)
        )
        os.close(fd)  # Close file descriptor, we just need the path
        
        temp_path = Path(temp_path)
        self.temp_files.append(temp_path)
        
        try:
            yield temp_path
        finally:
            # Clean up the temporary file
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if temp_path in self.temp_files:
                    self.temp_files.remove(temp_path)
            except OSError as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
    
    @contextmanager
    def temp_directory(self, prefix: str = "viral_local_"):
        """Context manager for temporary directories.
        
        Args:
            prefix: Directory prefix
            
        Yields:
            Path to temporary directory
        """
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(self.temp_dir)))
        
        try:
            yield temp_dir
        finally:
            # Clean up the temporary directory
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except OSError as e:
                self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    def cleanup_temp_files(self):
        """Clean up all tracked temporary files."""
        cleaned_count = 0
        
        for temp_file in self.temp_files[:]:  # Copy list to avoid modification during iteration
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned_count += 1
                self.temp_files.remove(temp_file)
            except OSError as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "max_mb": self.max_memory_bytes / (1024 * 1024)
            }
        except ImportError:
            self.logger.warning("psutil not available, cannot get memory usage")
            return {}
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits.
        
        Returns:
            True if memory usage is acceptable, False otherwise
        """
        memory_info = self.get_memory_usage()
        if not memory_info:
            return True  # Can't check, assume OK
        
        current_mb = memory_info.get("rss_mb", 0)
        max_mb = memory_info.get("max_mb", float('inf'))
        
        if current_mb > max_mb:
            self.logger.warning(f"Memory usage ({current_mb:.1f} MB) exceeds limit ({max_mb:.1f} MB)")
            return False
        
        return True
    
    def get_disk_usage(self, path: Optional[str] = None) -> Dict[str, float]:
        """Get disk usage statistics.
        
        Args:
            path: Path to check (defaults to temp directory)
            
        Returns:
            Dictionary with disk usage information
        """
        check_path = Path(path) if path else self.temp_dir
        
        try:
            stat = shutil.disk_usage(check_path)
            
            return {
                "total_gb": stat.total / (1024 ** 3),
                "used_gb": (stat.total - stat.free) / (1024 ** 3),
                "free_gb": stat.free / (1024 ** 3),
                "used_percent": ((stat.total - stat.free) / stat.total) * 100
            }
        except OSError as e:
            self.logger.error(f"Failed to get disk usage for {check_path}: {e}")
            return {}
    
    def ensure_disk_space(self, required_mb: float, path: Optional[str] = None) -> bool:
        """Ensure sufficient disk space is available.
        
        Args:
            required_mb: Required space in megabytes
            path: Path to check (defaults to temp directory)
            
        Returns:
            True if sufficient space is available, False otherwise
        """
        disk_info = self.get_disk_usage(path)
        if not disk_info:
            return True  # Can't check, assume OK
        
        free_mb = disk_info.get("free_gb", 0) * 1024
        
        if free_mb < required_mb:
            self.logger.error(f"Insufficient disk space: {free_mb:.1f} MB available, {required_mb:.1f} MB required")
            return False
        
        return True


# Global instances for easy access
_cache_manager: Optional[CacheManager] = None
_resource_manager: Optional[ResourceManager] = None


def get_cache_manager(config=None) -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        if config:
            _cache_manager = CacheManager(
                cache_dir=config.cache_dir,
                max_size_mb=getattr(config, 'cache_max_size_mb', 1000),
                default_ttl=getattr(config, 'cache_default_ttl', None)
            )
        else:
            _cache_manager = CacheManager()
    
    return _cache_manager


def get_resource_manager(config=None) -> ResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    
    if _resource_manager is None:
        if config:
            _resource_manager = ResourceManager(
                temp_dir=config.temp_dir,
                max_memory_mb=getattr(config, 'max_memory_mb', 2000)
            )
        else:
            _resource_manager = ResourceManager()
    
    return _resource_manager
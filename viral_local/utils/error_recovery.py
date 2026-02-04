"""
Error recovery and detailed logging utilities for the Viral-Local system.

This module provides comprehensive error recovery strategies, detailed error logging,
and system health monitoring capabilities.
"""

import os
import json
import time
import traceback
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

from .errors import ViralLocalError, ProcessingError, APIError
from .cache import get_resource_manager


@dataclass
class ErrorReport:
    """Detailed error report with context and recovery suggestions."""
    timestamp: str
    error_type: str
    error_message: str
    error_code: str
    stage: Optional[str]
    stack_trace: str
    system_info: Dict[str, Any]
    context: Dict[str, Any]
    recovery_suggestions: List[str]
    severity: str  # "low", "medium", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error report to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "medium"
    ) -> "ErrorReport":
        """Create error report from exception.
        
        Args:
            exception: Exception that occurred
            stage: Processing stage where error occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            ErrorReport instance
        """
        # Get system information
        system_info = _get_system_info()
        
        # Generate recovery suggestions
        recovery_suggestions = _generate_recovery_suggestions(exception, stage)
        
        # Get error details
        if isinstance(exception, ViralLocalError):
            error_code = exception.error_code
            error_type = exception.__class__.__name__
        else:
            error_code = "UNKNOWN_ERROR"
            error_type = exception.__class__.__name__
        
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            error_message=str(exception),
            error_code=error_code,
            stage=stage,
            stack_trace=traceback.format_exc(),
            system_info=system_info,
            context=context or {},
            recovery_suggestions=recovery_suggestions,
            severity=severity
        )


class ErrorLogger:
    """Advanced error logging with structured reporting."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        max_log_files: int = 10,
        max_file_size_mb: int = 50
    ):
        """Initialize error logger.
        
        Args:
            log_dir: Directory for error log files
            max_log_files: Maximum number of log files to keep
            max_file_size_mb: Maximum size per log file in MB
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_log_files = max_log_files
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        self.logger = logging.getLogger(__name__)
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_stage": {},
            "errors_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0}
        }
    
    def log_error(
        self,
        exception: Exception,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "medium"
    ) -> ErrorReport:
        """Log an error with detailed reporting.
        
        Args:
            exception: Exception that occurred
            stage: Processing stage where error occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            ErrorReport instance
        """
        # Create error report
        error_report = ErrorReport.from_exception(exception, stage, context, severity)
        
        # Update statistics
        self._update_stats(error_report)
        
        # Write to log file
        self._write_to_log(error_report)
        
        # Log to standard logger
        log_level = self._get_log_level(severity)
        self.logger.log(
            log_level,
            f"Error in {stage or 'unknown'}: {exception}",
            extra={"error_report": error_report.to_dict()}
        )
        
        return error_report
    
    def _update_stats(self, error_report: ErrorReport):
        """Update error statistics."""
        self.error_stats["total_errors"] += 1
        
        # By type
        error_type = error_report.error_type
        self.error_stats["errors_by_type"][error_type] = (
            self.error_stats["errors_by_type"].get(error_type, 0) + 1
        )
        
        # By stage
        if error_report.stage:
            stage = error_report.stage
            self.error_stats["errors_by_stage"][stage] = (
                self.error_stats["errors_by_stage"].get(stage, 0) + 1
            )
        
        # By severity
        self.error_stats["errors_by_severity"][error_report.severity] += 1
    
    def _write_to_log(self, error_report: ErrorReport):
        """Write error report to log file."""
        try:
            # Get current log file
            log_file = self._get_current_log_file()
            
            # Write error report
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(error_report.to_dict(), f)
                f.write('\n')
            
            # Check if we need to rotate logs
            self._rotate_logs_if_needed()
            
        except Exception as e:
            self.logger.error(f"Failed to write error log: {e}")
    
    def _get_current_log_file(self) -> Path:
        """Get the current log file path."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"errors_{timestamp}.jsonl"
    
    def _rotate_logs_if_needed(self):
        """Rotate log files if they exceed size limit."""
        log_file = self._get_current_log_file()
        
        if log_file.exists() and log_file.stat().st_size > self.max_file_size_bytes:
            # Rename current file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = self.log_dir / f"errors_{timestamp}.jsonl"
            log_file.rename(rotated_file)
            
            # Clean up old files
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove old log files beyond the limit."""
        log_files = sorted(
            self.log_dir.glob("errors_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Remove files beyond the limit
        for old_file in log_files[self.max_log_files:]:
            try:
                old_file.unlink()
                self.logger.debug(f"Removed old log file: {old_file}")
            except OSError:
                pass
    
    def _get_log_level(self, severity: str) -> int:
        """Get logging level for severity."""
        levels = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }
        return levels.get(severity, logging.WARNING)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return self.error_stats.copy()
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorReport]:
        """Get recent error reports.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent ErrorReport instances
        """
        errors = []
        
        # Get recent log files
        log_files = sorted(
            self.log_dir.glob("errors_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Read errors from files
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(errors) >= limit:
                            break
                        
                        try:
                            error_data = json.loads(line.strip())
                            errors.append(ErrorReport(**error_data))
                        except (json.JSONDecodeError, TypeError):
                            continue
                
                if len(errors) >= limit:
                    break
                    
            except OSError:
                continue
        
        return errors[:limit]


class RecoveryManager:
    """Manages error recovery strategies and system health."""
    
    def __init__(self, error_logger: Optional[ErrorLogger] = None):
        """Initialize recovery manager.
        
        Args:
            error_logger: Optional error logger instance
        """
        self.error_logger = error_logger or ErrorLogger()
        self.logger = logging.getLogger(__name__)
        
        # Recovery strategies
        self.recovery_strategies = {
            "download": self._recover_download_error,
            "transcription": self._recover_transcription_error,
            "translation": self._recover_translation_error,
            "audio_generation": self._recover_audio_generation_error,
            "video_assembly": self._recover_video_assembly_error
        }
        
        # System health metrics
        self.health_metrics = {
            "last_successful_operation": None,
            "consecutive_failures": 0,
            "system_status": "healthy"  # healthy, degraded, critical
        }
    
    def handle_error(
        self,
        exception: Exception,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle an error with logging and optional recovery.
        
        Args:
            exception: Exception that occurred
            stage: Processing stage where error occurred
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Determine severity
        severity = self._determine_severity(exception, stage)
        
        # Log the error
        error_report = self.error_logger.log_error(exception, stage, context, severity)
        
        # Update health metrics
        self._update_health_metrics(error_report)
        
        # Attempt recovery if requested
        if attempt_recovery and stage in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[stage](exception, context)
                if recovery_result is not None:
                    self.logger.info(f"Successfully recovered from error in {stage}")
                    self._record_successful_recovery()
                    return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {stage}: {recovery_error}")
        
        return None
    
    def _determine_severity(self, exception: Exception, stage: Optional[str]) -> str:
        """Determine error severity based on exception and stage.
        
        Args:
            exception: Exception that occurred
            stage: Processing stage
            
        Returns:
            Severity level string
        """
        # Critical errors
        if isinstance(exception, (SystemError, MemoryError, OSError)):
            return "critical"
        
        # High severity errors
        if isinstance(exception, (APIError, ProcessingError)):
            if stage in ["video_assembly", "audio_generation"]:
                return "high"
        
        # Medium severity errors (default)
        if isinstance(exception, ViralLocalError):
            return "medium"
        
        # Low severity errors
        if isinstance(exception, (ValueError, TypeError)):
            return "low"
        
        return "medium"
    
    def _update_health_metrics(self, error_report: ErrorReport):
        """Update system health metrics based on error report.
        
        Args:
            error_report: Error report to process
        """
        self.health_metrics["consecutive_failures"] += 1
        
        # Update system status based on consecutive failures
        if self.health_metrics["consecutive_failures"] >= 10:
            self.health_metrics["system_status"] = "critical"
        elif self.health_metrics["consecutive_failures"] >= 5:
            self.health_metrics["system_status"] = "degraded"
        
        # Critical errors immediately set status to critical
        if error_report.severity == "critical":
            self.health_metrics["system_status"] = "critical"
    
    def _record_successful_recovery(self):
        """Record a successful recovery operation."""
        self.health_metrics["last_successful_operation"] = time.time()
        self.health_metrics["consecutive_failures"] = 0
        self.health_metrics["system_status"] = "healthy"
    
    def _recover_download_error(self, exception: Exception, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Attempt to recover from download errors."""
        # Try alternative download methods or formats
        self.logger.info("Attempting download error recovery...")
        # Implementation would depend on specific download service
        return None
    
    def _recover_transcription_error(self, exception: Exception, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Attempt to recover from transcription errors."""
        # Try different Whisper model or audio preprocessing
        self.logger.info("Attempting transcription error recovery...")
        # Implementation would depend on specific transcription service
        return None
    
    def _recover_translation_error(self, exception: Exception, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Attempt to recover from translation errors."""
        # Try alternative API or simplified prompts
        self.logger.info("Attempting translation error recovery...")
        # Implementation would depend on specific translation service
        return None
    
    def _recover_audio_generation_error(self, exception: Exception, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Attempt to recover from audio generation errors."""
        # Try alternative TTS engine or voice settings
        self.logger.info("Attempting audio generation error recovery...")
        # Implementation would depend on specific TTS service
        return None
    
    def _recover_video_assembly_error(self, exception: Exception, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Attempt to recover from video assembly errors."""
        # Try alternative video processing settings
        self.logger.info("Attempting video assembly error recovery...")
        # Implementation would depend on specific video assembly service
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status.
        
        Returns:
            Dictionary with system health information
        """
        resource_manager = get_resource_manager()
        
        health_info = self.health_metrics.copy()
        health_info.update({
            "error_stats": self.error_logger.get_error_stats(),
            "memory_usage": resource_manager.get_memory_usage(),
            "disk_usage": resource_manager.get_disk_usage(),
            "timestamp": time.time()
        })
        
        return health_info


def _get_system_info() -> Dict[str, Any]:
    """Get system information for error reporting."""
    import platform
    import sys
    
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        cpu_info = {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent()
        }
    except ImportError:
        memory_info = None
        disk_info = None
        cpu_info = None
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "memory": {
            "total": memory_info.total if memory_info else None,
            "available": memory_info.available if memory_info else None,
            "percent": memory_info.percent if memory_info else None
        } if memory_info else None,
        "disk": {
            "total": disk_info.total if disk_info else None,
            "free": disk_info.free if disk_info else None,
            "used": disk_info.used if disk_info else None
        } if disk_info else None,
        "cpu": cpu_info,
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith(('VIRAL_LOCAL_', 'PYTHON', 'PATH'))
        }
    }


def _generate_recovery_suggestions(exception: Exception, stage: Optional[str]) -> List[str]:
    """Generate recovery suggestions based on error type and stage."""
    suggestions = []
    
    # General suggestions based on exception type
    if isinstance(exception, ConnectionError):
        suggestions.extend([
            "Check your internet connection",
            "Verify API endpoints are accessible",
            "Try again after a few minutes"
        ])
    
    elif isinstance(exception, APIError):
        suggestions.extend([
            "Verify your API keys are valid and not expired",
            "Check if you've exceeded API rate limits",
            "Ensure the API service is operational"
        ])
    
    elif isinstance(exception, MemoryError):
        suggestions.extend([
            "Close other applications to free up memory",
            "Try processing smaller video segments",
            "Restart the application"
        ])
    
    elif isinstance(exception, OSError):
        suggestions.extend([
            "Check available disk space",
            "Verify file permissions",
            "Ensure the file path is accessible"
        ])
    
    # Stage-specific suggestions
    if stage == "download":
        suggestions.extend([
            "Verify the YouTube URL is valid and accessible",
            "Check if the video is available in your region",
            "Try a different video format or quality"
        ])
    
    elif stage == "transcription":
        suggestions.extend([
            "Ensure the audio quality is sufficient",
            "Try a different Whisper model size",
            "Check if the audio file is corrupted"
        ])
    
    elif stage == "translation":
        suggestions.extend([
            "Verify the target language is supported",
            "Check if the source text is too long",
            "Try breaking the content into smaller segments"
        ])
    
    elif stage == "audio_generation":
        suggestions.extend([
            "Try a different TTS engine or voice",
            "Check if the text contains unsupported characters",
            "Verify audio output settings"
        ])
    
    elif stage == "video_assembly":
        suggestions.extend([
            "Ensure sufficient disk space for output",
            "Check video and audio file compatibility",
            "Try different output format settings"
        ])
    
    # Default suggestions if none specific
    if not suggestions:
        suggestions.extend([
            "Check the application logs for more details",
            "Restart the application and try again",
            "Contact support if the problem persists"
        ])
    
    return suggestions


# Global instances
_error_logger: Optional[ErrorLogger] = None
_recovery_manager: Optional[RecoveryManager] = None


def get_error_logger() -> ErrorLogger:
    """Get the global error logger instance."""
    global _error_logger
    
    if _error_logger is None:
        _error_logger = ErrorLogger()
    
    return _error_logger


def get_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_manager
    
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager(get_error_logger())
    
    return _recovery_manager
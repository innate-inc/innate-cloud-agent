"""
Unified Logger for Brain and Primitives

This module provides a centralized logging system that captures all logs
from the main brain agent and all primitives in a single, unified HTML view.
"""

import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
from dataclasses import dataclass, asdict


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    BRAIN = "BRAIN"
    PRIMITIVE = "PRIMITIVE"
    GEMINI_REQUEST = "GEMINI_REQUEST"
    GEMINI_RESPONSE = "GEMINI_RESPONSE"
    SYSTEM = "SYSTEM"


@dataclass
class LogEntry:
    timestamp: str
    level: LogLevel
    source: LogSource
    component: str  # e.g., "brain", "navigate_in_sight", "turn_and_move"
    message: str
    data: Optional[Dict[str, Any]] = None
    connection_id: Optional[str] = None
    robot_position: Optional[Dict[str, float]] = None  # x, y, theta
    image_data: Optional[str] = None  # base64 encoded image data


class UnifiedLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnifiedLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logs: List[LogEntry] = []
            self.max_logs = 100
            self.debug_dir = Path("debug_logs/unified")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.initialized = True

    def log(
        self,
        level: LogLevel,
        source: LogSource,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        connection_id: Optional[str] = None,
        robot_position: Optional[Dict[str, float]] = None,
        image_data: Optional[str] = None,
    ):
        """Add a log entry to the unified log."""
        # Filter out "processing message" logs
        if "Processing message:" in message:
            return
            
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            source=source,
            component=component,
            message=message,
            data=data,
            connection_id=connection_id,
            robot_position=robot_position,
            image_data=image_data,
        )

        with self._lock:
            self.logs.append(entry)
            # Keep only the last max_logs entries
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

        # Auto-generate HTML after each log (for real-time viewing)
        self._generate_html()

    def debug(self, source: LogSource, component: str, message: str, **kwargs):
        self.log(LogLevel.DEBUG, source, component, message, **kwargs)

    def info(self, source: LogSource, component: str, message: str, **kwargs):
        self.log(LogLevel.INFO, source, component, message, **kwargs)

    def warning(self, source: LogSource, component: str, message: str, **kwargs):
        self.log(LogLevel.WARNING, source, component, message, **kwargs)

    def error(self, source: LogSource, component: str, message: str, **kwargs):
        self.log(LogLevel.ERROR, source, component, message, **kwargs)

    def critical(self, source: LogSource, component: str, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, source, component, message, **kwargs)

    def log_gemini_request(
        self,
        component: str,
        prompt: str,
        image_data: Optional[str] = None,
        connection_id: Optional[str] = None,
        robot_position: Optional[Dict[str, float]] = None,
    ):
        """Log a Gemini API request."""
        data = {"prompt": prompt}
        if image_data:
            data["has_image"] = True
            # Don't store the full image data in the data field, just metadata
            if isinstance(image_data, str) and len(image_data) > 100:
                data["image_size"] = f"{len(image_data)} characters"
            else:
                data["image_size"] = str(image_data)
        
        self.log(
            LogLevel.INFO,
            LogSource.GEMINI_REQUEST,
            component,
            f"Gemini API request sent",
            data=data,
            connection_id=connection_id,
            robot_position=robot_position,
            image_data=image_data if isinstance(image_data, str) and len(image_data) > 100 else None,
        )

    def log_gemini_response(
        self,
        component: str,
        response: Dict[str, Any],
        connection_id: Optional[str] = None,
        robot_position: Optional[Dict[str, float]] = None,
    ):
        """Log a Gemini API response."""
        self.log(
            LogLevel.INFO,
            LogSource.GEMINI_RESPONSE,
            component,
            f"Gemini API response received",
            data=response,
            connection_id=connection_id,
            robot_position=robot_position,
        )

    def _generate_html(self):
        """Generate the unified HTML log view."""
        try:
            html_content = self._create_html_content()
            html_path = self.debug_dir / "unified_logs.html"
            
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"Failed to generate unified HTML log: {e}")

    def _create_html_content(self) -> str:
        """Create the HTML content for the unified log view."""
        entries_html = ""
        
        with self._lock:
            # Reverse to show newest first
            for entry in reversed(self.logs):
                entries_html += self._create_log_entry_html(entry)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Logs - Brain & Primitives</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0;
            font-size: 16px;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat {{
            background: rgba(255,255,255,0.2);
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
        }}
        .log-entry {{
            background: white;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            border-left: 4px solid #ddd;
        }}
        .log-entry.DEBUG {{ border-left-color: #6c757d; }}
        .log-entry.INFO {{ border-left-color: #17a2b8; }}
        .log-entry.WARNING {{ border-left-color: #ffc107; }}
        .log-entry.ERROR {{ border-left-color: #dc3545; }}
        .log-entry.CRITICAL {{ border-left-color: #6f42c1; }}
        .log-header {{
            padding: 12px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .log-meta {{
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 14px;
        }}
        .log-level {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .log-level.DEBUG {{ background: #6c757d; }}
        .log-level.INFO {{ background: #17a2b8; }}
        .log-level.WARNING {{ background: #ffc107; color: #000; }}
        .log-level.ERROR {{ background: #dc3545; }}
        .log-level.CRITICAL {{ background: #6f42c1; }}
        .log-source {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            background: #e9ecef;
            color: #495057;
        }}
        .log-source.BRAIN {{ background: #d4edda; color: #155724; }}
        .log-source.PRIMITIVE {{ background: #fff3cd; color: #856404; }}
        .log-source.GEMINI_REQUEST {{ background: #cce5ff; color: #004085; }}
        .log-source.GEMINI_RESPONSE {{ background: #d1ecf1; color: #0c5460; }}
        .log-source.SYSTEM {{ background: #f8d7da; color: #721c24; }}
        .log-component {{
            font-weight: bold;
            color: #495057;
        }}
        .log-timestamp {{
            font-size: 12px;
            color: #6c757d;
            font-family: monospace;
        }}
        .log-content {{
            padding: 20px;
        }}
        .log-message {{
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 15px;
            color: #212529;
        }}
        .log-data {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
        }}
        .log-data pre {{
            margin: 0;
            font-size: 14px;
            color: #495057;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .connection-id {{
            font-size: 12px;
            color: #6c757d;
            font-family: monospace;
        }}
        .robot-position {{
            font-size: 12px;
            color: #007bff;
            font-family: monospace;
            background: #e3f2fd;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .log-image {{
            max-width: 300px;
            max-height: 200px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .log-image:hover {{
            transform: scale(1.05);
        }}
        .log-image img {{
            max-width: 100%;
            max-height: 100%;
            display: block;
        }}
        .auto-refresh {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            z-index: 1000;
        }}
    </style>
    <script>
        // Auto-refresh every 5 seconds
        setTimeout(function() {{
            location.reload();
        }}, 5000);
    </script>
</head>
<body>
    <div class="auto-refresh">🔄 Auto-refresh: 5s</div>
    
    <div class="header">
        <h1>🤖 Unified Logs - Brain & Primitives</h1>
        <p>Real-time view of all system logs</p>
        <div class="stats">
            <div class="stat">Total Logs: {len(self.logs)}</div>
            <div class="stat">Max History: {self.max_logs}</div>
            <div class="stat">Last Updated: {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
    </div>

    <div class="logs-container">
        {entries_html}
    </div>
</body>
</html>"""

    def _create_log_entry_html(self, entry: LogEntry) -> str:
        """Create HTML for a single log entry."""
        data_html = ""
        if entry.data:
            try:
                data_json = json.dumps(entry.data, indent=2, ensure_ascii=False)
                data_html = f"""
                <div class="log-data">
                    <pre>{self._escape_html(data_json)}</pre>
                </div>"""
            except Exception:
                data_html = f"""
                <div class="log-data">
                    <pre>{self._escape_html(str(entry.data))}</pre>
                </div>"""

        connection_html = ""
        if entry.connection_id:
            connection_html = f'<span class="connection-id">Connection: {entry.connection_id}</span>'

        robot_position_html = ""
        if entry.robot_position:
            x = entry.robot_position.get("x", 0.0)
            y = entry.robot_position.get("y", 0.0)
            theta_rad = entry.robot_position.get("theta", 0.0)
            theta_deg = theta_rad * 180.0 / math.pi  # Convert radians to degrees
            robot_position_html = f'<span class="robot-position">Robot: ({x:.2f}, {y:.2f}, θ={theta_deg:.1f}°)</span>'

        image_html = ""
        if entry.image_data:
            image_html = f"""
            <div class="log-image-container">
                <img class="log-image" src="data:image/jpeg;base64,{self._escape_html(entry.image_data)}" alt="Log Image" onclick="window.open('data:image/jpeg;base64,{self._escape_html(entry.image_data)}', '_blank')">
            </div>
            """

        return f"""
        <div class="log-entry {entry.level.value}">
            <div class="log-header">
                <div class="log-meta">
                    <span class="log-level {entry.level.value}">{entry.level.value}</span>
                    <span class="log-source {entry.source.value}">{entry.source.value}</span>
                    <span class="log-component">{entry.component}</span>
                    {connection_html}
                    {robot_position_html}
                </div>
                <div class="log-timestamp">{entry.timestamp}</div>
            </div>
            <div class="log-content">
                <div class="log-message">{self._escape_html(entry.message)}</div>
                {data_html}
                {image_html}
            </div>
        </div>"""

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


# Global instance
unified_logger = UnifiedLogger() 
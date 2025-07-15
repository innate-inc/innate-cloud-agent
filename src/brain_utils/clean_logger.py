"""
Clean Logger for Essential Information Only

This module provides a focused logging system that captures only the most
important information: images, thoughts, and decisions.
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from dataclasses import dataclass


@dataclass
class CleanLogEntry:
    timestamp: str
    entry_type: str  # "vision_decision", "navigate_decision", "user_message"
    component: str   # "brain" or "navigate_in_sight"
    message: str
    data: Optional[Dict[str, Any]] = None
    image_b64: Optional[str] = None
    connection_id: Optional[str] = None


class CleanLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CleanLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logs: List[CleanLogEntry] = []
            self.max_logs = 50  # Reduced for cleaner view
            self.debug_dir = Path("debug_logs/clean")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.initialized = True

    def log_vision_decision(
        self,
        thoughts: str,
        observation: str,
        decision: str,
        anticipation: str = "",
        image_b64: Optional[str] = None,
        connection_id: Optional[str] = None,
    ):
        """Log a vision agent decision with thoughts and image."""
        entry = CleanLogEntry(
            timestamp=datetime.now().isoformat(),
            entry_type="vision_decision",
            component="brain",
            message=f"Decision: {decision}",
            data={
                "thoughts": thoughts,
                "observation": observation,
                "decision": decision,
                "anticipation": anticipation,
            },
            image_b64=image_b64,
            connection_id=connection_id,
        )
        self._add_entry(entry)

    def log_navigate_decision(
        self,
        target: str,
        prompt: str,
        response: Dict[str, Any],
        selected_point: str,
        image_b64: Optional[str] = None,
        connection_id: Optional[str] = None,
    ):
        """Log a navigate_in_sight decision with prompt and response."""
        entry = CleanLogEntry(
            timestamp=datetime.now().isoformat(),
            entry_type="navigate_decision",
            component="navigate_in_sight",
            message=f"Target: {target} → Point {selected_point}",
            data={
                "target": target,
                "prompt": prompt,
                "response": response,
                "selected_point": selected_point,
            },
            image_b64=image_b64,
            connection_id=connection_id,
        )
        self._add_entry(entry)

    def log_user_message(
        self,
        message: str,
        connection_id: Optional[str] = None,
    ):
        """Log a user message."""
        entry = CleanLogEntry(
            timestamp=datetime.now().isoformat(),
            entry_type="user_message",
            component="brain",
            message=f"User: {message}",
            data={"message": message},
            connection_id=connection_id,
        )
        self._add_entry(entry)

    def log_turn_and_move(
        self,
        angle_degrees: float,
        distance: float,
        connection_id: Optional[str] = None,
    ):
        """Log a turn_and_move action."""
        entry = CleanLogEntry(
            timestamp=datetime.now().isoformat(),
            entry_type="turn_and_move",
            component="turn_and_move",
            message=f"Turn {angle_degrees}° and move {distance}m",
            data={
                "angle_degrees": angle_degrees,
                "distance": distance,
            },
            connection_id=connection_id,
        )
        self._add_entry(entry)

    def _add_entry(self, entry: CleanLogEntry):
        """Add an entry and maintain the log limit."""
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
        
        # Generate HTML after each important entry
        self._generate_html()

    def _generate_html(self):
        """Generate the clean HTML log view."""
        try:
            html_content = self._create_html_content()
            html_path = self.debug_dir / "clean_logs.html"
            
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"Failed to generate clean HTML log: {e}")

    def _create_html_content(self) -> str:
        """Create the HTML content for the clean log view."""
        entries_html = ""
        
        with self._lock:
            # Show newest first
            for entry in reversed(self.logs):
                entries_html += self._create_log_entry_html(entry)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clean Logs - Essential Information</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #4a90e2 0%, #7b68ee 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }}
        .header p {{
            margin: 8px 0 0;
            font-size: 16px;
            opacity: 0.9;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
        }}
        .log-entry {{
            background: white;
            margin: 20px 0;
            border-radius: 12px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.08);
            overflow: hidden;
            border-left: 4px solid #ddd;
        }}
        .log-entry.vision_decision {{ border-left-color: #4a90e2; }}
        .log-entry.navigate_decision {{ border-left-color: #28a745; }}
        .log-entry.turn_and_move {{ border-left-color: #17a2b8; }}
        .log-entry.user_message {{ border-left-color: #ffc107; }}
        .log-header {{
            padding: 16px 24px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .log-meta {{
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .log-type {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .log-type.vision_decision {{ background: #e3f2fd; color: #1976d2; }}
        .log-type.navigate_decision {{ background: #e8f5e8; color: #2e7d32; }}
        .log-type.turn_and_move {{ background: #e0f2f1; color: #00695c; }}
        .log-type.user_message {{ background: #fff3e0; color: #f57c00; }}
        .log-component {{
            font-weight: 600;
            color: #495057;
            font-size: 14px;
        }}
        .log-timestamp {{
            font-size: 12px;
            color: #6c757d;
            font-family: 'SF Mono', Monaco, monospace;
        }}
        .log-content {{
            padding: 24px;
        }}
        .log-message {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: #212529;
        }}
        .log-image {{
            margin: 16px 0;
            text-align: center;
        }}
        .log-image img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .thoughts-section {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 4px solid #17a2b8;
        }}
        .thoughts-title {{
            font-weight: 600;
            color: #17a2b8;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .thoughts-text {{
            color: #495057;
            line-height: 1.6;
            font-size: 15px;
        }}
        .observation-section {{
            background: #fff3cd;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 4px solid #ffc107;
        }}
        .observation-title {{
            font-weight: 600;
            color: #856404;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .observation-text {{
            color: #495057;
            line-height: 1.6;
            font-size: 15px;
        }}
        .anticipation-section {{
            background: #e1f5fe;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 4px solid #0277bd;
        }}
        .anticipation-title {{
            font-weight: 600;
            color: #01579b;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .anticipation-text {{
            color: #495057;
            line-height: 1.6;
            font-size: 15px;
        }}
        .navigate-details {{
            background: #e8f5e8;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 4px solid #28a745;
        }}
        .navigate-title {{
            font-weight: 600;
            color: #155724;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .navigate-text {{
            color: #495057;
            line-height: 1.6;
            font-size: 15px;
        }}
        .prompt-section {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: #495057;
            white-space: pre-wrap;
        }}
        .response-section {{
            background: #d4edda;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }}
        .response-title {{
            font-weight: 600;
            color: #155724;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .response-text {{
            color: #495057;
            line-height: 1.6;
            font-size: 15px;
        }}
        .auto-refresh {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 10px 16px;
            border-radius: 25px;
            font-size: 12px;
            font-weight: 600;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .scroll-indicator {{
            position: fixed;
            bottom: 70px;
            right: 20px;
            background: #6c757d;
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 11px;
            z-index: 1000;
            display: none;
        }}
    </style>
    <script>
        let lastScrollPosition = 0;
        let isUserScrolling = false;
        
        // Track user scrolling
        window.addEventListener('scroll', function() {{
            isUserScrolling = true;
            lastScrollPosition = window.scrollY;
            
            // Show scroll indicator if not at top
            const scrollIndicator = document.querySelector('.scroll-indicator');
            const isAtTop = window.scrollY <= 50; // Within 50px of top
            
            if (isAtTop) {{
                scrollIndicator.style.display = 'none';
            }} else {{
                scrollIndicator.style.display = 'block';
            }}
        }});
        
        // Auto-refresh with smart scrolling
        setTimeout(function() {{
            const currentScrollPosition = window.scrollY;
            const isAtTop = currentScrollPosition <= 50; // Within 50px of top
            
            // Stay at top if user was at top, otherwise preserve position
            if (isAtTop || !isUserScrolling) {{
                sessionStorage.setItem('shouldStayAtTop', 'true');
            }} else {{
                sessionStorage.setItem('shouldStayAtTop', 'false');
                sessionStorage.setItem('scrollPosition', currentScrollPosition.toString());
            }}
            
            location.reload();
        }}, 8000);
        
        // Restore scroll position after reload
        window.addEventListener('load', function() {{
            const shouldStayAtTop = sessionStorage.getItem('shouldStayAtTop');
            const savedScrollPosition = sessionStorage.getItem('scrollPosition');
            
            if (shouldStayAtTop === 'true') {{
                window.scrollTo(0, 0); // Stay at top
            }} else if (savedScrollPosition) {{
                window.scrollTo(0, parseInt(savedScrollPosition));
            }}
        }});
    </script>
</head>
<body>
    <div class="auto-refresh">🔄 Auto-refresh: 8s</div>
    <div class="scroll-indicator">📍 Scroll position saved</div>
    
    <div class="header">
        <h1>🎯 Clean Logs - Essential Information</h1>
        <p>Images, thoughts, and decisions from brain and navigation</p>
        <div class="stats">
            <div class="stat">Entries: {len(self.logs)}</div>
            <div class="stat">Updated: {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
    </div>

    <div class="logs-container">
        {entries_html}
    </div>
</body>
</html>"""

    def _create_log_entry_html(self, entry: CleanLogEntry) -> str:
        """Create HTML for a single log entry."""
        image_html = ""
        if entry.image_b64:
            image_html = f"""
            <div class="log-image">
                <img src="data:image/jpeg;base64,{entry.image_b64}" alt="Context image" />
            </div>"""

        content_html = ""
        
        if entry.entry_type == "vision_decision" and entry.data:
            thoughts = entry.data.get("thoughts", "")
            observation = entry.data.get("observation", "")
            anticipation = entry.data.get("anticipation", "")
            
            anticipation_html = ""
            if anticipation:
                anticipation_html = f"""
                <div class="anticipation-section">
                    <div class="anticipation-title">🔮 Anticipation</div>
                    <div class="anticipation-text">{self._escape_html(anticipation)}</div>
                </div>"""
            
            content_html = f"""
            {image_html}
            <div class="thoughts-section">
                <div class="thoughts-title">🧠 Thoughts</div>
                <div class="thoughts-text">{self._escape_html(thoughts)}</div>
            </div>
            <div class="observation-section">
                <div class="observation-title">👁️ Observation</div>
                <div class="observation-text">{self._escape_html(observation)}</div>
            </div>
            {anticipation_html}"""
            
        elif entry.entry_type == "navigate_decision" and entry.data:
            target = entry.data.get("target", "")
            prompt = entry.data.get("prompt", "")
            response = entry.data.get("response", {})
            explanation = response.get("explanation", "")
            
            content_html = f"""
            {image_html}
            <div class="navigate-details">
                <div class="navigate-title">🎯 Navigation Target</div>
                <div class="navigate-text">{self._escape_html(target)}</div>
            </div>
            <div class="prompt-section">{self._escape_html(prompt)}</div>
            <div class="response-section">
                <div class="response-title">🤖 AI Decision</div>
                <div class="response-text">{self._escape_html(explanation)}</div>
            </div>"""
            
        elif entry.entry_type == "turn_and_move" and entry.data:
            angle_degrees = entry.data.get("angle_degrees", 0)
            distance = entry.data.get("distance", 0)
            
            # Create descriptive text
            turn_desc = ""
            if angle_degrees != 0:
                direction = "counterclockwise" if angle_degrees > 0 else "clockwise"
                turn_desc = f"Turn {abs(angle_degrees)}° {direction}"
            
            move_desc = ""
            if distance != 0:
                move_desc = f"Move {distance}m forward"
            
            action_desc = ""
            if turn_desc and move_desc:
                action_desc = f"{turn_desc}, then {move_desc}"
            elif turn_desc:
                action_desc = turn_desc
            elif move_desc:
                action_desc = move_desc
            else:
                action_desc = "No movement"
            
            content_html = f"""
            <div class="navigate-details">
                <div class="navigate-title">🔄 Turn and Move Action</div>
                <div class="navigate-text">{self._escape_html(action_desc)}</div>
            </div>"""
            
        elif entry.entry_type == "user_message" and entry.data:
            message = entry.data.get("message", "")
            content_html = f"""
            <div class="navigate-details">
                <div class="navigate-title">💬 User Message</div>
                <div class="navigate-text">{self._escape_html(message)}</div>
            </div>"""

        connection_html = ""
        if entry.connection_id:
            connection_html = f' • {entry.connection_id}'

        return f"""
        <div class="log-entry {entry.entry_type}">
            <div class="log-header">
                <div class="log-meta">
                    <span class="log-type {entry.entry_type}">{entry.entry_type.replace('_', ' ')}</span>
                    <span class="log-component">{entry.component}</span>
                </div>
                <div class="log-timestamp">{entry.timestamp.split('T')[1][:8]}{connection_html}</div>
            </div>
            <div class="log-content">
                <div class="log-message">{self._escape_html(entry.message)}</div>
                {content_html}
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
clean_logger = CleanLogger() 
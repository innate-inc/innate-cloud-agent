"""
HTML Debug Generator for Gemini Content Parts

This module provides functionality to generate beautiful HTML visualizations
of content parts sent to the Gemini API for debugging purposes.
"""

import os
import json
import base64
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime


def save_content_parts_html(
    content_parts: List[Union[str, dict]],
    filename: str,
    debug_dir: Path,
    response_data: Optional[dict] = None,
) -> str:
    """
    Save content parts to an HTML file for debugging, showing actual content as seen by the model.
    Images are embedded as base64 data URLs for visual inspection.

    Args:
        content_parts: List of content parts (text, images, dicts, etc.)
        filename: Base filename for the HTML file
        debug_dir: Directory to save the debug file in

    Returns:
        Path to the saved HTML file, or empty string if failed
    """
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)

        file_path = (
            debug_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            # Write HTML header
            f.write(_get_html_header())

            for i, part in enumerate(content_parts):
                f.write(f'    <div class="part">\n')
                f.write(f'        <div class="part-header">PART {i + 1}</div>\n')

                if isinstance(part, str):
                    _write_text_part(f, part)

                elif hasattr(part, "__class__") and "Part" in str(part.__class__):
                    _write_gemini_part(f, part)

                elif isinstance(part, dict):
                    _write_dict_part(f, part)

                else:
                    _write_unknown_part(f, part)

                f.write(f"    </div>\n\n")

            if response_data:
                _write_response_part(f, response_data)

            # Write HTML footer
            f.write(_get_html_footer())

        print(f"Saved content parts HTML to {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"Error saving content parts HTML: {e}")
        return ""


def _get_html_header() -> str:
    """Get the HTML header with embedded CSS styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Content Parts - Debug View</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
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
            font-size: 28px;
        }}
        .header p {{
            margin: 5px 0 0;
            font-size: 16px;
        }}
        .part {{
            background: white;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .part-header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
        }}
        .part-meta {{
            background-color: #34495e;
            color: #ecf0f1;
            padding: 10px 20px;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }}
        .part-content {{
            padding: 20px;
        }}
        .text-content {{
            white-space: pre-wrap;
            font-size: 16px;
            line-height: 1.6;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .json-content {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            overflow-x: auto;
        }}
        .image-content {{
            text-align: center;
            padding: 20px;
        }}
        .image-content img {{
            max-width: 100%;
            max-height: 600px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .image-info {{
            margin-top: 15px;
            padding: 10px;
            background-color: #e8f4fd;
            border-radius: 5px;
            font-size: 14px;
            color: #2c5282;
        }}
        .error-content {{
            background-color: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #e53e3e;
        }}
        .response-header {{
            background-color: #27ae60; /* A nice green for success */
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Gemini Content Parts - Debug View</h1>
        <p>Visual representation of content sent to Gemini API</p>
        <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
"""


def _get_html_footer() -> str:
    """Get the HTML footer."""
    return """</body>
</html>"""


def _write_text_part(f, part: str):
    """Write a text part to the HTML file."""
    f.write(f'        <div class="part-meta">TYPE: Text</div>\n')
    f.write(f'        <div class="part-content">\n')
    f.write(f'            <div class="text-content">{escape_html(part)}</div>\n')
    f.write(f"        </div>\n")


def _write_gemini_part(f, part):
    """Write a Gemini Part object (image/file) to the HTML file."""
    # Extract metadata from the Part object
    mime_type = "unknown"
    data_size = 0
    image_data = None

    if hasattr(part, "inline_data") and part.inline_data:
        blob = part.inline_data
        if hasattr(blob, "mime_type"):
            mime_type = blob.mime_type
        if hasattr(blob, "data") and blob.data:
            data_size = len(blob.data)
            image_data = blob.data

    f.write(
        f'        <div class="part-meta">TYPE: {type(part).__name__} | MIME: {mime_type} | SIZE: {data_size} bytes</div>\n'
    )
    f.write(f'        <div class="part-content">\n')

    if image_data and mime_type.startswith("image/"):
        _write_image_content(f, image_data, mime_type, data_size)
    else:
        _write_non_image_content(f, part, mime_type, data_size)

    f.write(f"        </div>\n")


def _write_image_content(f, image_data: bytes, mime_type: str, data_size: int):
    """Write image content embedded as base64 data URL."""
    b64_data = base64.b64encode(image_data).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"

    f.write(f'            <div class="image-content">\n')
    f.write(f'                <img src="{data_url}" alt="Embedded image content" />\n')
    f.write(f'                <div class="image-info">\n')
    f.write(f"                    <strong>Image Details:</strong><br/>\n")
    f.write(f"                    MIME Type: {mime_type}<br/>\n")
    f.write(f"                    File Size: {data_size:,} bytes<br/>\n")
    f.write(f"                    This is exactly what the Gemini model sees\n")
    f.write(f"                </div>\n")
    f.write(f"            </div>\n")


def _write_non_image_content(f, part, mime_type: str, data_size: int):
    """Write non-image content with error styling."""
    f.write(f'            <div class="error-content">\n')
    f.write(
        f"                <strong>Non-image data or data not accessible</strong><br/>\n"
    )
    f.write(f"                Type: {type(part).__name__}<br/>\n")
    f.write(f"                MIME Type: {mime_type}<br/>\n")
    f.write(f"                Data Size: {data_size} bytes<br/>\n")
    f.write(
        f"                Raw data not displayed (not an image or data inaccessible)\n"
    )
    f.write(f"            </div>\n")


def _write_dict_part(f, part: dict):
    """Write a dictionary part with JSON formatting."""
    f.write(f'        <div class="part-meta">TYPE: Dictionary</div>\n')
    f.write(f'        <div class="part-content">\n')
    f.write(f'            <div class="json-content">\n')
    json_str = json.dumps(part, indent=2, ensure_ascii=False)
    f.write(f"                <pre>{escape_html(json_str)}</pre>\n")
    f.write(f"            </div>\n")
    f.write(f"        </div>\n")


def _write_response_part(f, response_data: dict):
    """Write a response part with JSON formatting."""
    f.write('    <div class="part">\n')
    f.write('        <div class="part-header response-header">✅ GEMINI RESPONSE</div>\n')
    f.write(f'        <div class="part-content">\n')
    f.write(f'            <div class="json-content">\n')
    json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
    f.write(f"                <pre>{escape_html(json_str)}</pre>\n")
    f.write(f"            </div>\n")
    f.write(f"        </div>\n")
    f.write(f"    </div>\n\n")


def _write_unknown_part(f, part):
    """Write an unknown part type with fallback string representation."""
    f.write(f'        <div class="part-meta">TYPE: {type(part).__name__}</div>\n')
    f.write(f'        <div class="part-content">\n')
    f.write(f'            <div class="text-content">{escape_html(str(part))}</div>\n')
    f.write(f"        </div>\n")


def escape_html(text: str) -> str:
    """
    Escape HTML characters in text content.

    Args:
        text: Text to escape

    Returns:
        HTML-escaped text
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

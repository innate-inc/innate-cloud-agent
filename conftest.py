# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import sys
from pathlib import Path

# Add the project root to the Python path so tests can import from src
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

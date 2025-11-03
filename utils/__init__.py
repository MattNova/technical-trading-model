# Compatibility shim - redirects to new location
import sys
from pathlib import Path

# Add new location to path
new_path = Path(__file__).parent.parent.parent / 'trading_models' / 'shared' / 'utils'
if str(new_path) not in sys.path:
    sys.path.insert(0, str(new_path))

# Re-export
from plotting import create_analysis_chart

__all__ = ['create_analysis_chart']


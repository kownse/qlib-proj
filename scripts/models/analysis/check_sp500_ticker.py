import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from data.stock_pools import SP500_SYMBOLS
problem = [s for s in SP500_SYMBOLS if '.' in s]
print(f'可能有问题的 ticker: {problem}')
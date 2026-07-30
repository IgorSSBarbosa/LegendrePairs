import os
import sys

# make `import lp_rle` work when running pytest from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

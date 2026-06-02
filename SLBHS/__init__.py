from .version import __version__

# TEMP PATCH: only import modules that actually exist
try:
    from .data.loader import DataLoader
except ImportError:
    pass

from .similarity import CosineSimilarity, HandLabeler, TransitionCounter

try:
    from .clustering.theta_clusterer import ThetaClusterer
except ImportError:
    pass

try:
    from .viz.visualizer import SLBHSViz
except ImportError:
    pass

__all__ = [
    '__version__',
    'DataLoader',
    'CosineSimilarity',
    'HandLabeler',
    'TransitionCounter',
    'ThetaClusterer',
    'SLBHSViz',
]
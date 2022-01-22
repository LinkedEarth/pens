from .ens import *
from .utils import *
from .visual import *
set_style(style='web', font_scale=1.4)

# get the version
from importlib.metadata import version
__version__ = version('pens')
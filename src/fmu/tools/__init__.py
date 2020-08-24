# -*- coding: utf-8 -*-

"""Top-level package for fmu-tools"""

from fmu.tools.qcforward.qcforward import wellzonation_vs_grid  # noqa
from fmu.tools.qcproperties.qcproperties import QCProperties  # noqa

try:
    import roxar  # noqa

    ROXAR = True
except ImportError:
    ROXAR = False

if not ROXAR:

    from .rms import volumetrics  # noqa

    from .sensitivities import DesignMatrix  # noqa
    from .sensitivities import summarize_design  # noqa
    from .sensitivities import calc_tornadoinput  # noqa
    from .sensitivities import find_combinations  # noqa
    from .sensitivities import excel2dict_design  # noqa

    try:
        from .sensitivities import add_webviz_tornadoplots  # noqa
    except ImportError:
        pass  # Separate warning in _add_webviz_tornadoplots

try:
    from .version import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"

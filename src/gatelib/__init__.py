"""gatelib -- the frozen gate library.

The disposers the confirmatory referee runs, built test-first and (for the
statistics) differential-tested against a named third-party reference. Nothing
here reads the network or the cluster; every gate operates on arrays and
checksummed artifacts so it can be validated on the Mac against synthetic data.
"""
from .backbone import BackboneResult, backbone_check
from .bh import BHResult, benjamini_hochberg
from .consequence import consequence_check
from .library import GateLibraryError, library_digest, verify_gate_library
from .effect import mean_ci, paired_diff_ci
from .floor import FloorResult, floor_separation
from .g0 import G0Result, g0_detectable
from .magnitude import (
    capability_gate,
    classify_magnitude,
    law_shape_gate,
    magnitude_gate_for,
    magnitude_pvalue,
    phenomenon_gate,
)
from .verdicts import (
    EXCEEDS_MIE,
    FAIL,
    HALT_RETRY,
    INCONCLUSIVE,
    PASS,
    POWERED_NULL,
    PROCEED,
    REJECT,
)
from .mechanism import mechanism_check
from .novelty import CorpusStatus, NoveltyDecision, NoveltyResult, novelty_check, novelty_gate

__all__ = [
    "benjamini_hochberg",
    "BHResult",
    "classify_magnitude",
    "magnitude_pvalue",
    "magnitude_gate_for",
    "phenomenon_gate",
    "capability_gate",
    "law_shape_gate",
    "EXCEEDS_MIE",
    "POWERED_NULL",
    "INCONCLUSIVE",
    "PASS",
    "FAIL",
    "PROCEED",
    "REJECT",
    "HALT_RETRY",
    "g0_detectable",
    "G0Result",
    "mean_ci",
    "paired_diff_ci",
    "floor_separation",
    "FloorResult",
    "backbone_check",
    "BackboneResult",
    "mechanism_check",
    "novelty_check",
    "NoveltyResult",
    "novelty_gate",
    "NoveltyDecision",
    "CorpusStatus",
    "consequence_check",
    "library_digest",
    "verify_gate_library",
    "GateLibraryError",
]

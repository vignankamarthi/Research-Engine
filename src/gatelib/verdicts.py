"""The gate verdict vocabulary, in one place. Gates and their callers reference these constants
instead of string-matching magic literals scattered across modules, so a typo is an import error
rather than a silently-never-matching branch."""

# Magnitude classification (three-valued: an effect's interest vs the MIE).
EXCEEDS_MIE = "exceeds_mie"
POWERED_NULL = "powered_null"
INCONCLUSIVE = "inconclusive"

# The binary claim-type gates (phenomenon / capability / law-shape).
PASS = "pass"
FAIL = "fail"

# Novelty checkpoint decisions.
PROCEED = "PROCEED"
REJECT = "REJECT"
HALT_RETRY = "HALT_RETRY"

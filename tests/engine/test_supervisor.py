"""The self-chaining supervisor. It guarantees a base-case halt on ANY of three budgets
(GPU-hours, boxes, maturations), a stall backstop catches a step that stops making progress,
a durable human-clearable HALT flag stops it out of band, and a health heartbeat halts on an
integrity fault. The two sides of the liveness probe: the environment is healthy AND work is
advancing."""
from engine.health import HALT, HealthGate, Probe
from engine.supervisor import Budget, HaltFlag, SupervisorState, base_case_reached, run_supervisor


def _ok_gate():
    return HealthGate([])  # no probes -> always healthy


def _bad_gate():
    return HealthGate([Probe("integrity", lambda: False, HALT)])


def _spend_box(s):
    return SupervisorState(s.gpu_hours_spent, s.boxes_spent + 1, s.maturations)


def test_base_case_on_boxes(tmp_path):
    budget = Budget(max_gpu_hours=1e9, max_boxes=3, max_maturations=1e9)
    flag = HaltFlag(tmp_path / "halt")
    assert run_supervisor(_spend_box, budget, flag, _ok_gate()) == "BASE_CASE"


def test_base_case_on_maturations(tmp_path):
    budget = Budget(1e9, 1e9, 2)
    flag = HaltFlag(tmp_path / "halt")
    step = lambda s: SupervisorState(s.gpu_hours_spent, s.boxes_spent, s.maturations + 1)
    assert run_supervisor(step, budget, flag, _ok_gate()) == "BASE_CASE"


def test_base_case_on_gpu_hours(tmp_path):
    budget = Budget(max_gpu_hours=10.0, max_boxes=1e9, max_maturations=1e9)
    flag = HaltFlag(tmp_path / "halt")
    step = lambda s: SupervisorState(s.gpu_hours_spent + 4.0, s.boxes_spent, s.maturations)
    assert run_supervisor(step, budget, flag, _ok_gate()) == "BASE_CASE"


def test_halt_flag_stops_even_with_infinite_budget(tmp_path):
    budget = Budget(1e9, 1e9, 1e9)
    flag = HaltFlag(tmp_path / "halt")
    flag.set("manual stop")
    assert run_supervisor(_spend_box, budget, flag, _ok_gate()) == "HALTED"


def test_preflight_health_halt_never_runs_a_step(tmp_path):
    budget = Budget(1e9, 3, 1e9)
    flag = HaltFlag(tmp_path / "halt")
    called = {"n": 0}

    def step(s):
        called["n"] += 1
        return _spend_box(s)

    assert run_supervisor(step, budget, flag, _bad_gate()) == "HEALTH_HALT"
    assert called["n"] == 0


def test_stall_backstop_halts_a_non_advancing_step(tmp_path):
    budget = Budget(1e9, 5, 1e9)  # boxes never reached because step never advances
    flag = HaltFlag(tmp_path / "halt")
    assert run_supervisor(lambda s: s, budget, flag, _ok_gate()) == "BACKSTOP"
    assert flag.is_set()  # a stall is a bug -> it pages via the flag


def test_halt_flag_is_durable_and_human_clearable(tmp_path):
    p = tmp_path / "halt"
    HaltFlag(p).set("boom")
    assert HaltFlag(p).is_set()      # persists across reopen
    assert HaltFlag(p).reason() == "boom"
    HaltFlag(p).clear()              # the human clears it
    assert not HaltFlag(p).is_set()


def test_default_budget_uses_fifteen_maturations():
    from engine.supervisor import DEFAULT_MAX_MATURATIONS
    assert DEFAULT_MAX_MATURATIONS == 15
    assert Budget.default().max_maturations == 15


def test_base_case_predicate():
    assert base_case_reached(SupervisorState(0, 3, 0), Budget(1e9, 3, 1e9))
    assert not base_case_reached(SupervisorState(0, 2, 0), Budget(1e9, 3, 1e9))

"""
Tests for lmc.scheduler: the hours-window logic and the enable/disable/set-hours
toggle's persistence. Does not exercise run_batch() itself -- that pulls in
network calls and audio/ML models, well outside a unit test's scope.
"""

from lmc.scheduler import SchedulerState, disable, enable, in_window, set_hours, status


def test_in_window_simple_range():
    assert in_window(3, 1, 6)
    assert not in_window(0, 1, 6)
    assert not in_window(6, 1, 6)


def test_in_window_wraps_past_midnight():
    # 22:00-06:00 covers 22,23,0..5
    assert in_window(23, 22, 6)
    assert in_window(0, 22, 6)
    assert in_window(5, 22, 6)
    assert not in_window(6, 22, 6)
    assert not in_window(21, 22, 6)


def test_in_window_equal_bounds_means_always_on():
    for h in range(24):
        assert in_window(h, 5, 5)


def test_state_defaults_when_no_file(tmp_path):
    path = tmp_path / "scheduler_state.json"
    s = SchedulerState.load(path)
    assert s.enabled is False
    assert (s.start_hour, s.end_hour) == (1, 6)


def test_enable_disable_round_trip(tmp_path):
    path = tmp_path / "scheduler_state.json"
    enable(path)
    assert SchedulerState.load(path).enabled is True
    disable(path)
    assert SchedulerState.load(path).enabled is False


def test_set_hours_persists(tmp_path):
    path = tmp_path / "scheduler_state.json"
    set_hours(9, 17, path)
    s = SchedulerState.load(path)
    assert (s.start_hour, s.end_hour) == (9, 17)


def test_set_hours_rejects_out_of_range(tmp_path):
    path = tmp_path / "scheduler_state.json"
    try:
        set_hours(0, 24, path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_status_reflects_state(tmp_path):
    path = tmp_path / "scheduler_state.json"
    enable(path)
    set_hours(0, 0, path)  # equal bounds -- always-on window
    st = status(path)
    assert st["enabled"] is True
    assert st["would_run_now"] is True


def test_status_off_by_default(tmp_path):
    path = tmp_path / "scheduler_state.json"
    st = status(path)
    assert st["enabled"] is False
    assert st["would_run_now"] is False

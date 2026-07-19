"""CPU-only regressions for G1 training-data and evaluation identities."""

import ast
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_postprocessor():
    path = REPO_ROOT / "scripts" / "postprocess_psi0.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = {"STATE_SLICES", "ACTION_SLICES"}
    functions = {
        "history_cmd_to_torso_rpy",
        "build_vectors",
        "build_proprio_obs",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)

    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    assert constants | functions <= namespace.keys()
    return SimpleNamespace(**{name: namespace[name] for name in constants | functions})


def _load_class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="_TestAgent",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace = {"warnings": warnings}
    exec(compile(module, str(path), "exec"), namespace)
    return getattr(namespace["_TestAgent"], method_name)


def _load_functions(path: Path, *function_names: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {function.name for function in functions} == set(function_names)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"Any": Any, "Callable": Callable}
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in function_names)


def _load_eval_shutdown_helpers(eval_filename="eval.py"):
    eval_path = REPO_ROOT / "src" / "simple" / "cli" / eval_filename
    helpers = _load_functions(
        eval_path,
        "_success_rate",
        "_append_success_rate_footer_if_owner",
        "_finalize_eval_worker",
    )
    namespace = helpers[0].__globals__
    for helper in helpers:
        namespace[helper.__name__] = helper
    return helpers, namespace


def test_torso_mapping_reverses_axes_without_reversing_time():
    postprocess = _load_postprocessor()
    history_cmd = np.zeros((3, 9), dtype=np.float64)
    history_cmd[:, 3:6] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    expected = np.array(
        [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0], [9.0, 8.0, 7.0]],
        dtype=np.float32,
    )

    mapped = postprocess.history_cmd_to_torso_rpy(history_cmd, 3)
    np.testing.assert_array_equal(mapped, expected)
    assert mapped.dtype == np.float32
    assert mapped.flags.c_contiguous

    proprio = np.zeros((3, 43), dtype=np.float32)
    states, _ = postprocess.build_vectors(
        proprio,
        np.zeros((3, 9), dtype=np.float32),
        history_cmd,
        np.zeros((3, 43), dtype=np.float32),
        target_yaw=np.zeros(3, dtype=np.float32),
        turning_flag=np.zeros(3, dtype=np.float32),
    )
    *_, observation_torso_rpy, _ = postprocess.build_proprio_obs(proprio, history_cmd)
    np.testing.assert_array_equal(states[:, 28:31], expected)
    np.testing.assert_array_equal(observation_torso_rpy, expected)


@pytest.mark.parametrize(
    ("history_cmd", "n_rows", "message"),
    [
        (np.zeros(5), 1, "shape"),
        (np.zeros((2, 5)), 1, "shape"),
        (np.zeros((2, 6)), -1, "requested"),
        (np.zeros((2, 6)), 3, "requested"),
    ],
)
def test_torso_mapping_rejects_invalid_shapes_and_lengths(history_cmd, n_rows, message):
    postprocess = _load_postprocessor()
    with pytest.raises(ValueError, match=message):
        postprocess.history_cmd_to_torso_rpy(history_cmd, n_rows)


def test_dp_g1_episode_index_has_a_nonnegative_stable_fallback():
    method = _load_class_method(
        REPO_ROOT / "src" / "simple" / "baselines" / "dp_g1.py",
        "DpG1Agent",
        "_request_episode_index",
    )
    agent = type(
        "Agent",
        (),
        {"_session_idx": 7, "_warned_episode_index_fallback": False},
    )()

    assert method(agent, None) == 7
    assert method(agent, {}) == 7
    assert method(agent, {"episode_index": -1}) == 7
    assert method(agent, {"episode_index": 0}) == 0
    assert method(agent, {"episode_index": 13}) == 13


def test_dp_g1_warns_once_when_using_the_noncanonical_fallback():
    method = _load_class_method(
        REPO_ROOT / "src" / "simple" / "baselines" / "dp_g1.py",
        "DpG1Agent",
        "_warn_if_noncanonical_episode_index",
    )
    agent = type("Agent", (), {"_warned_episode_index_fallback": False})()

    with pytest.warns(RuntimeWarning, match="unsafe across workers or shards"):
        method(agent, None)
    assert agent._warned_episode_index_fallback is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        method(agent, {})
        method(agent, {"episode_index": 3})


def test_evaluator_overrides_stale_info_with_global_episode_index():
    eval_path = REPO_ROOT / "src" / "simple" / "cli" / "eval.py"
    (with_episode_index,) = _load_functions(eval_path, "_with_episode_index")
    original = {"episode_index": 999, "reward_terms": {"lift": 1.0}}

    action_info = with_episode_index(original, np.int64(23))

    assert action_info == {
        "episode_index": 23,
        "reward_terms": {"lift": 1.0},
    }
    assert original["episode_index"] == 999
    assert with_episode_index(None, 4) == {"episode_index": 4}


def test_first_and_later_policy_requests_receive_the_same_global_episode_id():
    eval_path = REPO_ROOT / "src" / "simple" / "cli" / "eval.py"
    _, get_action = _load_functions(
        eval_path,
        "_with_episode_index",
        "_get_action_with_episode_index",
    )

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def get_action(self, observation, *, info, instruction):
            self.calls.append((observation, info, instruction))
            return len(self.calls)

    agent = FakeAgent()
    reset_info = {"phase": "reset"}
    later_info = {"phase": "step", "episode_index": -1}
    assert get_action(agent, "obs-0", reset_info, "pick", 23) == 1
    assert get_action(agent, "obs-1", later_info, "pick", 23) == 2

    assert [call[1]["episode_index"] for call in agent.calls] == [23, 23]
    assert [call[1]["phase"] for call in agent.calls] == ["reset", "step"]
    assert "episode_index" not in reset_info
    assert later_info["episode_index"] == -1


def test_worker_shards_preserve_global_episode_ids():
    eval_path = REPO_ROOT / "src" / "simple" / "cli" / "eval.py"
    (indices_for_worker,) = _load_functions(eval_path, "_episode_indices_for_worker")

    args = (100, 20, 5)
    assert indices_for_worker(*args, worker_id=0, num_workers=2) == [20, 22, 24]
    assert indices_for_worker(*args, worker_id=1, num_workers=2) == [21, 23]


@pytest.mark.parametrize("eval_filename", ["eval.py", "eval_decoupled_wbc.py"])
def test_single_worker_writes_footer_before_nonreturning_simulator_close(
    eval_filename,
):
    (_, _, finalize_worker), namespace = _load_eval_shutdown_helpers(eval_filename)
    events = []

    def append_line(eval_dir, line):
        events.append(("footer", eval_dir, line))

    namespace["_append_eval_stats_line"] = append_line

    class SimulationShutdown(BaseException):
        pass

    class NonReturningEnv:
        def close(self):
            events.append(("close",))
            raise SimulationShutdown

    def persist(kind, payload):
        events.append(("persist", kind, payload))

    def report(event, **payload):
        events.append(("report", event, payload))

    with pytest.raises(SimulationShutdown):
        finalize_worker(
            NonReturningEnv(),
            {"episode_0": True, "episode_1": False},
            "/eval",
            num_workers=1,
            persist_payload=persist,
            report=report,
        )

    assert [event[0] for event in events] == ["persist", "footer", "report", "close"]
    assert events[1] == ("footer", "/eval", "success rate: 0.50 \n")


@pytest.mark.parametrize("eval_filename", ["eval.py", "eval_decoupled_wbc.py"])
def test_returning_single_worker_close_does_not_duplicate_footer_in_parent(
    eval_filename,
):
    (_, append_footer, finalize_worker), namespace = _load_eval_shutdown_helpers(
        eval_filename
    )
    footers = []
    namespace["_append_eval_stats_line"] = lambda eval_dir, line: footers.append(
        (eval_dir, line)
    )

    class ReturningEnv:
        def close(self):
            return None

    stats = {"episode_0": True, "episode_1": True}
    finalize_worker(
        ReturningEnv(),
        stats,
        "/eval",
        num_workers=1,
        persist_payload=lambda kind, payload: None,
        report=lambda event, **payload: None,
    )
    appended_by_parent = append_footer(
        "/eval",
        stats,
        1,
        owner="parent",
    )

    assert appended_by_parent is False
    assert footers == [("/eval", "success rate: 1.00 \n")]


@pytest.mark.parametrize("eval_filename", ["eval.py", "eval_decoupled_wbc.py"])
def test_multiworker_footer_is_written_once_from_aggregate_parent_stats(eval_filename):
    (_, append_footer, finalize_worker), namespace = _load_eval_shutdown_helpers(
        eval_filename
    )
    footers = []
    namespace["_append_eval_stats_line"] = lambda eval_dir, line: footers.append(
        (eval_dir, line)
    )

    class ReturningEnv:
        def close(self):
            return None

    worker_stats = [
        {"episode_0": True, "episode_2": False},
        {"episode_1": True, "episode_3": True},
    ]
    for stats in worker_stats:
        finalize_worker(
            ReturningEnv(),
            stats,
            "/eval",
            num_workers=2,
            persist_payload=lambda kind, payload: None,
            report=lambda event, **payload: None,
        )

    aggregate = {key: value for stats in worker_stats for key, value in stats.items()}
    assert append_footer("/eval", aggregate, 2, owner="parent") is True
    assert footers == [("/eval", "success rate: 0.75 \n")]

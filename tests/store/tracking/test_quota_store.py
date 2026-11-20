import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlflow.entities import ExperimentTag, Metric, Param, RunTag, ViewType
from mlflow.environment_variables import (
    MLFLOW_QUOTA_ASYNC_WRITES,
    MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT,
    MLFLOW_QUOTA_TAG,
)
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.quota import (
    QUOTA_TAG_MAX_RUNS,
    QuotaExceededException,
    QuotaTrackingStore,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def quota_store(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_QUOTA_ASYNC_WRITES.name, raising=False)
    monkeypatch.delenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, raising=False)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = QuotaTrackingStore(f"quota:sqlite:///{tmp_path / 'backend.db'}", str(artifact_dir))
    try:
        yield store
    finally:
        store.close()


def _create_runs(store, experiment_id, count):
    return [store.create_run(experiment_id, "user", 0, [], f"run-{idx}") for idx in range(count)]


def test_quota_store_wraps_sqlalchemy_backend(quota_store):
    assert isinstance(quota_store, QuotaTrackingStore)
    assert isinstance(quota_store.backend_store, SqlAlchemyStore)


def test_quota_store_registered_as_scheme(tmp_path, monkeypatch):
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    assert "quota" in _tracking_store_registry._registry

    monkeypatch.delenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, raising=False)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = _tracking_store_registry.get_store(
        f"quota:sqlite:///{tmp_path / 'registry.db'}", str(artifact_dir)
    )
    try:
        assert isinstance(store, QuotaTrackingStore)
    finally:
        store.close()


def test_invalid_quota_uri_rejected():
    with pytest.raises(MlflowException, match="must start with 'quota:'"):
        QuotaTrackingStore("sqlite:////tmp/no-prefix.db", None)


def test_global_quota_blocks_create_run(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "2")
    experiment_id = quota_store.create_experiment("limited", None, [])
    _create_runs(quota_store, experiment_id, 2)

    with pytest.raises(QuotaExceededException, match="reached its quota") as exc:
        quota_store.create_run(experiment_id, "user", 0, [], "overflow")
    assert exc.value.error_code == "QUOTA_EXCEEDED"


def test_quota_counts_only_active_runs(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "1")
    experiment_id = quota_store.create_experiment("recycle", None, [])
    run = _create_runs(quota_store, experiment_id, 1)[0]

    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(experiment_id, "user", 0, [], "blocked")

    quota_store.delete_run(run.info.run_id)
    replacement = quota_store.create_run(experiment_id, "user", 0, [], "replacement")
    assert replacement.info.experiment_id == experiment_id


def test_experiment_tag_overrides_global_quota(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "10")
    experiment_id = quota_store.create_experiment(
        "tagged", None, [ExperimentTag(QUOTA_TAG_MAX_RUNS, "1")]
    )
    _create_runs(quota_store, experiment_id, 1)
    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(experiment_id, "user", 0, [], "overflow")


def test_custom_quota_tag_key(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_TAG.name, "team.quota")
    experiment_id = quota_store.create_experiment(
        "custom-tag", None, [ExperimentTag("team.quota", "1")]
    )
    _create_runs(quota_store, experiment_id, 1)
    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(experiment_id, "user", 0, [], "overflow")


def test_invalid_quota_tag_falls_back_to_global_limit(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "1")
    experiment_id = quota_store.create_experiment(
        "bad-tag", None, [ExperimentTag(QUOTA_TAG_MAX_RUNS, "not-a-number")]
    )
    _create_runs(quota_store, experiment_id, 1)
    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(experiment_id, "user", 0, [], "overflow")


def test_zero_quota_means_unlimited(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "0")
    experiment_id = quota_store.create_experiment("unlimited", None, [])
    runs = _create_runs(quota_store, experiment_id, 5)
    assert len(runs) == 5


def test_quota_is_scoped_per_experiment(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "1")
    exp_a = quota_store.create_experiment("a", None, [])
    exp_b = quota_store.create_experiment("b", None, [])
    _create_runs(quota_store, exp_a, 1)
    _create_runs(quota_store, exp_b, 1)
    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(exp_a, "user", 0, [], "a-overflow")
    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.create_run(exp_b, "user", 0, [], "b-overflow")


def test_restore_run_enforces_quota(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "1")
    experiment_id = quota_store.create_experiment("restore", None, [])
    deleted = _create_runs(quota_store, experiment_id, 1)[0]
    quota_store.delete_run(deleted.info.run_id)
    _create_runs(quota_store, experiment_id, 1)

    with pytest.raises(QuotaExceededException, match="reached its quota"):
        quota_store.restore_run(deleted.info.run_id)


def test_concurrent_create_runs_respect_quota(quota_store, monkeypatch):
    limit = 5
    workers = 20
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, str(limit))
    experiment_id = quota_store.create_experiment("concurrent", None, [])
    barrier = threading.Barrier(workers)
    errors: list[Exception] = []
    errors_lock = threading.Lock()

    def create_one(index):
        barrier.wait()
        try:
            quota_store.create_run(experiment_id, "user", 0, [], f"c-{index}")
        except QuotaExceededException as exc:
            with errors_lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(create_one, range(workers)))

    runs, token = quota_store._search_runs(
        [experiment_id], "", ViewType.ACTIVE_ONLY, workers * 2, None, None
    )
    assert not token
    assert len(runs) == limit
    assert len(errors) == workers - limit
    assert all(exc.error_code == "QUOTA_EXCEEDED" for exc in errors)


def test_concurrent_writes_to_different_experiments_are_independent(quota_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.name, "1")
    exp_a = quota_store.create_experiment("a", None, [])
    exp_b = quota_store.create_experiment("b", None, [])

    def create(experiment_id):
        quota_store.create_run(experiment_id, "user", 0, [], "run")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create, exp_a), executor.submit(create, exp_b)]
        for future in futures:
            future.result()

    for experiment_id in (exp_a, exp_b):
        runs, _ = quota_store._search_runs(
            [experiment_id], "", ViewType.ACTIVE_ONLY, 10, None, None
        )
        assert len(runs) == 1


def test_async_writes_buffer_and_flush(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_ASYNC_WRITES.name, "true")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = QuotaTrackingStore(f"quota:sqlite:///{tmp_path / 'async.db'}", str(artifact_dir))
    try:
        assert store.async_writes_enabled
        experiment_id = store.create_experiment("async", None, [])
        run = store.create_run(experiment_id, "user", 0, [], "run")
        run_id = run.info.run_id

        for index in range(50):
            store.log_metric(run_id, Metric(f"metric-{index}", float(index), 0, index))
        store.log_param(run_id, Param("param", "value"))
        store.set_tag(run_id, RunTag("tag", "value"))
        store.log_batch(run_id, [Metric("batch-metric", 1.0, 0, 0)], [], [])

        store.flush()
        persisted = store.get_run(run_id)
        assert len(persisted.data.metrics) == 51
        assert persisted.data.params["param"] == "value"
        assert persisted.data.tags["tag"] == "value"
    finally:
        store.close()


def test_async_write_failures_surface_on_flush(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_ASYNC_WRITES.name, "true")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = QuotaTrackingStore(f"quota:sqlite:///{tmp_path / 'fail.db'}", str(artifact_dir))

    def boom():
        raise RuntimeError("background failure")

    store._write_queue.submit(boom)
    with pytest.raises(RuntimeError, match="background failure"):
        store.flush()
    store.close()


def test_synchronous_mode_writes_immediately(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_QUOTA_ASYNC_WRITES.name, raising=False)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = QuotaTrackingStore(f"quota:sqlite:///{tmp_path / 'sync.db'}", str(artifact_dir))
    try:
        assert not store.async_writes_enabled
        experiment_id = store.create_experiment("sync", None, [])
        run = store.create_run(experiment_id, "user", 0, [], "run")
        store.log_metric(run.info.run_id, Metric("immediate", 1.0, 0, 0))
        assert "immediate" in store.get_run(run.info.run_id).data.metrics
    finally:
        store.close()


def test_concurrent_async_writes_are_all_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_QUOTA_ASYNC_WRITES.name, "true")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = QuotaTrackingStore(f"quota:sqlite:///{tmp_path / 'casync.db'}", str(artifact_dir))
    workers = 8
    writes_per_worker = 25
    try:
        experiment_id = store.create_experiment("casync", None, [])
        run = store.create_run(experiment_id, "user", 0, [], "run")
        run_id = run.info.run_id

        def write_batch(worker_id):
            for index in range(writes_per_worker):
                step = worker_id * writes_per_worker + index
                store.log_metric(run_id, Metric(f"m-{worker_id}", float(step), 0, step))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(write_batch, range(workers)))
        store.flush()

        for worker_id in range(workers):
            history = store.get_metric_history(run_id, f"m-{worker_id}")
            assert len(history) == writes_per_worker
    finally:
        store.close()


def test_read_apis_are_delegated(quota_store):
    experiment_id = quota_store.create_experiment("reads", None, [])
    run = quota_store.create_run(experiment_id, "user", 0, [], "run")
    quota_store.log_metric(run.info.run_id, Metric("m", 3.0, 0, 0))

    fetched = quota_store.get_run(run.info.run_id)
    assert fetched.info.run_id == run.info.run_id
    assert fetched.data.metrics["m"] == 3.0

    experiments = quota_store.search_experiments(view_type=ViewType.ALL)
    assert any(exp.experiment_id == experiment_id for exp in experiments)


def test_quota_store_supports_abstract_store_interface(quota_store):
    from mlflow.store.tracking.abstract_store import AbstractStore

    assert isinstance(quota_store, AbstractStore)
    # Attributes that only exist on the delegated backend are exposed via __getattr__.
    assert quota_store.engine is quota_store.backend_store.engine

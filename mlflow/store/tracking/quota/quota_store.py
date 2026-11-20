"""Tracking store plugin adding per-experiment quotas and asynchronous writes.

The store is selected with a ``quota:`` tracking URI, e.g.
``quota:sqlite:////path/to/backend.db``. The URI suffix after ``quota:`` is a
fully valid tracking URI that is used to build the delegated backend store via
the standard tracking store registry, so the plugin works on top of any
existing backend without changing its behavior.

Two capabilities are layered on top of the delegated store:

* Per-experiment active-run quotas enforced in ``create_run`` (and
  ``restore_run``), configurable globally through the
  ``MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT`` environment variable and per
  experiment through a configurable experiment tag.
* Optional asynchronous dispatch of fire-and-forget writes (metrics, params,
  tags, ...) through a bounded background queue when
  ``MLFLOW_QUOTA_ASYNC_WRITES`` is enabled. Reads and run lifecycle operations
  stay synchronous so that the existing MLflow API contract (including return
  values) is preserved.
"""

import logging
from typing import Any, Callable

from mlflow.entities import ViewType
from mlflow.environment_variables import MLFLOW_QUOTA_ASYNC_WRITES
from mlflow.exceptions import MlflowException
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.quota.async_write_queue import AsyncWriteQueue
from mlflow.store.tracking.quota.policy import QuotaPolicy

_logger = logging.getLogger(__name__)

QUOTA_URI_PREFIX = "quota:"

# Writes without return values that are safe to offload to the background
# worker. Everything else (run creation, reads, ...) stays synchronous.
_ASYNC_WRITE_METHODS = frozenset({
    "log_metric",
    "log_param",
    "set_tag",
    "log_batch",
    "set_experiment_tag",
    "delete_experiment_tag",
    "log_inputs",
    "log_outputs",
    "record_logged_model",
    "set_trace_tag",
    "delete_trace_tag",
    "start_trace",
    "log_spans",
    "create_assessment",
    "update_assessment",
    "delete_assessment",
    "link_traces_to_run",
    "unlink_traces_from_run",
    "delete_traces",
    "archive_traces",
})


def _strip_quota_scheme(store_uri: str) -> str:
    if not store_uri.startswith(QUOTA_URI_PREFIX):
        raise MlflowException(
            f"Quota store URI must start with '{QUOTA_URI_PREFIX}', got: {store_uri}"
        )
    return store_uri[len(QUOTA_URI_PREFIX) :]


def _build_backend_store(store_uri: str, artifact_uri: str | None) -> AbstractStore:
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    # Resolve the builder directly instead of going through
    # ``TrackingStoreRegistry.get_store``: the quota store itself is being built
    # while that method holds its (non-reentrant) construction lock, so nested
    # ``get_store`` calls would deadlock.
    builder = _tracking_store_registry.get_store_builder(store_uri)
    store = builder(store_uri=store_uri, artifact_uri=artifact_uri)
    if not isinstance(store, AbstractStore):
        raise MlflowException(
            f"Backend tracking store for URI '{store_uri}' does not implement AbstractStore: "
            f"{type(store).__name__}"
        )
    return store


class QuotaTrackingStore(AbstractStore):
    """Decorator-style tracking store enforcing quotas and optional async writes."""

    def __init__(self, store_uri: str, artifact_uri: str | None = None):
        super().__init__()
        backend_uri = _strip_quota_scheme(store_uri)
        self._backend_store = _build_backend_store(backend_uri, artifact_uri)
        self._quota_policy = QuotaPolicy(self)
        self._async_writes_enabled = MLFLOW_QUOTA_ASYNC_WRITES.get()
        self._write_queue: AsyncWriteQueue | None = (
            AsyncWriteQueue() if self._async_writes_enabled else None
        )

    @property
    def backend_store(self) -> AbstractStore:
        return self._backend_store

    @property
    def quota_policy(self) -> QuotaPolicy:
        return self._quota_policy

    @property
    def async_writes_enabled(self) -> bool:
        enabled: bool = self._async_writes_enabled
        return enabled

    @property
    def supports_workspaces(self) -> bool:
        supported: bool = self._backend_store.supports_workspaces
        return supported

    @property
    def supports_trace_archival(self) -> bool:
        supported: bool = self._backend_store.supports_trace_archival
        return supported

    def _delegate(self, method_name: str) -> Callable[..., Any]:
        method: Callable[..., Any] = getattr(self._backend_store, method_name)
        return method

    def _dispatch_write(self, method_name: str, *args, **kwargs):
        # Resolve the method straight from the backend store. Going through
        # ``self`` would re-enter ``__getattr__`` for dynamically wrapped writes.
        method = getattr(self._backend_store, method_name)
        if self._write_queue is None:
            return method(*args, **kwargs)
        self._write_queue.submit(method, *args, **kwargs)
        return None

    def flush(self) -> None:
        """Wait for all queued background writes (and async log_batch) to finish."""
        if self._write_queue is not None:
            self._write_queue.flush()
        self.flush_async_logging()

    def close(self) -> None:
        if self._write_queue is not None:
            self._write_queue.close()
        self.shut_down_async_logging()

    # -- Run lifecycle: quota enforcement happens here ----------------------

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        policy = self._quota_policy
        with policy.get_lock(experiment_id):
            policy.enforce(experiment_id)
            return self._delegate("create_run")(experiment_id, user_id, start_time, tags, run_name)

    def restore_run(self, run_id):
        run = self.get_run(run_id)
        experiment_id = run.info.experiment_id
        policy = self._quota_policy
        with policy.get_lock(experiment_id):
            policy.enforce(experiment_id)
            return self._delegate("restore_run")(run_id)

    def update_run_info(self, run_id, run_status, end_time, run_name):
        # Make sure queued writes for this run are durable before it terminates.
        if self._write_queue is not None:
            self._write_queue.flush()
        return self._delegate("update_run_info")(run_id, run_status, end_time, run_name)

    def delete_run(self, run_id):
        return self._delegate("delete_run")(run_id)

    # -- Writes eligible for asynchronous dispatch --------------------------

    def log_metric(self, run_id, metric):
        return self._dispatch_write("log_metric", run_id, metric)

    def log_param(self, run_id, param):
        return self._dispatch_write("log_param", run_id, param)

    def set_tag(self, run_id, tag):
        return self._dispatch_write("set_tag", run_id, tag)

    def log_batch(self, run_id, metrics, params, tags):
        return self._dispatch_write("log_batch", run_id, metrics, params, tags)

    def set_experiment_tag(self, experiment_id, tag):
        return self._dispatch_write("set_experiment_tag", experiment_id, tag)

    def delete_experiment_tag(self, experiment_id, key):
        return self._dispatch_write("delete_experiment_tag", experiment_id, key)

    def log_inputs(self, run_id, datasets=None, models=None):
        return self._dispatch_write("log_inputs", run_id, datasets, models)

    def link_traces_to_run(self, trace_ids, run_id):
        return self._dispatch_write("link_traces_to_run", trace_ids, run_id)

    # -- Remaining abstract methods: straight delegation --------------------

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        return self._delegate("search_experiments")(
            view_type=view_type,
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by,
            page_token=page_token,
        )

    def create_experiment(self, name, artifact_location, tags):
        return self._delegate("create_experiment")(name, artifact_location, tags)

    def get_experiment(self, experiment_id):
        return self._delegate("get_experiment")(experiment_id)

    def delete_experiment(self, experiment_id):
        return self._delegate("delete_experiment")(experiment_id)

    def restore_experiment(self, experiment_id):
        return self._delegate("restore_experiment")(experiment_id)

    def rename_experiment(self, experiment_id, new_name):
        return self._delegate("rename_experiment")(experiment_id, new_name)

    def get_run(self, run_id):
        return self._delegate("get_run")(run_id)

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        return self._delegate("get_metric_history")(
            run_id, metric_key, max_results=max_results, page_token=page_token
        )

    def _search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        return self._delegate("_search_runs")(
            experiment_ids,
            filter_string,
            run_view_type,
            max_results,
            order_by,
            page_token,
        )

    # -- Pass-through for the rest of the AbstractStore surface -------------

    def __getattr__(self, item):
        # __getattr__ is only called when normal lookup fails, so there is no
        # recursion risk for attributes assigned in __init__.
        backend = self.__dict__.get("_backend_store")
        if backend is None:
            raise AttributeError(item)
        attr = getattr(backend, item)
        if item in _ASYNC_WRITE_METHODS and self.__dict__.get("_write_queue") is not None:
            return lambda *args, **kwargs: self._dispatch_write(item, *args, **kwargs)
        return attr

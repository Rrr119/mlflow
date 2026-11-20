"""Per-experiment quota enforcement for the quota tracking store plugin.

The policy is deliberately backend agnostic: it counts active runs through the
delegated store's public ``search_runs`` API so that it works on top of any
tracking backend (SQLAlchemy, file, REST, ...). Per-experiment locks make the
check-and-create sequence safe under concurrent writers sharing a store
instance, and the count is capped at the configured limit to avoid paging
through an experiment's entire run history.
"""

import threading

from mlflow.entities import ViewType
from mlflow.environment_variables import (
    MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT,
    MLFLOW_QUOTA_TAG,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import QUOTA_EXCEEDED
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

QUOTA_TAG_MAX_RUNS = "mlflow.quota.maxRuns"


class QuotaExceededException(MlflowException):
    """Raised when creating a run would violate an experiment's run quota."""

    def __init__(self, experiment_id: str, limit: int, current: int):
        super().__init__(
            f"Experiment '{experiment_id}' has reached its quota of {limit} active run(s) "
            f"(current: {current}). Increase the quota via the '{MLFLOW_QUOTA_TAG.get()}' "
            "experiment tag or the MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT environment variable, "
            "or terminate/delete an existing run.",
            error_code=QUOTA_EXCEEDED,
        )


class QuotaPolicy:
    """Resolves and enforces per-experiment active-run quotas."""

    def __init__(self, store, default_limit: int | None = None):
        self._store = store
        # ``None`` means the limit is resolved from the environment on every
        # check, so tests and deployments can tune the env var at runtime.
        self._default_limit_override = default_limit
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def get_lock(self, experiment_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(experiment_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[experiment_id] = lock
            return lock

    def get_limit(self, experiment_id: str) -> int | None:
        """Return the active-run limit for an experiment, or None when unlimited."""
        tag_limit: int | None = self._get_tag_limit(experiment_id)
        if tag_limit is not None:
            return tag_limit
        if self._default_limit_override is not None:
            return self._default_limit_override
        limit: int = MLFLOW_QUOTA_MAX_RUNS_PER_EXPERIMENT.get()
        return limit

    def _get_tag_limit(self, experiment_id: str) -> int | None:
        tag_key = MLFLOW_QUOTA_TAG.get()
        if not tag_key:
            return None
        try:
            experiment = self._store.get_experiment(experiment_id)
        except MlflowException:
            return None
        if experiment is None:
            return None
        tags: dict[str, str] = experiment.tags or {}
        raw_value = tags.get(tag_key)
        if raw_value is None:
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def count_active_runs(self, experiment_id: str, limit: int) -> int:
        """Count active runs, stopping as soon as ``limit + 1`` runs are seen."""
        page_token = None
        total = 0
        while True:
            max_results = min(SEARCH_MAX_RESULTS_DEFAULT, limit + 1 - total)
            if max_results <= 0:
                return total
            runs, page_token = self._store._search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results,
                order_by=None,
                page_token=page_token,
            )
            total += len(runs)
            if total > limit or not page_token:
                return total

    def enforce(self, experiment_id: str) -> None:
        """Raise :class:`QuotaExceededException` if the experiment is at capacity."""
        limit = self.get_limit(experiment_id)
        if limit is None or limit <= 0:
            return
        current = self.count_active_runs(experiment_id, limit)
        if current >= limit:
            raise QuotaExceededException(experiment_id, limit, current)

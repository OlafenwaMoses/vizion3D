"""Small in-memory job store for long-running server requests."""

from __future__ import annotations

import os
import pickle
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class JobResultUnavailable(RuntimeError):
    """Raised when a completed job result can no longer be read."""


@dataclass
class JobRecord(Generic[T]):
    job_id: str
    kind: str
    status: JobStatus
    created_at: datetime
    expires_at: datetime
    max_result_reads: int
    result_reads: int = 0
    result_path: Path | None = None
    result: T | None = None
    error: str = ""
    future: Future | None = None

    @property
    def result_reads_remaining(self) -> int:
        return max(0, self.max_result_reads - self.result_reads)


class InMemoryJobStore:
    """Process-local background job executor with result read limits."""

    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(hours=24),
        max_result_reads: int = 10,
        max_workers: int = 2,
        storage_dir: str | Path | None = None,
    ) -> None:
        self.ttl = ttl
        self.max_result_reads = max_result_reads
        root = (
            storage_dir
            or os.environ.get("VIZION3D_JOB_DIR")
            or tempfile.mkdtemp(prefix="vizion3d-reconstruction-jobs-")
        )
        self.storage_dir = Path(root)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def submit(self, kind: str, fn: Callable[[], T]) -> JobRecord[T]:
        now = datetime.now(UTC)
        job = JobRecord[T](
            job_id=uuid.uuid4().hex,
            kind=kind,
            status=JobStatus.QUEUED,
            created_at=now,
            expires_at=now + self.ttl,
            max_result_reads=self.max_result_reads,
        )
        with self._lock:
            self._jobs[job.job_id] = job

        def _run() -> None:
            with self._lock:
                if not self._is_expired(job):
                    job.status = JobStatus.RUNNING
            result_path = None
            try:
                result = fn()
                result_path = self._result_path(job.job_id)
                with result_path.open("wb") as handle:
                    pickle.dump(result, handle)
            except Exception as exc:  # pragma: no cover - exercised through callers
                if result_path is not None:
                    result_path.unlink(missing_ok=True)
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                return
            with self._lock:
                if self._is_expired(job):
                    result_path.unlink(missing_ok=True)
                    job.status = JobStatus.EXPIRED
                    self._delete_result_file(job)
                else:
                    job.status = JobStatus.SUCCEEDED
                    job.result_path = result_path

        job.future = self._executor.submit(_run)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and self._is_expired(job):
                job.status = JobStatus.EXPIRED
                self._delete_result_file(job)
            return job

    def consume_result(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if self._is_expired(job):
                job.status = JobStatus.EXPIRED
                self._delete_result_file(job)
                raise JobResultUnavailable("Job result expired")
            if job.status != JobStatus.SUCCEEDED:
                return job
            if job.result_reads >= job.max_result_reads:
                self._delete_result_file(job)
                raise JobResultUnavailable("Job result read limit exceeded")
            if job.result_path is None or not job.result_path.is_file():
                raise JobResultUnavailable("Job result file is unavailable")
            job.result_reads += 1
            with job.result_path.open("rb") as handle:
                job.result = pickle.load(handle)
            if job.result_reads >= job.max_result_reads:
                self._delete_result_file(job)
            return job

    @staticmethod
    def _is_expired(job: JobRecord) -> bool:
        return datetime.now(UTC) >= job.expires_at

    def _result_path(self, job_id: str) -> Path:
        return self.storage_dir / f"{job_id}.pkl"

    @staticmethod
    def _delete_result_file(job: JobRecord) -> None:
        if job.result_path is None:
            return
        try:
            job.result_path.unlink(missing_ok=True)
        finally:
            job.result_path = None
            job.result = None


reconstruction_jobs = InMemoryJobStore()

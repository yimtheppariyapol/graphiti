"""Tests for QueueService worker lifetime and bookkeeping.

Why these exist (2026-07-16): `_queue_workers[group_id]` was set inside the worker coroutine,
which does not run until after `create_task` returns. Concurrent `add_episode_task` calls all read
the flag as False in that window and each spawn a worker — measured: 8 adds, 8 workers on one
group, in a class whose whole purpose is "sequential episode processing queues by group_id".

The flag is not a cache. `add_episode_task` reads it to decide whether to spawn, so any state
where the flag is True with no live worker is terminal: the queue keeps accepting episodes,
nothing drains them, and no error is raised. These tests pin the properties that make that state
unreachable, whatever kills the worker.
"""

import asyncio
import gc

import pytest

from src.services.queue_service import QueueService


async def _drain(queue_service: QueueService, group_id: str, timeout: float = 2.0) -> None:
    """Wait until the group's queue is empty, or fail the test. (asyncio.timeout is 3.11+; the
    box runs 3.10, so this uses wait_for.)"""

    async def _poll():
        while queue_service.get_queue_size(group_id) > 0:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)
    await asyncio.sleep(0.05)  # let the worker finish the in-flight item


@pytest.mark.asyncio
async def test_no_duplicate_workers_under_concurrent_adds():
    """The measured bug: setting the flag in the caller is what closes the double-spawn race.

    Against the old code this fails with 8 workers for one group.
    """
    qs = QueueService()

    async def work():
        await asyncio.sleep(0)

    await asyncio.gather(*[qs.add_episode_task('g', work) for _ in range(8)])
    tasks = [t for t in asyncio.all_tasks() if '_process_episode_queue' in str(t.get_coro())]
    assert len(tasks) <= 1, f'expected at most one worker per group, found {len(tasks)}'


@pytest.mark.asyncio
async def test_worker_task_is_strongly_referenced():
    """asyncio holds only weak references to tasks; the service must hold the task itself.

    Defensive rather than a reproduction: a worker parked on queue.get() is reachable via the
    queue's getter future, and no collection was observed in production. Pinned because the
    documented contract asks for it and the failure it prevents is unrecoverable.
    """
    qs = QueueService()
    done = asyncio.Event()

    async def work():
        done.set()

    await qs.add_episode_task('g', work)

    assert isinstance(qs._worker_tasks.get('g'), asyncio.Task), 'worker task must be retained'
    assert qs.is_worker_running('g') is True
    await asyncio.wait_for(done.wait(), timeout=2)


@pytest.mark.asyncio
async def test_queue_still_drains_after_gc_between_episodes():
    """An idle worker must survive a GC pass and keep draining.

    This passes on the old code too — it guards the property, it does not reproduce a known
    failure. Kept so a future change that drops the reference gets caught here.
    """
    qs = QueueService()
    processed = []

    async def work(n):
        processed.append(n)

    await qs.add_episode_task('g', lambda: work(1))
    await _drain(qs, 'g')
    assert processed == [1]

    # The worker is now idle at `await queue.get()`.
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0.01)

    await qs.add_episode_task('g', lambda: work(2))
    await _drain(qs, 'g')
    assert processed == [1, 2], 'queue stopped draining after GC — the worker was collected'


@pytest.mark.asyncio
async def test_failing_episode_does_not_kill_the_worker():
    """One bad episode must not take the queue down with it. (Also passes on the old code.)"""
    qs = QueueService()
    processed = []

    async def boom():
        raise RuntimeError('extraction blew up')

    async def ok():
        processed.append('ok')

    await qs.add_episode_task('g', boom)
    await qs.add_episode_task('g', ok)
    await _drain(qs, 'g')

    assert processed == ['ok']
    assert qs.is_worker_running('g') is True


@pytest.mark.asyncio
async def test_flag_is_cleared_when_the_worker_dies():
    """A flag left True without a live worker is the terminal state — never leave it."""
    qs = QueueService()
    started = asyncio.Event()

    async def work():
        started.set()

    await qs.add_episode_task('g', work)
    await asyncio.wait_for(started.wait(), timeout=2)

    task = qs._worker_tasks['g']
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.05)  # let the done-callback run

    assert qs.is_worker_running('g') is False, 'flag must not outlive its worker'
    assert 'g' not in qs._worker_tasks

    # ...and the service must accept work again rather than swallow it.
    done = asyncio.Event()

    async def work2():
        done.set()

    await qs.add_episode_task('g', work2)
    await asyncio.wait_for(done.wait(), timeout=2)


@pytest.mark.asyncio
async def test_work_queued_behind_a_dead_worker_still_gets_processed():
    """Episodes queued behind a worker that dies must not be stranded."""
    qs = QueueService()
    processed = []
    gate = asyncio.Event()

    async def slow():
        await gate.wait()
        processed.append('slow')

    async def later():
        processed.append('later')

    await qs.add_episode_task('g', slow)
    await asyncio.sleep(0.02)          # worker picks up `slow` and blocks on the gate
    await qs.add_episode_task('g', later)  # queued behind it

    qs._worker_tasks['g'].cancel()     # worker dies holding an unfinished queue
    await asyncio.gather(qs._worker_tasks.get('g'), return_exceptions=True)
    gate.set()
    await asyncio.sleep(0.05)

    # Cancellation is a deliberate shutdown, so no replacement is spawned — but the flag must be
    # clear so the very next episode revives the queue instead of vanishing into it.
    assert qs.is_worker_running('g') is False
    await qs.add_episode_task('g', lambda: asyncio.sleep(0))
    await _drain(qs, 'g')
    assert 'later' in processed, 'work queued behind a dead worker must still be processed'

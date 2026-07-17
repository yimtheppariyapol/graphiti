"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Dictionary to track if a worker is running for each group_id
        self._queue_workers: dict[str, bool] = {}
        # Strong references to the worker tasks. asyncio keeps only WEAK references to tasks
        # (see the asyncio.create_task docs), so a task nothing else refers to may be collected
        # mid-flight. Defensive: a worker parked on queue.get() is in fact reachable via the
        # queue's getter future, and we have not observed a collection here — but the documented
        # contract says to hold the reference, and the failure it would cause is unrecoverable.
        self._worker_tasks: dict[str, asyncio.Task] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already running.
        # _start_worker sets the flag SYNCHRONOUSLY, before yielding. Setting it inside the
        # worker instead (as this used to) leaves a window between create_task and the worker's
        # first line in which every other caller still reads False and spawns its own worker:
        # 8 concurrent adds → 8 workers on one group, i.e. the opposite of the sequential
        # processing this class exists to provide.
        if not self._queue_workers.get(group_id, False):
            self._start_worker(group_id)

        return self._episode_queues[group_id].qsize()

    def _start_worker(self, group_id: str) -> None:
        """Spawn the queue worker for a group and keep it referenced and accounted for."""
        self._queue_workers[group_id] = True
        task = asyncio.create_task(self._process_episode_queue(group_id))
        self._worker_tasks[group_id] = task
        task.add_done_callback(partial(self._on_worker_done, group_id))

    def _on_worker_done(self, group_id: str, task: asyncio.Task) -> None:
        """Clear worker state on EVERY exit path, and never leave queued work unattended.

        The flag is not a cache: add_episode_task reads it and declines to start a worker when it
        is True. So a flag left True without a live worker is terminal — the queue keeps accepting
        episodes and nobody ever drains them, with no error raised anywhere. The worker's own
        `finally` covers only the paths that unwind; a done-callback also covers cancellation and
        destruction.
        """
        self._queue_workers[group_id] = False
        self._worker_tasks.pop(group_id, None)

        if task.cancelled():
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        else:
            exc = task.exception()
            if exc is not None:
                logger.error(
                    f'Episode queue worker for group_id {group_id} died: {exc!r}'
                )

        # Work left behind must not sit in a queue nobody is reading. Cancellation is a deliberate
        # shutdown, so it gets no replacement — but the cleared flag above still lets the next
        # add_episode_task revive the queue rather than drop into it.
        queue = self._episode_queues.get(group_id)
        if queue is not None and not queue.empty() and not task.cancelled():
            logger.warning(
                f'Episode queue worker for group_id {group_id} exited with {queue.qsize()} '
                f'episode(s) still queued — starting a replacement worker'
            )
            self._start_worker(group_id)

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        # _queue_workers[group_id] is owned by _start_worker / _on_worker_done. The worker must
        # not touch it: setting it here is what opened the duplicate-spawn window, and clearing it
        # in a `finally` misses the exit paths that never unwind.

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await self._episode_queues[group_id].get()

                try:
                    # Process the episode
                    await process_func()
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {str(e)}'
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
            raise  # let the task settle as cancelled so _on_worker_done can tell it apart
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
            raise  # surface it on the task; _on_worker_done logs it and revives the queue
        finally:
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        return self._queue_workers.get(group_id, False)

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                # Process the episode using the graphiti client
                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)

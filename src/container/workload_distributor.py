#!/usr/bin/env python3
"""
Workload Distributor for Container System

This module provides workload distribution and prioritization capabilities
for the AI container system, ensuring optimal resource allocation.
"""

import time
import logging
import threading
import queue
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass
import uuid

# Configure logging
logger = logging.getLogger("WorkloadDistributor")

class WorkloadPriority(Enum):
    """
    Enum representing workload priority levels.
    
    Priorities determine workload scheduling order and resource allocation.
    """
    CRITICAL = auto()    # Highest priority, preempts other tasks
    HIGH = auto()        # High priority tasks, prioritized over normal
    NORMAL = auto()      # Standard priority for most workloads
    LOW = auto()         # Low priority background tasks
    BATCH = auto()       # Lowest priority batch processing

@dataclass
class Task:
    """
    Represents a task to be scheduled and executed.
    
    Contains all metadata necessary for task scheduling and monitoring.
    """
    task_id: str
    name: str
    input_data: Any
    priority: WorkloadPriority
    execution_fn: Callable
    max_retries: int = 3
    timeout_seconds: float = 300.0
    created_at: float = time.time()
    dependencies: Set[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()

@dataclass
class TaskStatus:
    """
    Represents the current status of a task.
    """
    task_id: str
    state: str  # 'pending', 'running', 'completed', 'failed', 'canceled'
    priority: WorkloadPriority
    progress: float  # 0.0 to 1.0
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    retries: int = 0
    result: Any = None

class WorkloadDistributor:
    """
    Distributes workloads across container resources.
    
    Handles task scheduling, prioritization, execution monitoring, and status tracking.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the workload distributor.
        
        Args:
            max_concurrent_tasks: Maximum number of tasks to run concurrently
        """
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.task_statuses: Dict[str, TaskStatus] = {}
        self.active_tasks: Dict[str, threading.Thread] = {}
        self.task_results: Dict[str, Any] = {}
        self.max_concurrent_tasks: int = max_concurrent_tasks
        self.completed_task_history: List[str] = []
        self.max_history_size: int = 1000
        
        # Threading controls
        self._stop_event: threading.Event = threading.Event()
        self._queue_lock: threading.Lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_task_complete: Optional[Callable[[str, Any], None]] = None
        self.on_task_failed: Optional[Callable[[str, str], None]] = None
        
        logger.info(f"WorkloadDistributor initialized with max {max_concurrent_tasks} concurrent tasks")
    
    def start(self) -> None:
        """
        Start the workload distributor scheduler thread.
        
        Begins processing tasks in the priority queue based on available resources.
        """
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            logger.warning("Workload scheduler is already running")
            return
            
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._process_queue,
            daemon=True
        )
        self._scheduler_thread.start()
        logger.info("Workload scheduler started")
    
    def stop(self) -> None:
        """
        Stop the workload distributor.
        
        Gracefully stops the scheduler and waits for it to terminate.
        """
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            logger.warning("Workload scheduler is not running")
            return
            
        self._stop_event.set()
        self._scheduler_thread.join(timeout=10)
        logger.info("Workload scheduler stopped")
    
    def schedule_task(self, task: Task) -> str:
        """
        Schedule a task for execution.
        
        Args:
            task: The Task object to schedule
            
        Returns:
            The task ID for tracking the task status
        """
        # Assign task ID if not already assigned
        if not task.task_id:
            task.task_id = str(uuid.uuid4())
            
        # Create initial task status
        status = TaskStatus(
            task_id=task.task_id,
            state='pending',
            priority=task.priority,
            progress=0.0,
            created_at=task.created_at
        )
        
        # Store task status
        with self._queue_lock:
            self.task_statuses[task.task_id] = status
            
            # Add to priority queue with priority value ensuring correct ordering
            # Lower enum values indicate higher priority (e.g., CRITICAL=1)
            priority_value = task.priority.value
            self.task_queue.put((priority_value, task.created_at, task))
        
        logger.info(f"Task {task.task_id} ({task.name}) scheduled with {task.priority.name} priority")
        return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was canceled, False if not found or already completed
        """
        with self._queue_lock:
            if task_id not in self.task_statuses:
                logger.warning(f"Cannot cancel task {task_id}: task not found")
                return False
                
            status = self.task_statuses[task_id]
            
            if status.state == 'completed' or status.state == 'failed':
                logger.warning(f"Cannot cancel task {task_id}: already in state {status.state}")
                return False
                
            if status.state == 'pending':
                # Task is still in queue, will be skipped when encountered
                status.state = 'canceled'
                logger.info(f"Canceled pending task {task_id}")
                return True
                
            if status.state == 'running':
                # Mark as canceled, but active thread will need to check this status
                status.state = 'canceled'
                logger.info(f"Marked running task {task_id} for cancellation")
                return True
                
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the current status of a task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            TaskStatus object if task exists, None otherwise
        """
        with self._queue_lock:
            if task_id in self.task_statuses:
                # Create a copy to avoid external modifications
                status_copy = TaskStatus(
                    task_id=self.task_statuses[task_id].task_id,
                    state=self.task_statuses[task_id].state,
                    priority=self.task_statuses[task_id].priority,
                    progress=self.task_statuses[task_id].progress,
                    created_at=self.task_statuses[task_id].created_at,
                    started_at=self.task_statuses[task_id].started_at,
                    completed_at=self.task_statuses[task_id].completed_at,
                    error=self.task_statuses[task_id].error,
                    retries=self.task_statuses[task_id].retries
                )
                
                # Include result if task is completed
                if status_copy.state == 'completed' and task_id in self.task_results:
                    status_copy.result = self.task_results[task_id]
                    
                return status_copy
        
        logger.warning(f"Task {task_id} not found")
        return None
    
    def get_all_tasks(self, filter_state: Optional[str] = None) -> List[TaskStatus]:
        """
        Get status objects for all tasks, optionally filtered by state.
        
        Args:
            filter_state: Optional state to filter by ('pending', 'running', etc.)
            
        Returns:
            List of TaskStatus objects
        """
        with self._queue_lock:
            if filter_state:
                return [status for status in self.task_statuses.values() 
                        if status.state == filter_state]
            else:
                return list(self.task_statuses.values())
    
    def _process_queue(self) -> None:
        """
        Main scheduler loop that processes tasks from the priority queue.
        
        Continuously checks for tasks to execute based on priority and resource availability.
        """
        logger.info("Task queue processing started")
        
        while not self._stop_event.is_set():
            try:
                # Check if we can run more tasks
                with self._queue_lock:
                    active_count = len([s for s in self.task_statuses.values() 
                                       if s.state == 'running'])
                    can_start_more = active_count < self.max_concurrent_tasks
                
                if can_start_more:
                    try:
                        # Get next task with timeout to allow checking stop event
                        priority, _, task = self.task_queue.get(timeout=1.0)
                        
                        with self._queue_lock:
                            # Check if task was canceled while in queue
                            if (task.task_id in self.task_statuses and 
                                self.task_statuses[task.task_id].state == 'canceled'):
                                logger.info(f"Skipping canceled task {task.task_id}")
                                self.task_queue.task_done()
                                continue
                                
                            # Check if dependencies are met
                            dependencies_met = True
                            for dep_id in task.dependencies:
                                if dep_id not in self.task_statuses:
                                    logger.warning(f"Dependency {dep_id} for task {task.task_id} not found")
                                    dependencies_met = False
                                    break
                                    
                                dep_status = self.task_statuses[dep_id]
                                if dep_status.state != 'completed':
                                    # Requeue with same priority but put at end of that priority level
                                    logger.debug(f"Dependency {dep_id} not complete, requeueing {task.task_id}")
                                    dependencies_met = False
                                    self.task_queue.put((priority, time.time(), task))
                                    self.task_queue.task_done()
                                    break
                            
                            if not dependencies_met:
                                continue
                            
                            # Update task status to running
                            status = self.task_statuses[task.task_id]
                            status.state = 'running'
                            status.started_at = time.time()
                            
                        # Start task execution in a separate thread
                        thread = threading.Thread(
                            target=self._execute_task,
                            args=(task,),
                            daemon=True
                        )
                        
                        with self._queue_lock:
                            self.active_tasks[task.task_id] = thread
                            
                        thread.start()
                        logger.info(f"Started execution of task {task.task_id} ({task.name})")
                        self.task_queue.task_done()
                    
                    except queue.Empty:
                        # No tasks in queue, just continue checking
                        pass
                
                # Clean up completed tasks periodically
                self._cleanup_completed_tasks()
                
                # Sleep briefly to avoid CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in task queue processing: {str(e)}", exc_info=True)
                time.sleep(1.0)  # Sleep longer on error
        
        logger.info("Task queue processing stopped")
    
    def _execute_task(self, task: Task) -> None:
        """
        Execute a task and handle its results.
        
        Args:
            task: The Task object to execute
        """
        task_id = task.task_id
        result = None
        error = None
        success = False
        
        try:
            logger.info(f"Executing task {task_id} ({task.name})")
            
            # Check for cancellation before starting
            with self._queue_lock:
                if (task_id in self.task_statuses and 
                    self.task_statuses[task_id].state == 'canceled'):
                    logger.info(f"Task {task_id} was canceled before execution")
                    return
            
            # Execute the task with timeout
            start_time = time.time()
            retry_count = 0
            max_retries = task.max_retries
            
            while retry_count <= max_retries:
                try:
                    # Execute the task function
                    result = task.execution_fn(task.input_data)
                    success = True
                    break
                except Exception as e:
                    retry_count += 1
                    error = str(e)
                    
                    with self._queue_lock:
                        if task_id in self.task_statuses:
                            self.task_statuses[task_id].retries = retry_count
                            self.task_statuses[task_id].error = error
                    
                    if retry_count <= max_retries:
                        logger.warning(
                            f"Task {task_id} failed (attempt {retry_count}/{max_retries}): {error}")
                        # Exponential backoff for retries
                        time.sleep(min(30, 2 ** retry_count))
                    else:
                        logger.error(f"Task {task_id} failed after {max_retries} attempts: {error}")
                        break
            
            execution_time = time.time() - start_time
            
            # Update task status based on result
            with self._queue_lock:
                if task_id in self.task_statuses:
                    status = self.task_statuses[task_id]
                    
                    if success:
                        status.state = 'completed'
                        status.progress = 1.0
                        status.completed_at = time.time()
                        self.task_results[task_id] = result
                        
                        logger.info(f"Task {task_id} completed successfully in {execution_time:.2f}s")
                        
                        # Add to history
                        self.completed_task_history.append(task_id)
                        if len(self.completed_task_history) > self.max_history_size:
                            self.completed_task_history.pop(0)
                            
                        # Invoke completion callback if registered
                        if self.on_task_complete:
                            try:
                                self.on_task_complete(task_id, result)
                            except Exception as e:
                                logger.error(f"Error in task completion callback: {str(e)}")
                    else:
                        status.state = 'failed'
                        status.completed_at = time.time()
                        status.error = error
                        
                        logger.error(f"Task {task_id} failed: {error}")
                        
                        # Invoke failure callback if registered
                        if self.on_task_failed:
                            try:
                                self.on_task_failed(task_id, error)
                            except Exception as e:
                                logger.error(f"Error in task failure callback: {str(e)}")
                    
                    # Remove from active tasks
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
        
        except Exception as e:
            logger.error(f"Unhandled exception in task executor for task {task_id}: {str(e)}", 
                         exc_info=True)
            
            with self._queue_lock:
                if task_id in self.task_statuses:
                    self.task_statuses[task_id].state = 'failed'
                    self.task_statuses[task_id].completed_at = time.time()
                    self.task_statuses[task_id].error = f"Unhandled exception: {str(e)}"
                
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def _cleanup_completed_tasks(self) -> None:
        """
        Clean up completed tasks from memory if they're older than retention period.
        
        This helps manage memory usage for long-running workload distributors.
        """
        current_time = time.time()
        retention_period = 3600  # 1 hour retention by default
        
        with self._queue_lock:
            task_ids_to_remove = []
            
            for task_id, status in self.task_statuses.items():
                if status.state in ('completed', 'failed', 'canceled'):
                    # Only remove if completed more than retention_period ago
                    if (status.completed_at and 
                        (current_time - status.completed_at) > retention_period):
                        task_ids_to_remove.append(task_id)
            
            # Remove old tasks
            for task_id in task_ids_to_remove:
                del self.task_statuses[task_id]
                if task_id in self.task_results:
                    del self.task_results[task_id]
            
            if task_ids_to_remove:
                logger.debug(f"Cleaned up {len(task_ids_to_remove)} old completed tasks")
    
    def update_task_progress(self, task_id: str, progress: float) -> bool:
        """
        Update the progress of a running task.
        
        Args:
            task_id: ID of the task to update
            progress: Progress value (0.0 to 1.0)
            
        Returns:
            True if task was updated, False if not found or not running
        """
        with self._queue_lock:
            if task_id not in self.task_statuses:
                return False
                
            status = self.task_statuses[task_id]
            if status.state != 'running':
                return False
                
            status.progress = max(0.0, min(0.99, progress))  # Cap at 0.99 (1.0 is reserved for completion)
            return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the workload distributor's performance.
        
        Returns:
            Dictionary containing metrics such as:
            - Total tasks processed
            - Tasks by state
            - Average execution time
            - Queue lengths
            - Resource utilization
        """
        with self._queue_lock:
            total_tasks = len(self.task_statuses)
            tasks_by_state = {
                'pending': 0,
                'running': 0,
                'completed': 0,
                'failed': 0,
                'canceled': 0
            }
            
            execution_times = []
            for status in self.task_statuses.values():
                # Count tasks by state
                if status.state in tasks_by_state:
                    tasks_by_state[status.state] += 1
                
                # Calculate execution times for completed tasks
                if status.state in ('completed', 'failed') and status.started_at and status.completed_at:
                    execution_times.append(status.completed_at - status.started_at)
            
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            # Calculate metrics
            metrics = {
                'total_tasks_processed': len(self.completed_task_history),
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'tasks_by_state': tasks_by_state,
                'avg_execution_time': avg_execution_time,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'current_utilization': len(self.active_tasks) / self.max_concurrent_tasks if self.max_concurrent_tasks > 0 else 0
            }
            
            return metrics
    
    def get_estimated_completion_time(self, task_id: str) -> Optional[float]:
        """
        Estimate when a task will be completed based on priority and current workloads.
        
        Args:
            task_id: ID of the task to estimate
            
        Returns:
            Estimated completion time in seconds from now, or None if cannot be estimated
        """
        with self._queue_lock:
            if task_id not in self.task_statuses:
                return None
                
            status = self.task_statuses[task_id]
            
            # If task is already completed or failed, return 0
            if status.state in ('completed', 'failed', 'canceled'):
                return 0
                
            # If task is running, estimate based on progress
            if status.state == 'running':
                if status.progress > 0:
                    elapsed_time = time.time() - status.started_at if status.started_at else 0
                    estimated_total_time = elapsed_time / status.progress
                    return max(0, estimated_total_time - elapsed_time)
                return None  # Cannot estimate without progress
            
            # For pending tasks, count higher priority tasks and estimate queue wait time
            # This is a simple estimation - in reality would depend on many factors
            higher_priority_count = 0
            avg_task_time = 60  # Default assumption: 60 seconds per task
            
            # Use actual average if we have data
            completed_tasks = [s for s in self.task_statuses.values() 
                              if s.state == 'completed' and s.started_at and s.completed_at]
            if completed_tasks:
                avg_task_time = sum(s.completed_at - s.started_at for s in completed_tasks) / len(completed_tasks)
            
            # Count tasks with higher or equal priority that are ahead in the queue
            for s in self.task_statuses.values():
                if s.state == 'pending' and s.priority.value <= status.priority.value:
                    if s.created_at < status.created_at:
                        higher_priority_count += 1
            
            # Calculate wait time
            wait_time = higher_priority_count * avg_task_time / max(1, self.max_concurrent_tasks)
            return wait_time

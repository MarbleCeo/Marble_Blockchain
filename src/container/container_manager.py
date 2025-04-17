import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any

from src.container.resource_monitor import ResourceMonitor, ResourceAlert, ResourceType
from src.container.workload_distributor import (
    WorkloadDistributor, 
    WorkloadPriority,
    Task,
    TaskStatus
)
class ContainerManager:
    """
    Manages the lifecycle and orchestration of AI fragment containers.
    
    This class coordinates between the ResourceMonitor and WorkloadDistributor
    to ensure optimal resource utilization and efficient task processing.
    """
    
    def __init__(
        self, 
        max_containers: int = 10,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
        monitoring_interval: int = 5
    ) -> None:
        """
        Initialize the ContainerManager with resource monitoring and workload distribution.
        
        Args:
            max_containers: Maximum number of containers to manage simultaneously
            cpu_threshold: CPU usage percentage threshold for alerts
            memory_threshold: Memory usage percentage threshold for alerts
            disk_threshold: Disk usage percentage threshold for alerts
            monitoring_interval: Interval in seconds for resource monitoring
        """
        self.logger = logging.getLogger(__name__)
        self.max_containers = max_containers
        self.containers: Dict[str, Dict] = {}
        self.container_lock = threading.RLock()
        
        # Initialize resource monitoring
        self.resource_monitor = ResourceMonitor(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
            disk_threshold=disk_threshold,
            monitoring_interval=monitoring_interval
        )
        self.resource_monitor.register_callback(self._on_resource_update)
        
        # Initialize workload distribution
        self.workload_distributor = WorkloadDistributor(max_workers=max_containers)
        
        self.running = False
        self.management_thread = None

    def start(self) -> None:
        """
        Start the container manager, resource monitor, and workload distributor.
        """
        self.logger.info("Starting Container Manager...")
        self.running = True
        
        # Start the resource monitor
        self.resource_monitor.start()
        
        # Start the workload distributor
        self.workload_distributor.start()
        
        # Start the management thread
        self.management_thread = threading.Thread(
            target=self._management_loop,
            daemon=True
        )
        self.management_thread.start()
        
        self.logger.info("Container Manager started successfully")
    
    def stop(self) -> None:
        """
        Stop the container manager and all associated services.
        """
        self.logger.info("Stopping Container Manager...")
        self.running = False
        
        # Stop the resource monitor
        self.resource_monitor.stop()
        
        # Stop the workload distributor
        self.workload_distributor.stop()
        
        # Wait for the management thread to complete
        if self.management_thread and self.management_thread.is_alive():
            self.management_thread.join(timeout=10)
        
        # Clean up any remaining containers
        self._cleanup_all_containers(force=True)
        self.logger.info("Container Manager stopped successfully")
    
    def _management_loop(self) -> None:
        """
        Main management loop for container lifecycle and resource balancing.
        """
        self.logger.debug("Container management loop started")
        while self.running:
            try:
                # Check container health
                # Check container health
                self._check_container_health()
                
                # Balance workload based on resource availability
                self._balance_workload()
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error in container management loop: {str(e)}")
    
    def create_container(
        self, 
        image_name: str, 
        container_config: Dict = None, 
        resources: Dict = None
    ) -> str:
        """
        Create a new container with the specified configuration.
        
        Args:
            image_name: Name of the container image to use
            container_config: Additional configuration parameters for the container
            resources: Resource limits and requests for the container
            
        Returns:
            container_id: ID of the created container
        """
        container_id = str(uuid.uuid4())
        
        with self.container_lock:
            if len(self.containers) >= self.max_containers:
                self.logger.warning("Maximum container limit reached, cannot create new container")
                raise RuntimeError("Maximum container limit reached")
            
            # Default values if not provided
            container_config = container_config or {}
            resources = resources or {"cpu": 1.0, "memory": 512, "disk": 1024}
            
            # Create the container record
            container = {
                "id": container_id,
                "image": image_name,
                "config": container_config,
                "resources": resources,
                "status": "creating",
                "created_at": time.time(),
                "last_updated": time.time()
            }
            
            # Add to containers dictionary
            self.containers[container_id] = container
        
        # Request the workload distributor to prepare the environment
        try:
            # This is a placeholder for actual container creation logic
            # In a real implementation, this would interact with container runtime
            self.logger.info(f"Creating container {container_id} with image {image_name}")
            
            # Update container status
            with self.container_lock:
                if container_id in self.containers:
                    self.containers[container_id]["status"] = "running"
                    self.containers[container_id]["last_updated"] = time.time()
            
            return container_id
        except Exception as e:
            self.logger.error(f"Failed to create container: {str(e)}")
            with self.container_lock:
                if container_id in self.containers:
                    self.containers[container_id]["status"] = "failed"
                    self.containers[container_id]["error"] = str(e)
                    self.containers[container_id]["last_updated"] = time.time()
            raise
    
    def stop_container(self, container_id: str) -> bool:
        """
        Stop a running container.
        
        Args:
            container_id: ID of the container to stop
            
        Returns:
            bool: True if container was stopped successfully, False otherwise
        """
        with self.container_lock:
            if container_id not in self.containers:
                self.logger.warning(f"Container {container_id} not found")
                return False
                
            if self.containers[container_id]["status"] not in ["running", "paused"]:
                self.logger.warning(
                    f"Container {container_id} is not in a stoppable state. "
                    f"Current status: {self.containers[container_id]['status']}"
                )
                return False
                
            # Update container status
            self.containers[container_id]["status"] = "stopping"
            self.containers[container_id]["last_updated"] = time.time()
        
        try:
            # This is a placeholder for actual container stopping logic
            # In a real implementation, this would interact with container runtime
            self.logger.info(f"Stopping container {container_id}")
            
            # Update container status
            with self.container_lock:
                if container_id in self.containers:
                    self.containers[container_id]["status"] = "stopped"
                    self.containers[container_id]["last_updated"] = time.time()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop container {container_id}: {str(e)}")
            with self.container_lock:
                if container_id in self.containers:
                    self.containers[container_id]["status"] = "error"
                    self.containers[container_id]["error"] = str(e)
                    self.containers[container_id]["last_updated"] = time.time()
            return False
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """
        Remove a container from the system.
        
        Args:
            container_id: ID of the container to remove
            force: If True, force removal even if the container is running
            
        Returns:
            bool: True if container was removed successfully, False otherwise
        """
        with self.container_lock:
            if container_id not in self.containers:
                self.logger.warning(f"Container {container_id} not found")
                return False
                
            container_status = self.containers[container_id]["status"]
            if container_status == "running" and not force:
                self.logger.warning(
                    f"Container {container_id} is still running. "
                    f"Use force=True to force removal."
                )
                return False
                
            # Update container status
            self.containers[container_id]["status"] = "removing"
            self.containers[container_id]["last_updated"] = time.time()
        
        try:
            # This is a placeholder for actual container removal logic
            # In a real implementation, this would interact with container runtime
            self.logger.info(f"Removing container {container_id}")
            
            # If container is running and force is True, stop it first
            if container_status == "running" and force:
                self.stop_container(container_id)
            
            # Remove the container from our tracking
            with self.container_lock:
                if container_id in self.containers:
                    del self.containers[container_id]
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove container {container_id}: {str(e)}")
            with self.container_lock:
                if container_id in self.containers:
                    self.containers[container_id]["status"] = "error"
                    self.containers[container_id]["error"] = str(e)
                    self.containers[container_id]["last_updated"] = time.time()
            return False
    
    def get_container_status(self, container_id: str) -> Optional[Dict]:
        """
        Get the current status of a container.
        
        Args:
            container_id: ID of the container
            
        Returns:
            Optional[Dict]: Container status information or None if not found
        """
        with self.container_lock:
            if container_id not in self.containers:
                return None
            
            # Create a copy to avoid external modifications
            return dict(self.containers[container_id])
    
    def list_containers(self, status_filter: Optional[str] = None) -> List[Dict]:
        """
        List all managed containers, optionally filtered by status.
        
        Args:
            status_filter: If provided, only return containers with this status
            
        Returns:
            List[Dict]: List of container information dictionaries
        """
        with self.container_lock:
            if status_filter:
                return [
                    dict(container) for container in self.containers.values()
                    if container["status"] == status_filter
                ]
            else:
                return [dict(container) for container in self.containers.values()]
    
    def execute_task_in_container(
        self, 
        container_id: str, 
        task_definition: Dict, 
        priority: WorkloadPriority = WorkloadPriority.NORMAL
    ) -> Optional[str]:
        """
        Execute a task in the specified container.
        
        Args:
            container_id: ID of the container to use
            task_definition: Definition of the task to execute
            priority: Priority level for the task
            
        Returns:
            Optional[str]: Task ID if submitted successfully, None otherwise
        """
        # Check if the container exists and is running
        container_status = self.get_container_status(container_id)
        if not container_status or container_status["status"] != "running":
            self.logger.error(
                f"Container {container_id} is not available for task execution. "
                f"Status: {container_status['status'] if container_status else 'Not found'}"
            )
            return None
            
        # Create a new task
        task = Task(
            task_id=str(uuid.uuid4()),
            container_id=container_id,
            definition=task_definition,
            priority=priority
        )
        
        # Submit the task to the workload distributor
        return self.workload_distributor.submit_task(task)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Optional[TaskStatus]: Task status or None if not found
        """
        return self.workload_distributor.get_task_status(task_id)
    
    def get_system_metrics(self) -> Dict:
        """
        Get comprehensive system metrics including resource usage and container statistics.
        
        Returns:
            Dict: System metrics information
        """
        # Get resource metrics
        resource_metrics = self.resource_monitor.get_current_metrics()
        
        # Get container statistics
        container_count = 0
        running_containers = 0
        
        with self.container_lock:
            container_count = len(self.containers)
            running_containers = sum(
                1 for container in self.containers.values() 
                if container["status"] == "running"
            )
        
        # Get workload metrics
        workload_metrics = self.workload_distributor.get_metrics()
        
        # Combine all metrics
        return {
            "resources": resource_metrics,
            "containers": {
                "total": container_count,
                "running": running_containers
            },
            "workload": workload_metrics,
            "timestamp": time.time()
        }
    
    def _on_resource_update(self, alert: ResourceAlert) -> None:
        """
        Handle resource alerts from the resource monitor.
        
        Args:
            alert: Resource alert information
        """
        self.logger.warning(
            f"Resource alert: {alert.resource_type.name} usage at {alert.current_value}% "
            f"exceeds threshold of {alert.threshold}%"
        )
        
        # Take action based on the resource type and severity
        if alert.resource_type == ResourceType.CPU and alert.current_value > 90:
            self._handle_critical_cpu_usage()
        elif alert.resource_type == ResourceType.MEMORY and alert.current_value > 90:
            self._handle_critical_memory_usage()
        elif alert.resource_type == ResourceType.DISK and alert.current_value > 95:
            self._handle_critical_disk_usage()
    
    def _handle_critical_disk_usage(self) -> None:
        """Handle critical disk usage by cleaning up unnecessary containers and files."""
        self.logger.warning("Critical disk usage detected, cleaning up disk space")
        
        # Get a list of containers sorted by creation time (oldest first)
        with self.container_lock:
            # Prioritize stopped or failed containers for removal
            removable_containers = [
                container_id for container_id, container in self.containers.items()
                if container["status"] in ["stopped", "failed", "error"]
            ]
            
            # Remove old containers
            for container_id in removable_containers:
                self.logger.info(f"Removing container {container_id} due to disk pressure")
                self.remove_container(container_id)
                
            # If we still need to free up space
            metrics = self.resource_monitor.get_current_metrics()
            if any(partition["percent"] > 95 for partition in metrics["disk"]["partitions"]):
                self.logger.warning("Disk usage still critical after removing stopped containers")
                
                # As a more aggressive measure, remove oldest running containers
                running_containers = [
                    (container_id, container) for container_id, container in self.containers.items()
                    if container["status"] == "running"
                ]
                
                # Sort by creation time (oldest first)
                running_containers.sort(key=lambda x: x[1]["created_at"])
                
                # Remove oldest up to 10% of running containers
                remove_count = max(1, len(running_containers) // 10)
                for i in range(min(remove_count, len(running_containers))):
                    container_id, _ = running_containers[i]
                    self.logger.warning(f"Stopping and removing container {container_id} due to critical disk pressure")
                    self.stop_container(container_id)
                    self.remove_container(container_id, force=True)

    def _handle_critical_cpu_usage(self) -> None:
        """Handle critical CPU usage by pausing lower priority tasks."""
        self.logger.warning("Critical CPU usage detected, pausing low-priority tasks")
        self.workload_distributor.pause_low_priority_tasks()
    
    def _handle_critical_memory_usage(self) -> None:
        """Handle critical memory usage by stopping some containers."""
        self.logger.warning("Critical memory usage detected, stopping idle containers")
        
        with self.container_lock:
            # Find containers that are running but have been idle the longest
            idle_containers = [
                container_id for container_id, container in self.containers.items()
                if container["status"] == "running"
            ]
            
            # Sort by last updated time (oldest first)
            idle_containers.sort(key=lambda cid: self.containers[cid]["last_updated"])
            
            # If we have idle containers, stop the oldest ones until memory pressure is reduced
            if idle_containers:
                # Start with stopping up to 20% of containers
                stop_count = max(1, len(idle_containers) // 5)
                for i in range(min(stop_count, len(idle_containers))):
                    container_id = idle_containers[i]
                    self.logger.info(f"Stopping idle container {container_id} due to memory pressure")
                    self.stop_container(container_id)
            else:
                # If no idle containers, we may need to stop some running ones
                self.logger.warning("No idle containers found when handling critical memory usage")
                # Get current resource metrics
                metrics = self.resource_monitor.get_current_metrics()
                if metrics["memory"]["percent"] > 95:  # If still critical
                    self.logger.critical("Memory usage critically high, stopping all non-essential containers")
                return False
            
            container = self.containers[container_id]
            if container["state"] == ContainerState.RUNNING:
                logger.warning(f"Container {container_id} is already running")
                return True
            
            try:
                # Prepare the environment
                env = os.environ.copy()
                env.update(container["config"].environment)
                
                # Build the command
                cmd = container["config"].command
                if not cmd:
                    logger.error(f"No command specified for container {container_id}")
                    return False
                
                # Launch the process
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=container["dir"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Update container status
                container["state"] = ContainerState.RUNNING
                container["started_at"] = time.time()
                container["process"] = process
                
                logger.info(f"Container {container_id} ({container['config'].name}) started")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start container {container_id}: {e}")
                return False
    
    def stop_container(self, container_id: str) -> bool:
        """
        Stop a running container.
        
        Args:
            container_id: ID of the container to stop
            
        Returns:
            Success status
        """
        with self.container_lock:
            if container_id not in self.containers:
                logger.error(f"Cannot stop container {container_id}: not found")
                return False
            
            container = self.containers[container_id]
            if container["state"] != ContainerState.RUNNING:
                logger.warning(f"Container {container_id} is not running")
                return True
            
            try:
                process = container["process"]
                if process:
                    # Attempt graceful termination
                    process.terminate()
                    try:
                        # Wait for process to terminate
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        process.kill()
                
                # Update container status
                container["state"] = ContainerState.STOPPED
                container["stopped_at"] = time.time()
                container["exit_code"] = process.returncode if process else None
                
                logger.info(f"Container {container_id} ({container['config'].name}) stopped")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop container {container_id}: {e}")
                return False
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """
        Remove a container and its associated files.
        
        Args:
            container_id: ID of the container to remove
            force: Force removal even if running
            
        Returns:
            Success status
        """
        with self.container_lock:
            if container_id not in self.containers:
                logger.error(f"Cannot remove container {container_id}: not found")
                return False
            
            container = self.containers[container_id]
            
            # Stop the container first if it's running
            if container["state"] == ContainerState.RUNNING:
                if not force:
                    logger.error(f"Cannot remove running container {container_id} without force flag")
                    return False
                self.stop_container(container_id)
            
            try:
                # Clean up container directory
                container_dir = container["dir"]
                if container_dir.exists():
                    shutil.rmtree(container_dir)
                
                # Remove from containers dict
                del self.containers[container_id]
                
                logger.info(f"Container {container_id} removed")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove container {container_id}: {e}")
                return False
    
    def get_container_status(self, container_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a container.
        
        Args:
            container_id: ID of the container
            
        Returns:
            Container status dictionary or None if not found
        """
        with self.container_lock:
            if container_id not in self.containers:
                logger.warning(f"Container {container_id} not found")
                return None
            
            container = self.containers[container_id]
            
            # Check if the process is still alive
            if container["state"] == ContainerState.RUNNING and container["process"]:
                if container["process"].poll() is not None:
                    # Process has terminated
                    container["state"] = ContainerState.STOPPED
                    container["stopped_at"] = time.time()
                    container["exit_code"] = container["process"].returncode
            
            # Return status information
            return {
                "id": container_id,
                "name": container["config"].name,
                "state": container["state"].name,
                "created_at": container["created_at"],
                "started_at": container["started_at"],
                "stopped_at": container["stopped_at"],
                "exit_code": container["exit_code"],
                "cpu_limit": container["config"].cpu_limit,
                "memory_limit": container["config"].memory_limit,
                "disk_limit": container["config"].disk_limit
            }
    
    def _on_resource_update(self, metrics: Dict[str, Any]) -> None:
        """
        Handle resource updates from the resource monitor.
        
        This method is called whenever new resource metrics are available.
        It provides an opportunity to react to system resource changes.
        
        Args:
            metrics: Resource metrics dictionary
        """
        # Check for critical resource thresholds
        critical_cpu = metrics["cpu"]["percent"] > 95
        critical_memory = metrics["memory"]["percent"] > 90
        
        # React to critical resource conditions
        if critical_cpu or critical_memory:
            logger.warning("Critical resource condition detected")
            
            # Get running containers sorted by creation time (newest first)
            with self.container_lock:
                running_containers = [
                    (cid, c) for cid, c in self.containers.items() 
                    if c["state"] == ContainerState.RUNNING
                ]
                running_containers.sort(key=lambda x: x[1]["created_at"], reverse=True)
                
                # Stop containers if needed until resources are freed
                if running_containers:
                    # Stop the newest container
                    container_id, container = running_containers[0]
                    logger.warning(f"Stopping container {container_id} due to resource constraints")
                    self.stop_container(container_id)
                    
        # Log warning level alerts
        if metrics["cpu"]["percent"] > 80:
            logger.info(f"High CPU usage: {metrics['cpu']['percent']}%")
        
        if metrics["memory"]["percent"] > 75:
            logger.info(f"High memory usage: {metrics['memory']['percent']}%")
            
    def list_containers(self, filter_state: Optional[ContainerState] = None) -> List[Dict[str, Any]]:
        """
        List all containers with optional state filtering.
        
        Args:
            filter_state: Optional state to filter containers by
            
        Returns:
            List of container status dictionaries
        """
        with self.container_lock:
            containers = []
            for container_id in self.containers:
                status = self.get_container_status(container_id)
                if status and (filter_state is None or status["state"] == filter_state.name):
                    containers.append(status)
            
            return containers
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        try:
            # Stop all running containers
            with self.container_lock:
                for container_id, container in list(self.containers.items()):
                    if container["state"] == ContainerState.RUNNING:
                        try:
                            self.stop_container(container_id)
                        except Exception as e:
                            logger.error(f"Error stopping container {container_id} during cleanup: {e}")
            
            # Remove resource monitor listener
            if hasattr(self, 'resource_monitor'):
                self.resource_monitor.remove_listener(self._on_resource_update)
        except Exception as e:
            logger.error(f"Error during ContainerManager cleanup: {e}")


    
    def __init__(self, container_manager: ContainerManager, fragmenter: AIFragmenter):
        """
        Initialize the workload distributor.
        
        Args:
            container_manager: ContainerManager instance to use for container lifecycle
            fragmenter: AIFragmenter instance to use for task fragmentation
        """
        self.container_manager = container_manager
        self.fragmenter = fragmenter
        self.scheduler_lock = threading.RLock()
        self.fragment_states: Dict[str, Dict[str, Any]] = {}
        self.execution_queue: List[Tuple[int, Fragment]] = []  # (priority, fragment)
        self.active_fragments: Set[str] = set()
        self.max_parallel_fragments = os.cpu_count() or 4
        self.scheduler_thread = None
        self.running = False
        
        # Initialize metrics
        self.metrics = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "fragments_scheduled": 0,
            "fragments_completed": 0,
            "fragments_failed": 0,
            "total_execution_time": 0
        }
        
        logger.info("WorkloadDistributor initialized")
    
    def start(self):
        """Start the workload distribution scheduler thread."""
        if self.running:
            logger.warning("WorkloadDistributor is already running")
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("WorkloadDistributor scheduler started")
    
    def stop(self):
        """Stop the workload distribution scheduler thread."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
            self.scheduler_thread = None
        logger.info("WorkloadDistributor scheduler stopped")
    
    def distribute_workload(self, task: Task, strategy: Optional[str] = None) -> str:
        """
        Fragment and distribute a workload task for execution.
        
        Args:
            task: The task to distribute
            strategy: Optional fragmentation strategy
            
        Returns:
            Task ID
        """
        logger.info(f"Distributing workload for task {task.id} with {strategy or 'default'} strategy")
        
        # Fragment the task
        fragments = self.fragmenter.fragment_task(task, strategy)
        
        # Schedule the fragments
        with self.scheduler_lock:
            for fragment in fragments:
                self._schedule_fragment(fragment, task.priority)
            
            # Update metrics
            self.metrics["tasks_scheduled"] += 1
            self.metrics["fragments_scheduled"] += len(fragments)
        
        logger.info(f"Scheduled {len(fragments)} fragments for task {task.id}")
        return task.id
    
    def _schedule_fragment(self, fragment: Fragment, priority: int) -> None:
        """
        Schedule a fragment for execution.
        
        Args:
            fragment: The fragment to schedule
            priority: Fragment priority (higher number = higher priority)
        """
        with self.scheduler_lock:
            # Register the fragment state
            self.fragment_states[fragment.id] = {
                "fragment": fragment,
                "state": "queued",
                "queued_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "container_id": None,
                "result": None,
                "error": None
            }
            
            # Add to priority queue (using list as a simple priority queue)
            self.execution_queue.append((priority, fragment))
            
            # Sort the queue by priority (highest first)
            self.execution_queue.sort(key=lambda x: x[0], reverse=True)
            
            logger.debug(f"Fragment {fragment.id} scheduled with priority {priority}")
    
    def _scheduler_loop(self) -> None:
        """Main scheduling loop that processes fragments from the queue."""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                self._process_queue()
                self._monitor_execution()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)  # Wait longer after error
    
    def _process_queue(self) -> None:
        """Process the execution queue and start new fragments when resources are available."""
        with self.scheduler_lock:
            # Check if we can start new fragments
            available_slots = self.max_parallel_fragments - len(self.active_fragments)
            
            if available_slots <= 0 or not self.execution_queue:
                return
            
            # Get resource metrics to make informed decisions
            resource_metrics = self.container_manager.resource_monitor.get_current_metrics()
            cpu_available = 100 - resource_metrics["cpu"]["percent"]
            memory_available = resource_metrics["memory"]["available"] / (1024 * 1024)  # MB
            
            # Process queue - try to start as many fragments as possible
            i = 0
            while i < len(self.execution_queue) and available_slots > 0:
                priority, fragment = self.execution_queue[i]
                
                # Check dependencies
                for dep_id in fragment.dependencies:
                    if dep_id not in self.fragment_states or \
                       self.fragment_states[dep_id]["state"] != "completed":
                        dependencies_met = False
                        break
                
                # Check resource requirements against available resources
                resource_requirements_met = (
                    fragment.cpu_allocation <= cpu_available / 100 and
                    fragment.memory_allocation <= memory_available
                )
                
                if dependencies_met and resource_requirements_met:
                    # Remove from queue and start fragment
                    self.execution_queue.pop(i)
                    self._start_fragment(fragment)
                    available_slots -= 1
                else:
                    i += 1  # Check next fragment
    
    def _start_fragment(self, fragment: Fragment) -> None:
        """
        Start execution of a fragment in a container.
        
        Args:
            fragment: The fragment to start
        """
        try:
            logger.info(f"Starting fragment {fragment.id} for task {fragment.task_id}")
            
            # Create container configuration
            container_config = ContainerConfig(
                name=f"fragment-{fragment.id}",
                memory_limit=fragment.memory_allocation,
                cpu_limit=fragment.cpu_allocation,
                disk_limit=fragment.storage_allocation,
                environment={
                    "FRAGMENT_ID": fragment.id,
                    "TASK_ID": fragment.task_id,
                    "PYTHONPATH": "/app"
                },
                command=["python", "-m", "fragment_executor", fragment.code_path]
            )
            
            # Create input data file for the container
            container_id = self.container_manager.create_container(container_config)
            
            # Write input data to container directory
            container_dir = self.container_manager.containers[container_id]["dir"]
            with open(container_dir / "input.json", "w") as f:
                json.dump(fragment.input_data, f)
            
            # Update fragment state
            with self.scheduler_lock:
                self.fragment_states[fragment.id]["state"] = "running"
                self.fragment_states[fragment.id]["started_at"] = time.time()
                self.fragment_states[fragment.id]["container_id"] = container_id
                self.active_fragments.add(fragment.id)
            
            # Start the container
            success = self.container_manager.start_container(container_id)
            if not success:
                raise RuntimeError(f"Failed to start container for fragment {fragment.id}")
                
            logger.info(f"Fragment {fragment.id} started successfully in container {container_id}")
            
        except Exception as e:
            logger.error(f"Error starting fragment {fragment.id}: {e}")
            with self.scheduler_lock:
                self.fragment_states[fragment.id]["state"] = "failed"
                self.fragment_states[fragment.id]["error"] = str(e)
                if fragment.id in self.active_fragments:
                    self.active_fragments.remove(fragment.id)
                self.metrics["fragments_failed"] += 1
    
    def _monitor_execution(self) -> None:
        """Monitor running fragments and handle completed or failed fragments."""
        with self.scheduler_lock:
            # Create a copy to avoid modification during iteration
            active_fragments = list(self.active_fragments)
        
        for fragment_id in active_fragments:
            try:
                fragment_state = self.fragment_states.get(fragment_id)
                if not fragment_state or fragment_state["state"] != "running":
                    continue
                
                container_id = fragment_state["container_id"]
                if not container_id:
                    logger.warning(f"Fragment {fragment_id} has no associated container")
                    continue
                
                # Check container status
                container_status = self.container_manager.get_container_status(container_id)
                if not container_status:
                    logger.warning(f"Container {container_id} for fragment {fragment_id} not found")
                    continue
                
                if container_status["state"] == "STOPPED":
                    # Container has finished executing
                    self._cleanup_fragment(fragment_id, container_id, container_status["exit_code"] == 0)
                elif container_status["state"] == "FAILED":
                    # Container failed
                    self._cleanup_fragment(fragment_id, container_id, False)
                    
            except Exception as e:
                logger.error(f"Error monitoring fragment {fragment_id}: {e}")
    
    def _cleanup_fragment(self, fragment_id: str, container_id: str, success: bool) -> None:
        """
        Clean up resources after fragment execution.
        
        Args:
            fragment_id: ID of the fragment
            container_id: ID of the container
            success: Whether execution was successful
        """
        try:
            with self.scheduler_lock:
                fragment_state = self.fragment_states.get(fragment_id)
                if not fragment_state:
                    logger.warning(f"Fragment state not found for {fragment_id} during cleanup")
                    return
                
                # Update fragment state
                fragment_state["completed_at"] = time.time()
                if success:
                    fragment_state["state"] = "completed"
                    self.metrics["fragments_completed"] += 1
                    
                    # Try to read result from container
                    try:
                        container_dir = self.container_manager.containers[container_id]["dir"]
                        result_file = container_dir / "output.json"
                        if result_file.exists():
                            with open(result_file, "r") as f:
                                fragment_state["result"] = json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading result for fragment {fragment_id}: {e}")
                else:
                    fragment_state["state"] = "failed"
                    self.metrics["fragments_failed"] += 1
                    
                    # Try to read error from container
                    try:
                        container_dir = self.container_manager.containers[container_id]["dir"]
                        error_file = container_dir / "error.log"
                        if error_file.exists():
                            with open(error_file, "r") as f:
                                fragment_state["error"] = f.read()
                    except Exception as e:
                        logger.error(f"Error reading error log for fragment {fragment_id}: {e}")
                
                # Remove from active fragments
                if fragment_id in self.active_fragments:
                    self.active_fragments.remove(fragment_id)
                
                # Check if all fragments for the task are completed
                task_id = fragment_state["fragment"].task_id
                all_fragments_completed = True
                for fstate in self.fragment_states.values():
                    if fstate["fragment"].task_id == task_id and fstate["state"] not in ["completed", "failed"]:
                        all_fragments_completed = False
                        break
                
                if all_fragments_completed:
                    # Check if any fragments failed
                    any_failed = False
                    for fstate in self.fragment_states.values():
                        if fstate["fragment"].task_id == task_id and fstate["state"] == "failed":
                            any_failed = True
                            break
                    
                    # Update task metrics
                    if any_failed:
                        self.metrics["tasks_failed"] += 1
                    else:
                        self.metrics["tasks_completed"] += 1
                    
                    logger.info(f"All fragments for task {task_id} completed. Status: {'Failed' if any_failed else 'Success'}")
            
            # Clean up container
            try:
                self.container_manager.stop_container(container_id)
                self.container_manager.remove_container(container_id)
                logger.debug(f"Container {container_id} removed after fragment {fragment_id} completion")
            except Exception as e:
                logger.error(f"Error cleaning up container {container_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error in fragment cleanup for {fragment_id}: {e}")
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a distributed task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task status information
        """
        with self.scheduler_lock:
            # Collect status for all fragments of this task
            fragments = []
            task_fragments = {}
            for fragment_id, fragment_state in self.fragment_states.items():
                if fragment_state["fragment"].task_id == task_id:
                    fragments.append({
                        "id": fragment_id,
                        "state": fragment_state["state"],
                        "queued_at": fragment_state["queued_at"],
                        "started_at": fragment_state["started_at"],
                        "completed_at": fragment_state["completed_at"],
                        "container_id": fragment_state["container_id"],
                        "error": fragment_state["error"]
                    })
                    task_fragments[fragment_id] = fragment_state
            
            # Determine overall task status
            if not task_fragments:
                return {"task_id": task_id, "status": "unknown", "fragments": []}
            
            # Count fragments in each state
            states = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for fragment_state in task_fragments.values():
                state = fragment_state["state"]
                if state in


    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Fragmenter with optional configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.historical_execution_data: Dict[str, List[float]] = {}
        self.fragment_counter: int = 0
        self.strategies: Dict[str, Callable] = {
            "balanced": self._balanced_fragmentation,
            "memory_optimized": self._memory_optimized_fragmentation,
            "compute_optimized": self._compute_optimized_fragmentation,
            "minimal": self._minimal_fragmentation,
            "auto": self._auto_select_strategy
        }
        self.task_history: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("AIFragmenter initialized with %d strategies", len(self.strategies))
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "default_strategy": "balanced",
            "min_fragment_size_mb": 50,
            "max_fragment_size_mb": 2000,
            "max_fragments_per_task": 20,
            "fragment_overhead_mb": 25,
            "execution_time_safety_factor": 1.2,
            "resource_allocation_strategy": "dynamic",
            "historical_data_weight": 0.7
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info("Loaded configuration from %s", config_path)
            except Exception as e:
                logger.error("Failed to load config from %s: %s", config_path, str(e))
        
        return default_config
    
    def fragment_task(self, task: Task, strategy: Optional[str] = None) -> List[Fragment]:
        """
        Fragment a task into container units based on specified strategy.
        
        Args:
            task: The task to fragment
            strategy: Fragmentation strategy to use
            
        Returns:
            List of container fragments
        """
        if not strategy:
            strategy = self.config["default_strategy"]
        
        logger.info("Fragmenting task %s using %s strategy", task.id, strategy)
        
        # Check if strategy exists
        if strategy not in self.strategies:
            logger.warning("Unknown strategy %s, falling back to balanced", strategy)
            strategy = "balanced"
        
        # Estimate execution time for task scheduling
        task.estimated_execution_time = self._estimate_execution_time(task)
        
        # Create fragments using the selected strategy
        fragments = self.strategies[strategy](task)
        
        # Store fragments in the task
        task.fragments = fragments
        
        # Update task history for future optimizations
        self._update_task_history(task)
        
        logger.info("Task %s fragmented into %d units", task.id, len(fragments))
        return fragments
    
    def _estimate_execution_time(self, task: Task) -> float:
        """
        Estimate execution time for a task based on historical data and task properties.
        
        Args:
            task: The task to estimate
            
        Returns:
            Estimated execution time in seconds
        """
        # Start with a base estimate
        base_estimate = 10.0  # Default 10 seconds base
        
        # Consider resource requirements
        cpu_factor = task.resource_requirements.get("cpu_cores", 1) * 0.5
        memory_factor = task.resource_requirements.get("memory_mb", 100) / 100
        
        # Scale by input data size
        data_size_factor = 1.0
        input_size = len(json.dumps(task.input_data))
        if input_size > 1024:
            data_size_factor = (input_size / 1024) * 0.2  # Scale based on KB
        
        # Consider code complexity by file size
        code_factor = 1.0
        if os.path.exists(task.code_path):
            code_size = os.path.getsize(task.code_path) / 1024  # KB
            code_factor = max(1.0, code_size / 50)  # Scale based on code size
        
        # Check historical data for similar tasks
        historical_factor = 1.0
        task_type = task.name.split('_')[0] if '_' in task.name else task.name
        
        if task_type in self.historical_execution_data and self.historical_execution_data[task_type]:
            # Use exponential moving average of past execution times
            historical_times = self.historical_execution_data[task_type]
            if len(historical_times) >= 3:
                # Apply more weight to recent executions
                weights = np.exp(np.linspace(0, 1, len(historical_times)))
                weights = weights / weights.sum()
                weighted_avg = np.average(historical_times, weights=weights)
                historical_factor = weighted_avg / 10.0  # Normalize to the base estimate
        
        # Calculate the combined estimate
        estimate = base_estimate * cpu_factor * memory_factor * data_size_factor * code_factor * historical_factor
        
        # Apply safety factor for better scheduling
        safety_factor = self.config["execution_time_safety_factor"]
        final_estimate = estimate * safety_factor
        
        logger.debug(f"Estimated execution time for task {task.id}: {final_estimate:.2f} seconds")
        return final_estimate
    
    def _balanced_fragmentation(self, task: Task) -> List[Fragment]:
        """
        Create balanced fragments considering both memory and CPU requirements.
        
        Args:
            task: The task to fragment
            
        Returns:
            List of balanced fragments
        """
        # Determine optimal fragment count based on task complexity
        complexity_score = (
            task.resource_requirements.get("cpu_cores", 1) * 
            task.resource_requirements.get("memory_mb", 100) / 100
        )
        
        # Scale fragment count with complexity, but respect max_fragments_per_task
        fragment_count = min(
            max(2, int(complexity_score * 2)), 
            self.config["max_fragments_per_task"]
        )
        
        total_memory = task.resource_requirements.get("memory_mb", 500)
        total_cpu = task.resource_requirements.get("cpu_cores", 1)
        total_storage = task.resource_requirements.get("storage_mb", 1000)
        
        # Distribute resources evenly across fragments
        fragments = []
        for i in range(fragment_count):
            fragment_id = f"{task.id}_fragment_{self.fragment_counter}"
            self.fragment_counter += 1
            
            # Distribute resources (slightly more to earlier fragments)
            weight = 1 + (fragment_count - i) * 0.1
            memory_share = int((total_memory / fragment_count) * weight)
            cpu_share = (total_cpu / fragment_count) * weight
            storage_share = int((total_storage / fragment_count) * weight)
            
            # Create new fragment
            fragment = Fragment(
                id=fragment_id,
                task_id=task.id,
                cpu_allocation=cpu_share,
                memory_allocation=memory_share,
                storage_allocation=storage_share,
                code_path=task.code_path,
                container_type="python",
                input_data={"fragment_index": i, "total_fragments": fragment_count}
            )
            
            # Add dependencies between fragments
            if i > 0:
                fragment.dependencies.append(fragments[i-1].id)
            
            fragments.append(fragment)
        
        return fragments
    
    def _memory_optimized_fragmentation(self, task: Task) -> List[Fragment]:
        """
        Create fragments optimized for memory-intensive workloads.
        
        Args:
            task: The task to fragment
            
        Returns:
            List of memory-optimized fragments
        """
        # For memory-intensive tasks, use fewer fragments with more memory each
        total_memory = task.resource_requirements.get("memory_mb", 500)
        min_fragment_memory = self.config["min_fragment_size_mb"]
        
        # Calculate optimal fragment count based on memory requirements
        fragment_count = min(
            max(1, int(total_memory / (min_fragment_memory * 4))),
            self.config["max_fragments_per_task"] // 2  # Fewer fragments than balanced
        )
        
        fragments = []
        total_cpu = task.resource_requirements.get("cpu_cores", 1)
        total_storage = task.resource_requirements.get("storage_mb", 1000)
        
        for i in range(fragment_count):
            fragment_id = f"{task.id}_memopt_{self.fragment_counter}"
            self.fragment_counter += 1
            
            # For memory-optimized, allocate more memory, less CPU
            memory_share = int(total_memory / fragment_count)
            cpu_share = (total_cpu / fragment_count) * 0.8  # Slightly reduce CPU allocation
            storage_share = int(total_storage / fragment_count)
            
            fragment = Fragment(
                id=fragment_id,
                task_id=task.id,
                cpu_allocation=cpu_share,
                memory_allocation=memory_share,
                storage_allocation=storage_share,
                code_path=task.code_path,
                container_type="python",
                input_data={"fragment_index": i, "total_fragments": fragment_count}
            )
            
            fragments.append(fragment)
        
        return fragments
    
    def _compute_optimized_fragmentation(self, task: Task) -> List[Fragment]:
        """
        Create fragments optimized for compute-intensive workloads.
        
        Args:
            task: The task to fragment
            
        Returns:
            List of compute-optimized fragments
        """
        # For compute-intensive tasks, use more fragments with higher CPU allocation
        total_cpu = task.resource_requirements.get("cpu_cores", 1)
        
        # Calculate optimal fragment count based on CPU requirements
        fragment_count = min(
            max(2, int(total_cpu * 3)),  # More fragments for parallelization
            self.config["max_fragments_per_task"]
        )
        
        fragments = []
        total_memory = task.resource_requirements.get("memory_mb", 500)
        total_storage = task.resource_requirements.get("storage_mb", 1000)
        
        # Create dependency graph for fragments - compute tasks often benefit from a DAG
        # rather than simple linear dependencies
        dependency_matrix = self._create_dependency_graph(fragment_count)
        
        for i in range(fragment_count):
            fragment_id = f"{task.id}_cpuopt_{self.fragment_counter}"
            self.fragment_counter += 1
            
            # For compute-optimized, allocate more CPU, less memory
            cpu_share = (total_cpu / fragment_count) * 1.2  # Increase CPU allocation
            memory_share = int(total_memory / fragment_count * 0.9)  # Slightly reduce memory
            storage_share = int(total_storage / fragment_count)
            
            fragment = Fragment(
                id=fragment_id,
                task_id=task.id,
                cpu_allocation=cpu_share,
                memory_allocation=memory_share,
                storage_allocation=storage_share,
                code_path=task.code_path,
                container_type="python",
                input_data={"fragment_index": i, "total_fragments": fragment_count}
            )
            
            # Add dependencies based on the dependency graph
            for j in range(fragment_count):
                if dependency_matrix[i][j]:
                    if j < len(fragments):  # Ensure the dependency has been created
                        fragment.dependencies.append(fragments[j].id)
            
            fragments.append(fragment)
        
        return fragments
    
    def _create_dependency_graph(self, fragment_count: int) -> List[List[bool]]:
        """
        Create a dependency graph for compute-optimized fragmentation.
        
        Args:
            fragment_count: Number of fragments
            
        Returns:
            2D adjacency matrix representing dependencies (True if i depends on j)
        """
        # Initialize with no dependencies
        matrix = [[False for _ in range(fragment_count)] for _ in range(fragment_count)]
        
        # Create a DAG-like dependency structure for better parallelization
        # Each fragment depends on ~30% of previous fragments
        for i in range(1, fragment_count):
            # Each node depends on roughly log2(i) previous nodes
            dependency_count = max(1, int(np.log2(i + 1)))
            # Select random dependencies from previous fragments
            dependencies = random.sample(range(i), min(dependency_count, i))
            for j in dependencies:
                matrix[i][j] = True
        
        return matrix
    
    def _estimate_complexity(self, data: Dict[str, Any]) -> float:
        """
        Estimate computational complexity of a workload.
        
        Args:
            data: Workload data
            
        Returns:
            Complexity score (0.1-10.0)
        """
        nested_level = self._get_max_nesting(data)
        operation_count = self._count_potential_operations(data)
        data_variety = len(set(str(type(v)) for v in data.values()))
        
        return max(0.1, min(10.0, (nested_level * 0.5 + operation_count * 0.3 + data_variety * 0.2)))
    
    def _get_max_nesting(self, data: Dict[str, Any], current_level: int = 0) -> int:
        """
        Get the maximum nesting level in a dictionary.
        
        Args:
            data: Dictionary to analyze
            current_level: Current nesting level
            
        Returns:
            Maximum nesting level
        """
        max_level = current_level
        
        for value in data.values():
            if isinstance(value, dict):
                level = self._get_max_nesting(value, current_level + 1)
                max_level = max(max_level, level)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    if isinstance(item, dict):
                        level = self._get_max_nesting(item, current_level + 1)
                        max_level = max(max_level, level)
        
        return max_level
    
    def _count_potential_operations(self, data: Dict[str, Any]) -> int:
        """
        Estimate the number of operations based on workload structure.
        
        Args:
            data: Workload data
            
        Returns:
            Estimated operation count
        """
        # Simple estimation - more sophisticated in real implementation
        operation_count = len(data)
        
        for value in data.values():
            if isinstance(value, (list, dict)):
                operation_count += len(value)
        
        return operation_count
    
    def _divide_workload_data(self, 
                             workload_data: Dict[str, Any], 
                             fragment_index: int, 
                             total_fragments: int) -> Dict[str, Any]:
        """
        Divide workload data into fragment-specific portions.
        
        Args:
            workload_data: Complete workload data
            fragment_index: Index of the current fragment
            total_fragments: Total number of fragments
            
        Returns:
            Data specific to this fragment
        """
        fragment_data = {"fragment_index": fragment_index, "total_fragments": total_fragments}
        
        # Different strategies for different data types
        if "transactions" in workload_data and isinstance(workload_data["transactions"], list):
            # Distribute transactions across fragments
            transactions = workload_data["transactions"]
            start_idx = (fragment_index * len(transactions)) // total_fragments
            end_idx = ((fragment_index + 1) * len(transactions)) // total_fragments
            fragment_data["transactions"] = transactions[start_idx:end_idx]
        
        elif "blocks" in workload_data and isinstance(workload_data["blocks"], list):
            # Distribute blocks across fragments
            blocks = workload_data["blocks"]
            start_idx = (fragment_index * len(blocks)) // total_fragments
            end_idx = ((fragment_index + 1) * len(blocks)) // total_fragments
            fragment_data["blocks"] = blocks[start_idx:end_idx]
        
        else:
            # For non-list data, copy the data with fragment metadata
            fragment_data.update(workload_data)
            fragment_data["is_full_data"] = False
        
        return fragment_data
    
    def _calculate_fragment_priority(self, 
                                    fragment_index: int, 
                                    total_fragments: int, 
                                    workload_data: Dict[str, Any]) -> int:
        """
        Calculate priority for a fragment (1-10).
        
        Args:
            fragment_index: Index of fragment
            total_fragments: Total number of fragments
            workload_data: Original workload data
            
        Returns:
            Priority level (1-10)
        """
        base_priority = workload_data.get("priority", 5)
        
        # First fragments often have higher priority in blockchain workloads
        position_factor = (total_fragments - fragment_index) / total_fragments
        
        # Calculate priority - higher return value means higher priority
        return max(1, min(10, int(base_priority + position_factor * 3) - 1))
    
    def _determine_fragment_dependencies(self, 
                                       fragment_index: int, 
                                       total_fragments: int) -> List[FragmentId]:
        """
        Determine dependencies between fragments.
        
        Args:
            fragment_index: Index of current fragment
            total_fragments: Total number of fragments
            
        Returns:
            List of fragment IDs this fragment depends on
        """
        dependencies = []
        
        # Simple linear dependency chain - in real implementation, this would be more sophisticated
        if fragment_index > 0:
            dependencies.append(f"frag-{fragment_index-1}")
        
        return dependencies
    
    def _estimate_fragment_resources(self, fragment_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate resources needed for a fragment.
        
        Args:
            fragment_data: Fragment's workload data
            
        Returns:
            Resource estimates
        """
        # Simple resource estimation
        data_size = len(json.dumps(fragment_data))
        complexity = self._estimate_complexity(fragment_data)
        
        # Base resource estimates
        return {
            "cpu": 0.05 + (complexity * 0.095),  # 0.05-1.0 CPU cores
            "memory": 50 + (data_size / 20),     # 50-1000+ MB
            "storage": 10 + (data_size / 100),   # 10-100+ MB
            "network": 5 + (data_size / 200)     # 5-50+ MB/s
        }
    
    def _estimate_execution_time(self, 
                              fragment_data: Dict[str, Any], 
                              resources: Dict[str, float]) -> float:
        """
        Estimate execution time in seconds for a fragment based on its data and 
        allocated resources.
        
        Args:
            fragment_data: Fragment's workload data
            resources: Allocated resources
            
        Returns:
            Estimated execution time in seconds
        """
        # Base execution time estimate
        data_size = len(json.dumps(fragment_data))
        complexity = self._estimate_complexity(fragment_data)
        
        # Calculate raw computation time
        base_time = 0.5 + (complexity * 2) + (data_size / 10000)
        
        # Adjust based on available resources
        cpu_factor = 1.0 / max(0.1, resources.get("cpu", 0.5))
        memory_factor = 1.0 + max(0, (500 - resources.get("memory", 500)) / 1000)
        
        # Apply resource factors - fewer resources means longer execution time
        execution_time = base_time * cpu_factor * memory_factor
        
        # Apply randomness to account for unpredictable factors
        # In a real system, this would be based on historical performance data
        randomness = random.uniform(0.9, 1.1)
        
        # Apply safety factor from config
        safety_factor = self.config.get("execution_time_safety_factor", 1.2)
        
        return execution_time * randomness * safety_factor
        
    def _minimal_fragmentation(self, task: Task) -> List[Fragment]:
        """
        Create minimal fragmentation for simple tasks.
        
        Args:
            task: The task to fragment
            
        Returns:
            Minimal list of fragments (typically 1-2)
        """
        # For simple tasks, just use 1-2 fragments
        fragment_count = 1
        if task.resource_requirements.get("cpu_cores", 0.5) > 1 or \
           task.resource_requirements.get("memory_mb", 200) > 500:
            fragment_count = 2
            
        fragments = []
        total_cpu = task.resource_requirements.get("cpu_cores", 0.5)
        total_memory = task.resource_requirements.get("memory_mb", 200)
        total_storage = task.resource_requirements.get("storage_mb", 500)
        
        for i in range(fragment_count):
            fragment_id = f"{task.id}_minimal_{self.fragment_counter}"
            self.fragment_counter += 1
            
            # Simple allocation - divide resources
            cpu_share = total_cpu / fragment_count
            memory_share = int(total_memory / fragment_count)
            storage_share = int(total_storage / fragment_count)
            
            fragment = Fragment(
                id=fragment_id,
                task_id=task.id,
                cpu_allocation=cpu_share,
                memory_allocation=memory_share,
                storage_allocation=storage_share,
                code_path=task.code_path,
                container_type="python",
                input_data={"fragment_index": i, "total_fragments": fragment_count}
            )
            
            # Simple linear dependency for minimal fragmentation
            if i > 0:
                fragment.dependencies.append(fragments[i-1].id)
                
            fragments.append(fragment)
            
        return fragments
        
    def _auto_select_strategy(self, task: Task) -> List[Fragment]:
        """
        Automatically select the best fragmentation strategy based on task properties.
        
        Args:
            task: The task to fragment
            
        Returns:
            List of fragments using the best strategy
        """
        # Analyze task to determine optimal strategy
        cpu_requirement = task.resource_requirements.get("cpu_cores", 0.5)
        memory_requirement = task.resource_requirements.get("memory_mb", 200)
        
        # Decide strategy based on resource requirements
        if memory_requirement > 1000 and cpu_requirement < 2:
            logger.info(f"Auto-selecting memory_optimized strategy for task {task.id}")
            return self._memory_optimized_fragmentation(task)
        elif cpu_requirement > 2 and memory_requirement < 1000:
            logger.info(f"Auto-selecting compute_optimized strategy for task {task.id}")
            return self._compute_optimized_fragmentation(task)
        elif cpu_requirement < 1 and memory_requirement < 500:
            logger.info(f"Auto-selecting minimal strategy for task {task.id}")
            return self._minimal_fragmentation(task)
        else:
            logger.info(f"Auto-selecting balanced strategy for task {task.id}")
            return self._balanced_fragmentation(task)
            
    def _update_task_history(self, task: Task) -> None:
        """
        Update task history for future optimization.
        
        Args:
            task: The completed task
        """
        task_type = task.name.split('_')[0] if '_' in task.name else task.name
        
        if task_type not in self.task_history:
            self.task_history[task_type] = []
            
        # Store task information for future reference
        self.task_history[task_type].append({
            "id": task.id,
            "resource_requirements": task.resource_requirements,
            "fragment_count": len(task.fragments),
            "estimated_time": task.estimated_execution_time,
            "actual_time": (task.completed_at or time.time()) - (task.started_at or time.time()),
            "strategy": task.input_data.get("strategy", "default")
        })
        
        # Keep history to a reasonable size
        if len(self.task_history[task_type]) > 100:
            self.task_history[task_type] = self.task_history[task_type][-100:]
            
        # Update execution time history
        if task.completed_at and task.started_at:
            actual_time = task.completed_at - task.started_at
            if task_type not in self.historical_execution_data:
                self.historical_execution_data[task_type] = []
            self.historical_execution_data[task_type].append(actual_time)
            
            # Keep execution history to a reasonable size
            if len(self.historical_execution_data[task_type]) > 100:
                self.historical_execution_data[task_type] = self.historical_execution_data[task_type][-100:]


    
    This class provides real-time monitoring of system resources (CPU, memory, storage)
    and helps make intelligent container scheduling and scaling decisions.
    """
    
    def __init__(self, update_interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            update_interval: Interval between resource updates in seconds
        """
        self.update_interval: float = update_interval
        self.resources: ResourceMetrics = {}
        self.resource_history: Dict[str, List[Tuple[float, float]]] = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": []
        }
        self.running: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock: threading.Lock = threading.Lock()
        self.resource_alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Alert thresholds
        self.thresholds: Dict[str, Dict[str, float]] = {
            "cpu": {"warning": 80.0, "critical": 95.0},
            "memory": {"warning": 75.0, "critical": 90.0},
            "disk": {"warning": 85.0, "critical": 95.0}
        }
        
        logger.info("ResourceMonitor initialized with update interval of %.1f seconds", update_interval)
        
    def start(self) -> None:
        """Start the resource monitoring thread."""
        if self.running:
            logger.warning("ResourceMonitor is already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("ResourceMonitor started")
        
    def stop(self) -> None:
        """Stop the resource monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
        logger.info("ResourceMonitor stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects and records system metrics."""
        while self.running:
            try:
                self._update_resource_metrics()
                self._check_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error("Error in resource monitoring loop: %s", str(e))
                time.sleep(max(1.0, self.update_interval / 2))
                
    def _update_resource_metrics(self) -> None:
        """Collect and update resource metrics."""
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        metrics = {
            "timestamp": current_time,
            "cpu": {
                "percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0
            },
            "memory": {
                "percent": memory.percent,
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024),
                "swap_percent": swap.percent
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": disk.used / (1024 * 1024

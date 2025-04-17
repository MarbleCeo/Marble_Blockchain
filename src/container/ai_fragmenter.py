#!/usr/bin/env python3
"""
AI Fragmenter for Container Workloads

This module provides a mechanism to analyze AI workloads and split them into optimized
container fragments for better resource utilization and parallel execution.
"""

import os
import time
import json
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from dataclasses import dataclass
from enum import Enum, auto

# Configure logging
logger = logging.getLogger("AIFragmenter")

@dataclass
class Fragment:
    """Specification for a code fragment to be executed in a container"""
    id: str
    task_id: str
    cpu_allocation: float
    memory_allocation: int
    storage_allocation: int
    code_path: str
    container_type: str
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class Task:
    """Specification for a task to be fragmented and executed"""
    id: str
    name: str
    code_path: str
    resource_requirements: Dict[str, Any]
    input_data: Dict[str, Any]
    priority: int = 5
    dependencies: List[str] = None
    fragments: List[Fragment] = None
    estimated_execution_time: float = 0.0
    started_at: float = None
    completed_at: float = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.fragments is None:
            self.fragments = []

class AIFragmenter:
    """
    Fragmenter for AI workloads that splits tasks into optimized container units.
    
    This class analyzes AI tasks and fragments them into optimally sized container units
    based on resource requirements, dependencies, and execution patterns.
    """
    
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
                


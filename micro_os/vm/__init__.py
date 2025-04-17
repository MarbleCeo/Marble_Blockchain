from .controller_native import VMController

__all__ = ['VMController']

#!/usr/bin/env python3
"""
Virtual Machine (VM) Module for Micro OS

This module provides a comprehensive virtual machine environment that integrates
with the RegenerativeDeepSeekAI system. It includes VM emulation capabilities,
lens refraction logic for VM circuits, and performance monitoring.

Classes:
    VMEnvironment: Main class for VM environment management
    VMState: Enum for VM states
    CircuitType: Enum for VM circuit types
    PerformanceMetrics: Data class for VM performance metrics
    CircuitDefinition: Data class for VM circuit definitions
"""

# Import and export the main classes and enums from vm_environment.py
from micro_os.vm.vm_environment import (
    # Enums
    VMState,
    CircuitType,
    
    # Data classes
    PerformanceMetrics,
    CircuitDefinition,
    
    # Main class
    VMEnvironment,
)

# Define what should be available on import
__all__ = [
    'VMEnvironment',
    'VMState',
    'CircuitType',
    'PerformanceMetrics',
    'CircuitDefinition',
]


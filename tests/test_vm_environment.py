#!/usr/bin/env python3
"""
Unit tests for VMEnvironment class

This module contains test cases for the VMEnvironment class, including tests for:
1. VM initialization
2. Starting and stopping the VM
3. Circuit management
4. Performance monitoring
5. Health checks
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime
import threading

from micro_os.vm.vm_environment import VMEnvironment, VMState, CircuitType, PerformanceMetrics


class TestVMEnvironment(unittest.TestCase):
    """Test cases for the VMEnvironment class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a VM environment with a specific configuration for testing
        self.vm_id = "test-vm-001"
        self.vm_name = "Test VM"
        self.resources = {
            "cpu_cores": 2.0,
            "memory_mb": 1024.0,
            "disk_space_mb": 2048.0,
            "network_bandwidth_mbps": 20.0
        }
        self.ai_config = {
            "enabled": True,
            "model_complexity": 0.7,
            "optimization_level": 0.8,
            "learning_rate": 0.05
        }
        
        # Create the VM environment
        with patch('threading.Thread'):  # Prevent monitoring thread from starting
            self.vm = VMEnvironment(
                vm_id=self.vm_id,
                name=self.vm_name,
                resources=self.resources,
                ai_integration_config=self.ai_config
            )
        
        # Mock the monitoring thread to prevent actual performance monitoring
        self.vm._monitoring_thread = MagicMock()
        self.vm._monitoring_thread.is_alive.return_value = True

    def tearDown(self):
        """Clean up test environment after each test."""
        if hasattr(self, 'vm') and self.vm is not None:
            self.vm.cleanup()
            self.vm = None

    def test_vm_initialization(self):
        """Test that VM environment is properly initialized."""
        # Verify VM properties
        self.assertEqual(self.vm.vm_id, self.vm_id)
        self.assertEqual(self.vm.name, self.vm_name)
        self.assertEqual(self.vm.state, VMState.RUNNING)  # VM should be in RUNNING state after initialization
        self.assertEqual(self.vm.resources, self.resources)
        self.assertEqual(self.vm.ai_integration, self.ai_config)
        
        # Verify initial circuits were created
        self.assertGreater(len(self.vm.circuits), 0)
        
        # Verify circuit types
        circuit_types = set(circuit.circuit_type for circuit in self.vm.circuits.values())
        expected_types = {CircuitType.COMPUTATIONAL, CircuitType.MEMORY, 
                          CircuitType.NETWORKING, CircuitType.IO}
        self.assertTrue(all(ctype in circuit_types for ctype in expected_types))

    def test_vm_initialization_with_defaults(self):
        """Test VM initialization with default values."""
        with patch('threading.Thread'):
            default_vm = VMEnvironment(vm_id="default-vm")
        
        # Verify default name format
        self.assertEqual(default_vm.name, f"vm-default-vm")
        
        # Verify default resources were applied
        self.assertTrue("cpu_cores" in default_vm.resources)
        self.assertTrue("memory_mb" in default_vm.resources)
        self.assertTrue("disk_space_mb" in default_vm.resources)
        self.assertTrue("network_bandwidth_mbps" in default_vm.resources)
        
        # Verify default AI integration
        self.assertTrue(default_vm.ai_integration["enabled"])
        
        # Clean up
        default_vm.cleanup()

    def test_start_stop_vm(self):
        """Test starting and stopping the VM."""
        # VM is already started after initialization, so stop it first
        self.assertEqual(self.vm.state, VMState.RUNNING)
        self.assertTrue(self.vm.stop())
        self.assertEqual(self.vm.state, VMState.STOPPED)
        
        # Test starting a stopped VM
        self.assertTrue(self.vm.start())
        self.assertEqual(self.vm.state, VMState.RUNNING)
        
        # Test stopping a running VM
        self.assertTrue(self.vm.stop())
        self.assertEqual(self.vm.state, VMState.STOPPED)
        
        # Test trying to stop an already stopped VM
        self.assertTrue(self.vm.stop())  # Should return True, as it's already in the desired state
        self.assertEqual(self.vm.state, VMState.STOPPED)

    def test_pause_resume_vm(self):
        """Test pausing and resuming the VM."""
        # Ensure VM is running
        self.vm.state = VMState.RUNNING
        
        # Test pausing a running VM
        self.assertTrue(self.vm.pause())
        self.assertEqual(self.vm.state, VMState.PAUSED)
        
        # Test resuming a paused VM
        self.assertTrue(self.vm.resume())
        self.assertEqual(self.vm.state, VMState.RUNNING)
        
        # Test trying to pause a non-running VM
        self.vm.state = VMState.STOPPED
        self.assertFalse(self.vm.pause())
        self.assertEqual(self.vm.state, VMState.STOPPED)
        
        # Test trying to resume a non-paused VM
        self.assertFalse(self.vm.resume())
        self.assertEqual(self.vm.state, VMState.STOPPED)

    def test_reset_vm(self):
        """Test resetting the VM."""
        # Mock the initialize_environment method to avoid actual initialization
        self.vm._initialize_environment = MagicMock()
        
        # Add some test data to clear during reset
        self.vm.metrics = [PerformanceMetrics() for _ in range(5)]
        initial_circuit_count = len(self.vm.circuits)
        
        # Reset the VM
        self.assertTrue(self.vm.reset())
        
        # Verify reset cleared metrics and circuits
        self.assertEqual(len(self.vm.metrics), 0)
        self.assertEqual(len(self.vm.circuits), 0)
        
        # Verify initialize_environment was called
        self.vm._initialize_environment.assert_called_once()

    def test_add_circuit(self):
        """Test adding a circuit to the VM."""
        # Ensure VM is running
        self.vm.state = VMState.RUNNING
        
        # Get initial circuit count
        initial_count = len(self.vm.circuits)
        
        # Add a new circuit
        circuit_id = self.vm.add_circuit(CircuitType.BLOCKCHAIN)
        
        # Verify circuit was added
        self.assertIsNotNone(circuit_id)
        self.assertEqual(len(self.vm.circuits), initial_count + 1)
        self.assertIn(circuit_id, self.vm.circuits)
        self.assertEqual(self.vm.circuits[circuit_id].circuit_type, CircuitType.BLOCKCHAIN)
        
        # Test adding a circuit when VM is not running
        self.vm.state = VMState.STOPPED
        self.assertIsNone(self.vm.add_circuit(CircuitType.SECURITY))
        
        # Verify no new circuit was added
        self.assertEqual(len(self.vm.circuits), initial_count + 1)

    def test_remove_circuit(self):
        """Test removing a circuit from the VM."""
        # Ensure VM is running
        self.vm.state = VMState.RUNNING
        
        # Add a circuit to remove
        circuit_id = self.vm.add_circuit(CircuitType.SECURITY)
        self.assertIsNotNone(circuit_id)
        self.assertIn(circuit_id, self.vm.circuits)
        
        # Remove the circuit
        self.assertTrue(self.vm.remove_circuit(circuit_id))
        
        # Verify circuit was removed
        self.assertNotIn(circuit_id, self.vm.circuits)
        
        # Test removing a non-existent circuit
        self.assertFalse(self.vm.remove_circuit("non-existent-circuit"))
        
        # Test removing a circuit when VM is not running
        self.vm.state = VMState.STOPPED
        circuit_id = next(iter(self.vm.circuits.keys()))
        self.assertFalse(self.vm.remove_circuit(circuit_id))

    def test_circuit_generation(self):
        """Test circuit generation via lens refraction."""
        # Test generating various circuit types
        for circuit_type in CircuitType:
            circuit = self.vm._refract_lens_for_circuit(circuit_type)
            
            # Verify basic circuit properties
            self.assertEqual(circuit.circuit_type, circuit_type)
            self.assertGreater(circuit.complexity, 0)
            self.assertGreater(len(circuit.operations), 0)
            self.assertIn("security_level", vars(circuit))
            
            # Verify resource requirements are appropriate for type
            if circuit_type == CircuitType.COMPUTATIONAL:
                self.assertIn("cpu", circuit.resource_requirements)
            elif circuit_type == CircuitType.MEMORY:
                self.assertIn("memory", circuit.resource_requirements)
            elif circuit_type == CircuitType.NETWORKING:
                self.assertIn("network", circuit.resource_requirements)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    @patch('psutil.getloadavg')
    @patch('psutil.cpu_count')
    def test_performance_metrics_collection(self, mock_cpu_count, mock_getloadavg, 
                                          mock_net_io, mock_disk_io, mock_vmem, mock_cpu_percent):
        """Test collection of performance metrics."""
        # Mock psutil responses
        mock_cpu_count.return_value = 4
        mock_cpu_percent.return_value = 25.0
        
        mem_mock = MagicMock()
        mem_mock.total = 8 * 1024 * 1024 * 1024  # 8 GB
        mock_vmem.return_value = mem_mock
        
        disk_mock = MagicMock()
        disk_mock.read_bytes = 1024 * 1024  # 1 MB
        disk_mock.write_bytes = 2 * 1024 * 1024  # 2 MB
        mock_disk_io.return_value = disk_mock
        
        net_mock = MagicMock()
        net_mock.bytes_sent = 512 * 1024  # 512 KB
        net_mock.bytes_recv = 1024 * 1024  # 1 MB
        mock_net_io.return_value = net_mock
        
        mock_getloadavg.return_value = [1.0, 1.2, 1.5]
        
        # Collect metrics
        metrics = self.vm._collect_performance_metrics()
        
        # Verify metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreaterEqual(metrics.cpu_usage, 0)
        self.assertGreaterEqual(metrics.memory_usage, 0)
        self.assertEqual(len(metrics.disk_io), 2)
        self.assertEqual(len(metrics.network_io), 2)
        self.assertGreaterEqual(metrics.transaction_throughput, 0)
        self.assertGreaterEqual(metrics.response_time, 0)
        self.assertIsInstance(metrics.timestamp, datetime)

    @patch.object(VMEnvironment, '_collect_performance_metrics')
    @patch.object(VMEnvironment, '_try_optimize_resources')
    def test_health_check(self, mock_optimize, mock_collect_metrics):
        """Test VM health checking functionality."""
        # Set up test metrics with concerning values
        high_cpu_metrics = [
            PerformanceMetrics(cpu_usage=95.0, memory_usage=50.0, response_time=5.0),
            PerformanceMetrics(cpu_usage=92.0, memory_usage=55.0, response_time=6.0)
        ]
        
        high_memory_metrics = [
            PerformanceMetrics(cpu_usage=50.0, memory_usage=90.0, response_time=5.0),
            PerformanceMetrics(cpu_usage=55.0, memory_usage=92.0, response_time=6.0)
        ]
        
        high_response_metrics = [
            PerformanceMetrics(cpu_usage=50.0, memory_usage=50.0, response_time=25.0),
            PerformanceMetrics(cpu_usage=55.0, memory_usage=55.0, response_time=30.0)
        ]
        
        # Test high CPU usage
        self.vm.metrics = high_cpu_metrics
        self.vm._check_health()
        mock_optimize.assert_called_with("cpu")
        mock_optimize.reset_mock()
        
        # Test high memory usage
        self.vm.metrics = high_memory_metrics
        self.vm._check_health()
        mock_optimize.assert_called_with("memory")
        mock_optimize.reset_mock()
        
        # Test high response time (should log but not optimize)
        self.vm.metrics = high_response_metrics
        self.vm._check_health()
        mock_optimize.assert_not_called()
        
        # Test no metrics
        self.vm.metrics = []
        self.vm._check_health()
        mock_optimize.assert_not_called()

    def test_get_status(self):
        """Test retrieving VM status information."""
        # Add a test metric
        test_metric = PerformanceMetrics(
            cpu_usage=25.0,
            memory_usage=40.0,
            response_time=10.0
        )
        self.vm.metrics = [test_metric]
        
        # Get status
        status = self.vm.get_status()
        
        # Verify status content
        self.assertEqual(status["vm_id"], self.vm_id)
        self.assertEqual(status["name"], self.vm_name)
        self.assertEqual(status["state"], self.vm.state.value)
        self.assertGreaterEqual(status["uptime"], 0)
        self.assertEqual(status["circuit_count"], len(self.vm.circuits))
        self.assertEqual(status["resources"], self.resources)
        self.assertEqual(status["current_metrics"]["cpu"], test_metric.cpu_usage)
        self.assertEqual(status["current_metrics"]["memory"], test_metric.memory_usage)
        self.assertEqual(status["current_metrics"]["response_time"], test_metric.response_time)
        self.assertEqual(status["ai_integration_enabled"], self.ai_config["enabled"])

    def test_resource_optimization(self):
        """Test resource optimization capabilities."""
        # Add some test circuits
        comp_circuit = self.vm._refract_lens_for_circuit(CircuitType.COMPUTATIONAL)
        comp_circuit.complexity = 1.0
        comp_circuit.resource_requirements = {"cpu": 0.8, "memory": 0.2}
        
        mem_circuit = self.vm._refract_lens_for_circuit(CircuitType.MEMORY)
        mem_circuit.complexity = 0.9
        mem_circuit.resource_requirements = {"cpu": 0.2, "memory": 0.7}
        
        net_circuit = self.vm._refract_lens_for_circuit(CircuitType.NETWORKING)
        net_circuit.complexity = 0.7
        net_circuit.resource_requirements = {"cpu": 0.3, "memory": 0.4, "network": 0.8}
        
        self.vm.circuits = {
            "circuit-comp-1": comp_circuit,
            "circuit-mem-1": mem_circuit,
            "circuit-net-1": net_circuit
        }
        
        # Test CPU optimization
        with patch.object(self.vm, '_adjust_circuit_complexity') as mock_adjust:
            self.vm._try_optimize_resources("cpu")
            # Should optimize the computational circuit first (highest CPU usage)
            mock_adjust.assert_called_with("circuit-comp-1", 0.9)
        
        # Test memory optimization
        with patch.object(self.vm, '_adjust_circuit_complexity') as mock_adjust:
            self.vm._try_optimize_resources("memory")
            # Should optimize the memory circuit first (highest memory usage)
            mock_adjust.assert_called_with("circuit-mem-1", 0.9)
        
        # Test with disabled AI
        self.vm.ai_integration["enabled"] = False
        with patch.object(self.vm, '_adjust_circuit_complexity') as mock_adjust:
            self.vm._try_optimize_resources("cpu")
            # Should not attempt optimization with AI disabled
            mock_adjust.assert_not_called()

    def test_circuit_operations(self):
        """Test circuit operations execution."""
        # Get a circuit
        circuit_id = next(iter(self.vm.circuits.keys()))
        circuit = self.vm.circuits[circuit_id]
        
        # Ensure VM is running
        self.vm.state = VMState.RUNNING
        
        # Test execute_operation
        with patch.object(self.vm, '_execute_operation_internal') as mock_execute:
            mock_execute.return_value = {"status": "success", "result": "test result"}
            
            result = self.vm.execute_operation(circuit_id, "test_operation", {"param": "value"})
            
            # Verify operation was executed
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["result"], "test result")
            mock_execute.assert_called_with(circuit, "test_operation", {"param": "value"})
        
        # Test execute_operation with invalid circuit
        result = self.vm.execute_operation("invalid-circuit", "test_operation", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("circuit not found", result["message"].lower())
        
        # Test execute_operation when VM is not running
        self.vm.state = VMState.STOPPED
        result = self.vm.execute_operation(circuit_id, "test_operation", {})
        self.assertEqual(result["status"], "error")
        self.assertIn("not running", result["message"].lower())

    def test_ai_integration(self):
        """Test AI integration functionality."""
        # Ensure VM has AI integration enabled
        self.vm.ai_integration["enabled"] = True
        
        # Test AI-based circuit complexity adjustment
        with patch('micro_os.vm.vm_environment.VMEnvironment._execute_ai_analysis') as mock_ai:
            mock_ai.return_value = 0.7
            
            circuit_id = next(iter(self.vm.circuits.keys()))
            circuit = self.vm.circuits[circuit_id]
            original_complexity = circuit.complexity
            
            # Adjust complexity
            self.vm._adjust_circuit_complexity(circuit_id, 0.8)
            
            # Verify complexity was adjusted
            self.assertNotEqual(circuit.complexity, original_complexity)
            mock_ai.assert_called_once()
        
        # Test AI analysis with disabled AI
        self.vm.ai_integration["enabled"] = False
        with patch('micro_os.vm.vm_environment.VMEnvironment._execute_ai_analysis') as mock_ai:
            self.vm._execute_ai_analysis({})
            mock_ai.assert_not_called()

    def test_cleanup(self):
        """Test cleanup method properly releases resources."""
        # Ensure VM is running with a monitoring thread
        self.vm.state = VMState.RUNNING
        mock_thread = MagicMock()
        self.vm._monitoring_thread = mock_thread
        self.vm._monitoring_active = True
        
        # Mock the stop method
        with patch.object(self.vm, 'stop') as mock_stop:
            self.vm.cleanup()
            
            # Verify VM was stopped
            mock_stop.assert_called_once()
            
            # Verify monitoring was terminated
            self.assertFalse(self.vm._monitoring_active)
            
            # Verify resources were released
            self.assertEqual(len(self.vm.circuits), 0)
            self.assertEqual(len(self.vm.metrics), 0)
            
            # Verify the VM is now in a terminated state
            self.assertEqual(self.vm.state, VMState.TERMINATED)


if __name__ == "__main__":
    unittest.main()

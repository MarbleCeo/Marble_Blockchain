import pytest
import sys
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Import the terminal module
from micro_os.cli.terminal import (
    MicroOSTerminal,
    CommandParser,
    VMCommandHandler,
    ContainerCommandHandler,
    CircuitCommandHandler,
    MonitoringCommandHandler,
    CommandNotFoundError,
    InvalidArgumentError
)


@pytest.fixture
def terminal():
    """Fixture to create a fresh terminal instance for each test."""
    terminal = MicroOSTerminal()
    return terminal


@pytest.fixture
def mock_vm_handler():
    """Fixture to create a mock VM command handler."""
    with patch('micro_os.cli.terminal.VMCommandHandler') as mock:
        handler = mock.return_value
        yield handler


@pytest.fixture
def mock_container_handler():
    """Fixture to create a mock container command handler."""
    with patch('micro_os.cli.terminal.ContainerCommandHandler') as mock:
        handler = mock.return_value
        yield handler


@pytest.fixture
def mock_circuit_handler():
    """Fixture to create a mock circuit command handler."""
    with patch('micro_os.cli.terminal.CircuitCommandHandler') as mock:
        handler = mock.return_value
        yield handler


@pytest.fixture
def mock_monitoring_handler():
    """Fixture to create a mock monitoring command handler."""
    with patch('micro_os.cli.terminal.MonitoringCommandHandler') as mock:
        handler = mock.return_value
        yield handler


@pytest.fixture
def captured_output():
    """Fixture to capture stdout for testing output."""
    captured_output = StringIO()
    sys.stdout = captured_output
    yield captured_output
    sys.stdout = sys.__stdout__


class TestCommandParser:
    """Tests for the CommandParser class."""

    def test_parse_simple_command(self):
        """Test parsing a simple command without arguments."""
        parser = CommandParser()
        command, args = parser.parse("vm list")
        assert command == "vm list"
        assert args == {}

    def test_parse_command_with_arguments(self):
        """Test parsing a command with arguments."""
        parser = CommandParser()
        command, args = parser.parse("vm create --name test_vm --memory 1024 --cpu 2")
        assert command == "vm create"
        assert args == {"name": "test_vm", "memory": "1024", "cpu": "2"}

    def test_parse_command_with_positional_arguments(self):
        """Test parsing a command with positional arguments."""
        parser = CommandParser()
        command, args = parser.parse("vm start test_vm")
        assert command == "vm start"
        assert args == {"vm_name": "test_vm"}

    def test_parse_invalid_command(self):
        """Test parsing an invalid command format raises appropriate error."""
        parser = CommandParser()
        with pytest.raises(InvalidArgumentError):
            parser.parse("vm create --name")  # Missing value for argument

    def test_parse_empty_command(self):
        """Test parsing an empty command."""
        parser = CommandParser()
        command, args = parser.parse("")
        assert command == ""
        assert args == {}


class TestVMCommandHandler:
    """Tests for the VMCommandHandler class."""

    def test_create_vm(self, mock_vm_handler):
        """Test creating a VM with valid parameters."""
        mock_vm_handler.create.return_value = True
        
        handler = VMCommandHandler()
        result = handler.create(name="test_vm", memory="1024", cpu="2")
        
        assert result is True
        mock_vm_handler.create.assert_called_once_with(name="test_vm", memory="1024", cpu="2")

    def test_start_vm(self, mock_vm_handler):
        """Test starting a VM."""
        mock_vm_handler.start.return_value = True
        
        handler = VMCommandHandler()
        result = handler.start(vm_name="test_vm")
        
        assert result is True
        mock_vm_handler.start.assert_called_once_with(vm_name="test_vm")

    def test_stop_vm(self, mock_vm_handler):
        """Test stopping a VM."""
        mock_vm_handler.stop.return_value = True
        
        handler = VMCommandHandler()
        result = handler.stop(vm_name="test_vm")
        
        assert result is True
        mock_vm_handler.stop.assert_called_once_with(vm_name="test_vm")

    def test_pause_vm(self, mock_vm_handler):
        """Test pausing a VM."""
        mock_vm_handler.pause.return_value = True
        
        handler = VMCommandHandler()
        result = handler.pause(vm_name="test_vm")
        
        assert result is True
        mock_vm_handler.pause.assert_called_once_with(vm_name="test_vm")

    def test_resume_vm(self, mock_vm_handler):
        """Test resuming a VM."""
        mock_vm_handler.resume.return_value = True
        
        handler = VMCommandHandler()
        result = handler.resume(vm_name="test_vm")
        
        assert result is True
        mock_vm_handler.resume.assert_called_once_with(vm_name="test_vm")

    def test_inspect_vm(self, mock_vm_handler):
        """Test inspecting a VM."""
        mock_vm_handler.inspect.return_value = {"name": "test_vm", "status": "running"}
        
        handler = VMCommandHandler()
        result = handler.inspect(vm_name="test_vm")
        
        assert result == {"name": "test_vm", "status": "running"}
        mock_vm_handler.inspect.assert_called_once_with(vm_name="test_vm")

    def test_list_vms(self, mock_vm_handler):
        """Test listing all VMs."""
        mock_vm_handler.list.return_value = [
            {"name": "vm1", "status": "running"},
            {"name": "vm2", "status": "stopped"}
        ]
        
        handler = VMCommandHandler()
        result = handler.list()
        
        assert len(result) == 2
        assert result[0]["name"] == "vm1"
        assert result[1]["status"] == "stopped"
        mock_vm_handler.list.assert_called_once()

    def test_vm_with_invalid_parameters(self, mock_vm_handler):
        """Test VM operation with invalid parameters."""
        mock_vm_handler.create.side_effect = InvalidArgumentError("Missing required parameters")
        
        handler = VMCommandHandler()
        with pytest.raises(InvalidArgumentError):
            handler.create()
        
        mock_vm_handler.create.assert_called_once_with()


class TestContainerCommandHandler:
    """Tests for the ContainerCommandHandler class."""

    def test_create_container(self, mock_container_handler):
        """Test creating a container with valid parameters."""
        mock_container_handler.create.return_value = True
        
        handler = ContainerCommandHandler()
        result = handler.create(name="test_container", image="test_image", vm="test_vm")
        
        assert result is True
        mock_container_handler.create.assert_called_once_with(
            name="test_container", image="test_image", vm="test_vm"
        )

    def test_start_container(self, mock_container_handler):
        """Test starting a container."""
        mock_container_handler.start.return_value = True
        
        handler = ContainerCommandHandler()
        result = handler.start(container_name="test_container")
        
        assert result is True
        mock_container_handler.start.assert_called_once_with(container_name="test_container")

    def test_stop_container(self, mock_container_handler):
        """Test stopping a container."""
        mock_container_handler.stop.return_value = True
        
        handler = ContainerCommandHandler()
        result = handler.stop(container_name="test_container")
        
        assert result is True
        mock_container_handler.stop.assert_called_once_with(container_name="test_container")

    def test_pause_container(self, mock_container_handler):
        """Test pausing a container."""
        mock_container_handler.pause.return_value = True
        
        handler = ContainerCommandHandler()
        result = handler.pause(container_name="test_container")
        
        assert result is True
        mock_container_handler.pause.assert_called_once_with(container_name="test_container")

    def test_resume_container(self, mock_container_handler):
        """Test resuming a container."""
        mock_container_handler.resume.return_value = True
        
        handler = ContainerCommandHandler()
        result = handler.resume(container_name="test_container")
        
        assert result is True
        mock_container_handler.resume.assert_called_once_with(container_name="test_container")

    def test_inspect_container(self, mock_container_handler):
        """Test inspecting a container."""
        mock_container_handler.inspect.return_value = {"name": "test_container", "status": "running"}
        
        handler = ContainerCommandHandler()
        result = handler.inspect(container_name="test_container")
        
        assert result == {"name": "test_container", "status": "running"}
        mock_container_handler.inspect.assert_called_once_with(container_name="test_container")

    def test_list_containers(self, mock_container_handler):
        """Test listing all containers."""
        mock_container_handler.list.return_value = [
            {"name": "container1", "status": "running"},
            {"name": "container2", "status": "stopped"}
        ]
        
        handler = ContainerCommandHandler()
        result = handler.list()
        
        assert len(result) == 2
        assert result[0]["name"] == "container1"
        assert result[1]["status"] == "stopped"
        mock_container_handler.list.assert_called_once()

    def test_container_with_invalid_parameters(self, mock_container_handler):
        """Test container operation with invalid parameters."""
        mock_container_handler.create.side_effect = InvalidArgumentError("Missing required parameters")
        
        handler = ContainerCommandHandler()
        with pytest.raises(InvalidArgumentError):
            handler.create()
        
        mock_container_handler.create.assert_called_once_with()


class TestCircuitCommandHandler:
    """Tests for the CircuitCommandHandler class."""

    def test_view_circuit(self, mock_circuit_handler):
        """Test viewing a circuit."""
        mock_circuit_handler.view.return_value = {"name": "test_circuit", "status": "active"}
        
        handler = CircuitCommandHandler()
        result = handler.view(circuit_name="test_circuit")
        
        assert result == {"name": "test_circuit", "status": "active"}
        mock_circuit_handler.view.assert_called_once_with(circuit_name="test_circuit")

    def test_create_circuit(self, mock_circuit_handler):
        """Test creating a circuit."""
        mock_circuit_handler.create.return_value = True
        
        handler = CircuitCommandHandler()
        result = handler.create(name="test_circuit", nodes=["node1", "node2"])
        
        assert result is True
        mock_circuit_handler.create.assert_called_once_with(name="test_circuit", nodes=["node1", "node2"])

    def test_monitor_circuit(self, mock_circuit_handler):
        """Test monitoring a circuit."""
        mock_circuit_handler.monitor.return_value = {"name": "test_circuit", "status": "healthy"}
        
        handler = CircuitCommandHandler()
        result = handler.monitor(circuit_name="test_circuit")
        
        assert result == {"name": "test_circuit", "status": "healthy"}
        mock_circuit_handler.monitor.assert_called_once_with(circuit_name="test_circuit")

    def test_list_circuits(self, mock_circuit_handler):
        """Test listing all circuits."""
        mock_circuit_handler.list.return_value = [
            {"name": "circuit1", "status": "active"},
            {"name": "circuit2", "status": "inactive"}
        ]
        
        handler = CircuitCommandHandler()
        result = handler.list()
        
        assert len(result) == 2
        assert result[0]["name"] == "circuit1"
        assert result[1]["status"] == "inactive"
        mock_circuit_handler.list.assert_called_once()


class TestMonitoringCommandHandler:
    """Tests for the MonitoringCommandHandler class."""

    def test_status(self, mock_monitoring_handler):
        """Test getting status."""
        mock_monitoring_handler.status.return_value = {"status": "healthy", "uptime": "10h"}
        
        handler = MonitoringCommandHandler()
        result = handler.status()
        
        assert result["status"] == "healthy"
        mock_monitoring_handler.status.assert_called_once()

    def test_performance(self, mock_monitoring_handler):
        """Test getting performance metrics."""
        mock_monitoring_handler.performance.return_value = {
            "cpu": "30%", "memory": "45%", "network": "10Mbps"
        }
        
        handler = MonitoringCommandHandler()
        result = handler.performance()
        
        assert result["cpu"] == "30%"
        assert result["memory"] == "45%"
        mock_monitoring_handler.performance.assert_called_once()


class TestTerminal:
    """Tests for the main Terminal class."""

    def test_execute_vm_command(self, terminal, mock_vm_handler):
        """Test executing a VM command."""
        with patch.object(terminal, 'vm_handler', mock_vm_handler):
            mock_vm_handler.list.return_value = [{"name": "vm1"}, {"name": "vm2"}]
            
            result = terminal.execute_command("vm list")
            
            assert result == [{"name": "vm1"}, {"name": "vm2"}]
            mock_vm_handler.list.assert_called_once()

    def test_execute_container_command(self, terminal, mock_container_handler):
        """Test executing a container command."""
        with patch.object(terminal, 'container_handler', mock_container_handler):
            mock_container_handler.list.return_value = [{"name": "container1"}, {"name": "container2"}]
            
            result = terminal.execute_command("container list")
            
            assert result == [{"name": "container1"}, {"name": "container2"}]
            mock_container_handler.list.assert_called_once()

    def test_execute_circuit_command(self, terminal, mock_circuit_handler):
        """Test executing a circuit command."""
        with patch.object(terminal, 'circuit_handler', mock_circuit_handler):
            mock_circuit_handler.list.return_value = [{"name": "circuit1"}, {"name": "circuit2"}]
            
            result = terminal.execute_command("circuit list")
            
            assert result == [{"name": "circuit1"}, {"name": "circuit2"}]
            mock_circuit_handler.list.assert_called_once()

    def test_execute_monitoring_command(self, terminal, mock_monitoring_handler):
        """Test executing a monitoring command."""
        


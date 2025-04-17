#!/usr/bin/env python3
"""
MicroOS Terminal CLI Interface

A UNIX-like command-line interface for managing MicroOS components including:
- Virtual Machine operations
- Container management
- Circuit creation and monitoring
- Performance monitoring

The terminal includes auto-completion, help documentation, and robust error handling.
"""

import cmd
import sys
import os
import argparse
import shlex
import readline
import traceback
from typing import List, Dict, Any, Optional, Callable

# Import MicroOS components
try:
    from micro_os.vm.vm_environment import VMEnvironment
    from micro_os.containers.container_manager import ContainerManager
    from micro_os.network.vm_protocol import NetworkProtocol
    from micro_os.circuits.circuit_manager import CircuitManager
    from micro_os.monitoring.health_monitor import HealthMonitor
except ImportError as e:
    print(f"Failed to import MicroOS modules: {e}")
    print("Some functionality may be limited.")


class CommandResult:
    """Represents the result of a command execution"""
    
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "ERROR"
        return f"[{status}] {self.message}"


class CommandHandler:
    """Base class for command handlers"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.subcommands: Dict[str, Dict[str, Any]] = {}
    
    def register_subcommand(self, name: str, handler: Callable, help_text: str, 
                           arguments: Optional[List[Dict[str, Any]]] = None):
        """Register a subcommand with its handler function"""
        self.subcommands[name] = {
            'handler': handler,
            'help': help_text,
            'arguments': arguments or []
        }
    
    def get_subcommands(self) -> List[str]:
        """Return a list of available subcommands"""
        return list(self.subcommands.keys())
    
    def get_help(self, subcommand: Optional[str] = None) -> str:
        """Get help text for this command or a specific subcommand"""
        if subcommand and subcommand in self.subcommands:
            cmd_info = self.subcommands[subcommand]
            help_text = f"{self.name} {subcommand} - {cmd_info['help']}\n\nUsage:\n  {self.name} {subcommand}"
            
            if cmd_info['arguments']:
                help_text += " "
                for arg in cmd_info['arguments']:
                    if arg.get('required', False):
                        help_text += f" <{arg['name']}>"
                    else:
                        help_text += f" [{arg['name']}]"
                
                help_text += "\n\nArguments:\n"
                for arg in cmd_info['arguments']:
                    arg_text = f"  {arg['name']}: {arg['help']}"
                    if 'default' in arg:
                        arg_text += f" (default: {arg['default']})"
                    help_text += arg_text + "\n"
            
            return help_text
        else:
            help_text = f"{self.name} - {self.description}\n\nAvailable subcommands:\n"
            for cmd, info in self.subcommands.items():
                help_text += f"  {cmd}: {info['help']}\n"
            help_text += f"\nUse '{self.name} help <subcommand>' for more information about a specific subcommand."
            return help_text
    
    def execute(self, subcommand: str, args: List[str]) -> CommandResult:
        """Execute a subcommand with the given arguments"""
        if subcommand == "help":
            if args:
                return CommandResult(True, self.get_help(args[0]))
            else:
                return CommandResult(True, self.get_help())
                
        if subcommand in self.subcommands:
            try:
                # Parse arguments based on registered argument definitions
                parsed_args = self._parse_args(subcommand, args)
                if parsed_args is None:
                    return CommandResult(False, "Invalid arguments")
                
                # Call the handler with parsed arguments
                return self.subcommands[subcommand]['handler'](**parsed_args)
            except Exception as e:
                return CommandResult(False, f"Error executing {self.name} {subcommand}: {str(e)}", 
                                    traceback.format_exc())
        else:
            return CommandResult(False, f"Unknown subcommand: {subcommand}. Use '{self.name} help' to see available commands.")
    
    def _parse_args(self, subcommand: str, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse command arguments based on registered argument definitions"""
        cmd_info = self.subcommands[subcommand]
        if not cmd_info['arguments']:
            return {}
        
        parser = argparse.ArgumentParser(prog=f"{self.name} {subcommand}", add_help=False)
        for arg in cmd_info['arguments']:
            if arg.get('positional', True):
                parser.add_argument(arg['name'], 
                                   help=arg.get('help', ''),
                                   default=arg.get('default'),
                                   nargs='?' if not arg.get('required', False) else None)
            else:
                parser.add_argument(f"--{arg['name']}", 
                                   help=arg.get('help', ''),
                                   default=arg.get('default'),
                                   required=arg.get('required', False))
        
        try:
            parsed_args = parser.parse_args(args)
            return vars(parsed_args)
        except SystemExit:
            return None


class VMCommandHandler(CommandHandler):
    """Handler for VM-related commands"""
    
    def __init__(self, vm_environment: VMEnvironment):
        super().__init__("vm", "Manage virtual machines")
        self.vm_environment = vm_environment
        
        # Register VM subcommands
        self.register_subcommand("create", self.create_vm, 
                                "Create a new virtual machine",
                                [
                                    {"name": "name", "help": "Name for the VM", "required": True},
                                    {"name": "memory", "help": "Memory allocation in MB", "default": "1024"},
                                    {"name": "cpus", "help": "Number of CPU cores", "default": "1"}
                                ])
        
        self.register_subcommand("start", self.start_vm,
                                "Start a virtual machine",
                                [{"name": "name", "help": "Name of the VM to start", "required": True}])
        
        self.register_subcommand("stop", self.stop_vm,
                                "Stop a virtual machine",
                                [{"name": "name", "help": "Name of the VM to stop", "required": True}])
        
        self.register_subcommand("pause", self.pause_vm,
                                "Pause a virtual machine",
                                [{"name": "name", "help": "Name of the VM to pause", "required": True}])
        
        self.register_subcommand("resume", self.resume_vm,
                                "Resume a paused virtual machine",
                                [{"name": "name", "help": "Name of the VM to resume", "required": True}])
        
        self.register_subcommand("inspect", self.inspect_vm,
                                "Get detailed information about a virtual machine",
                                [{"name": "name", "help": "Name of the VM to inspect", "required": True}])
        
        self.register_subcommand("list", self.list_vms,
                                "List all virtual machines")
    
    def create_vm(self, name: str, memory: str = "1024", cpus: str = "1") -> CommandResult:
        try:
            # Convert string arguments to appropriate types
            memory_mb = int(memory)
            cpu_count = int(cpus)
            
            # Call VM environment to create a new VM
            result = self.vm_environment.create_vm(name, memory_mb, cpu_count)
            return CommandResult(True, f"Created VM '{name}' with {memory_mb}MB RAM and {cpu_count} CPUs", result)
        except Exception as e:
            return CommandResult(False, f"Failed to create VM: {str(e)}")
    
    def start_vm(self, name: str) -> CommandResult:
        try:
            result = self.vm_environment.start_vm(name)
            return CommandResult(True, f"Started VM '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to start VM: {str(e)}")
    
    def stop_vm(self, name: str) -> CommandResult:
        try:
            result = self.vm_environment.stop_vm(name)
            return CommandResult(True, f"Stopped VM '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to stop VM: {str(e)}")
    
    def pause_vm(self, name: str) -> CommandResult:
        try:
            result = self.vm_environment.pause_vm(name)
            return CommandResult(True, f"Paused VM '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to pause VM: {str(e)}")
    
    def resume_vm(self, name: str) -> CommandResult:
        try:
            result = self.vm_environment.resume_vm(name)
            return CommandResult(True, f"Resumed VM '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to resume VM: {str(e)}")
    
    def inspect_vm(self, name: str) -> CommandResult:
        try:
            vm_info = self.vm_environment.get_vm_info(name)
            if vm_info:
                return CommandResult(True, f"VM '{name}' information:", vm_info)
            else:
                return CommandResult(False, f"VM '{name}' not found")
        except Exception as e:
            return CommandResult(False, f"Failed to inspect VM: {str(e)}")
    
    def list_vms(self) -> CommandResult:
        try:
            vms = self.vm_environment.list_vms()
            if vms:
                vm_list = "\n".join([f"- {vm['name']}: {vm['status']}" for vm in vms])
                return CommandResult(True, f"Available VMs:\n{vm_list}", vms)
            else:
                return CommandResult(True, "No VMs available")
        except Exception as e:
            return CommandResult(False, f"Failed to list VMs: {str(e)}")


class ContainerCommandHandler(CommandHandler):
    """Handler for container-related commands"""
    
    def __init__(self, container_manager: ContainerManager):
        super().__init__("container", "Manage containers")
        self.container_manager = container_manager
        
        # Register container subcommands
        self.register_subcommand("create", self.create_container, 
                                "Create a new container",
                                [
                                    {"name": "name", "help": "Name for the container", "required": True},
                                    {"name": "image", "help": "Container image to use", "required": True},
                                    {"name": "vm", "help": "Target VM to host the container", "required": True}
                                ])
        
        self.register_subcommand("start", self.start_container,
                                "Start a container",
                                [{"name": "name", "help": "Name of the container to start", "required": True}])
        
        self.register_subcommand("stop", self.stop_container,
                                "Stop a container",
                                [{"name": "name", "help": "Name of the container to stop", "required": True}])
        
        self.register_subcommand("pause", self.pause_container,
                                "Pause a container",
                                [{"name": "name", "help": "Name of the container to pause", "required": True}])
        
        self.register_subcommand("resume", self.resume_container,
                                "Resume a paused container",
                                [{"name": "name", "help": "Name of the container to resume", "required": True}])
        
        self.register_subcommand("inspect", self.inspect_container,
                                "Get detailed information about a container",
                                [{"name": "name", "help": "Name of the container to inspect", "required": True}])
        
        self.register_subcommand("list", self.list_containers,
                                "List all containers",
                                [{"name": "vm", "help": "Filter containers by VM", "required": False}])
    
    def create_container(self, name: str, image: str, vm: str) -> CommandResult:
        try:
            result = self.container_manager.create_container(name, image, vm)
            return CommandResult(True, f"Created container '{name}' using image '{image}' on VM '{vm}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to create container: {str(e)}")
    
    def start_container(self, name: str) -> CommandResult:
        try:
            result = self.container_manager.start_container(name)
            return CommandResult(True, f"Started container '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to start container: {str(e)}")
    
    def stop_container(self, name: str) -> CommandResult:
        try:
            result = self.container_manager.stop_container(name)
            return CommandResult(True, f"Stopped container '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to stop container: {str(e)}")
    
    def pause_container(self, name: str) -> CommandResult:
        try:
            result = self.container_manager.pause_container(name)
            return CommandResult(True, f"Paused container '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to pause container: {str(e)}")
    
    def resume_container(self, name: str) -> CommandResult:
        try:
            result = self.container_manager.resume_container(name)
            return CommandResult(True, f"Resumed container '{name}'", result)
        except Exception as e:
            return CommandResult(False, f"Failed to resume container: {str(e)}")
    
    def inspect_container(self, name: str) -> CommandResult:
        try:
            container_info = self.container_manager.get_container_info(name)
            if container_info:


import asyncio
import logging
import os
import platform
import psutil
import uuid
import subprocess
import json
import time
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path

# Import the LensRefractor for resource optimization
from ..ai.circuits import LensRefractor

@dataclass
class VMInstance:
    """Representation of a MicroOS native VM instance."""
    vm_id: str
    name: str
    status: str
    resources: Dict[str, Any]
    pid: Optional[int] = None
    start_time: float = 0.0
    working_directory: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

class VMController:
    """
    Controller for managing native virtual machines without Docker.
    Uses MicroOS's native virtualization capabilities and VMIA neural AI.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lens_refractor = LensRefractor()
        self._active_vms: Dict[str, VMInstance] = {}
        self._db_path = Path("data/vm_store.db")
        self._setup_db()

    def _setup_db(self):
        """Set up the SQLite database for VM persistence."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vms (
            vm_id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            resources TEXT,
            pid INTEGER,
            start_time REAL,
            working_directory TEXT
        )
        ''')
        conn.commit()
        conn.close()
        
        # Load any persisted VMs
        self._load_persisted_vms()

    def _load_persisted_vms(self):
        """Load persisted VMs from the database."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vms")
            vms = cursor.fetchall()
            
            for vm_data in vms:
                vm_id, name, status, resources_json, pid, start_time, working_dir = vm_data
                resources = json.loads(resources_json)
                
                # Create VM instance
                vm = VMInstance(
                    vm_id=vm_id,
                    name=name,
                    status="stopped",  # Initially mark as stopped until we verify
                    resources=resources,
                    pid=pid,
                    start_time=start_time,
                    working_directory=working_dir
                )
                
                # Check if VM process is still running
                if pid and psutil.pid_exists(pid):
                    # Process exists, update status
                    vm.status = status
                    self._active_vms[vm_id] = vm
                else:
                    # Process doesn't exist, update DB to show it's stopped
                    cursor.execute(
                        "UPDATE vms SET status = ?, pid = ? WHERE vm_id = ?",
                        ("stopped", None, vm_id)
                    )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to load persisted VMs: {str(e)}")

    async def start_vm(self, image: str, container_name: Optional[str] = None, 
                     resources: Optional[Dict] = None) -> Dict:
        """
        Start a new VM using MicroOS native virtualization.
        
        Args:
            image: Type of VM to create (e.g., "compute", "network", "storage")
            container_name: Optional name for the VM
            resources: Optional resource constraints
            
        Returns:
            Dictionary with VM details
        """
        try:
            # Default resource constraints
            default_resources = {
                'cpu_count': 1,
                'memory_limit': 512,  # MB
                'disk_space': 1024,   # MB
                'vm_type': image
            }
            
            if resources:
                default_resources.update(resources)
            
            # Use the LensRefractor to optimize resource allocation
            optimized_resources = await self._optimize_resources(default_resources)
            
            # Generate a unique ID for the VM
            vm_id = str(uuid.uuid4())
            vm_name = container_name or f"vm-{vm_id[:8]}"
            
            # Create a working directory for the VM
            working_dir = os.path.abspath(f"vms/{vm_name}")
            os.makedirs(working_dir, exist_ok=True)
            
            # Prepare command to start the VM process
            vm_script_path = self._create_vm_script(vm_id, vm_name, working_dir, optimized_resources)
            
            # Start the VM as a separate process
            process = await self._start_vm_process(vm_script_path, working_dir)
            
            if process:
                # Create VM instance
                vm = VMInstance(
                    vm_id=vm_id,
                    name=vm_name,
                    status="running",
                    resources=optimized_resources,
                    pid=process.pid,
                    start_time=time.time(),
                    working_directory=working_dir
                )
                
                # Store VM in active VMs dictionary
                self._active_vms[vm_id] = vm
                
                # Persist VM details to database
                self._persist_vm(vm)
                
                return {
                    'vm_id': vm_id,
                    'status': vm.status,
                    'name': vm_name,
                    'resources': optimized_resources,
                    'pid': process.pid
                }
            else:
                raise RuntimeError("Failed to start VM process")
        
        except Exception as e:
            self.logger.error(f"Failed to start VM: {str(e)}")
            raise RuntimeError(f"VM start failed: {str(e)}")

    async def _optimize_resources(self, resources: Dict) -> Dict:
        """
        Use the LensRefractor to optimize resource allocation.
        """
        try:
            circuit_config = await self.lens_refractor.generate_circuit(resources)
            optimized_resources = circuit_config["resource_mapping"]
            
            # Ensure we have the minimum required resources
            for key, value in resources.items():
                if key not in optimized_resources or optimized_resources[key] < value * 0.5:
                    optimized_resources[key] = value
            
            return optimized_resources
        except Exception as e:
            self.logger.warning(f"Resource optimization failed, using defaults: {str(e)}")
            return resources

    def _create_vm_script(self, vm_id: str, vm_name: str, working_dir: str, resources: Dict) -> str:
        """
        Create a Python script that will run as the VM process.
        """
        script_path = os.path.join(working_dir, "vm_process.py")
        
        with open(script_path, "w") as f:
            f.write(f"""#!/usr/bin/env python3
import os
import sys
import time
import json
import socket
import threading
import signal
import psutil

# VM metadata
VM_ID = "{vm_id}"
VM_NAME = "{vm_name}"
RESOURCES = {json.dumps(resources)}
LOG_FILE = os.path.join("{working_dir}", "vm.log")

def log(message):
    with open(LOG_FILE, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{{timestamp}} - {{message}}\\n")

def collect_metrics():
    while True:
        try:
            # Collect system metrics
            process = psutil.Process()
            metrics = {{
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "threads": process.num_threads(),
                "uptime": time.time() - process.create_time()
            }}
            
            # Write metrics to file
            with open(os.path.join("{working_dir}", "metrics.json"), "w") as f:
                json.dump(metrics, f)
                
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            log(f"Error collecting metrics: {{str(e)}}")
            time.sleep(10)  # Retry after 10 seconds on error

def handle_shutdown(signum, frame):
    log(f"Received signal {{signum}}, shutting down VM")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == "__main__":
    log(f"Starting VM {{VM_NAME}} (ID: {{VM_ID}})")
    log(f"Resources: {{RESOURCES}}")
    
    # Start metrics collection in a background thread
    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()
    
    # Main VM loop - this would be where the actual VM workload runs
    try:
        while True:
            # Simulate VM workload
            time.sleep(1)
    except Exception as e:
        log(f"VM error: {{str(e)}}")
        sys.exit(1)
""")
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        return script_path

    async def _start_vm_process(self, script_path: str, working_dir: str):
        """
        Start the VM as a separate Python process.
        """
        try:
            # Use subprocess to start the VM process
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Ensures the process runs independently
            )
            
            # Wait a moment to ensure the process started
            await asyncio.sleep(0.5)
            
            # Check if process is still running
            if process.poll() is None:
                return process
            else:
                stdout, stderr = process.communicate()
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"VM process failed to start: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to start VM process: {str(e)}")
            return None

    def _persist_vm(self, vm: VMInstance):
        """Persist VM details to the database."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT OR REPLACE INTO vms VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    vm.vm_id,
                    vm.name,
                    vm.status,
                    json.dumps(vm.resources),
                    vm.pid,
                    vm.start_time,
                    vm.working_directory
                )
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to persist VM {vm.vm_id}: {str(e)}")

    async def stop_vm(self, vm_id: str) -> Dict:
        """
        Stop a running VM.
        
        Args:
            vm_id: ID of the VM to stop
            
        Returns:
            Status dictionary
        """
        try:
            if vm_id not in self._active_vms:
                raise ValueError(f"VM {vm_id} not found")
            
            vm = self._active_vms[vm_id]
            
            # If VM has a PID, attempt to terminate the process
            if vm.pid and psutil.pid_exists(vm.pid):
                process = psutil.Process(vm.pid)
                process.terminate()
                
                # Wait for process to terminate
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    # Force kill if terminate doesn't work
                    process.kill()
            
            # Update VM status
            vm.status = "stopped"
            vm.pid = None
            
            # Update database
            self._persist_vm(vm)
            
            return {'status': 'stopped', 'vm_id': vm_id}
        
        except Exception as e:
            self.logger.error(f"Failed to stop VM {vm_id}: {str(e)}")
            raise RuntimeError(f"VM stop failed: {str(e)}")

    async def get_vm_status(self, vm_id: str) -> Dict:
        """
        Get the status of a VM.
        
        Args:
            vm_id: ID of the VM to check
            
        Returns:
            Status dictionary with VM details
        """
        try:
            if vm_id not in self._active_vms:
                return {'status': 'not_found', 'vm_id': vm_id}
            
            vm = self._active_vms[vm_id]
            
            # Check if process is still running
            if vm.pid and psutil.pid_exists(vm.pid):
                process = psutil.Process(vm.pid)
                
                # Get metrics
                metrics = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'memory_mb': process.memory_info().rss / (1024 * 1024),
                    'threads': process.num_threads(),
                    'uptime': time.time() - process.create_time()
                }
                
                # Update VM metrics
                vm.metrics = metrics
                
                return {
                    'status': vm.status,
                    'vm_id': vm_id,
                    'name': vm.name,
                    'resources': vm.resources,
                    'metrics': metrics,
                    'start_time': vm.start_time
                }
            else:
                # VM process not running, update status
                vm.status = "stopped"
                self._persist_vm(vm)
                
                return {
                    'status': 'stopped',
                    'vm_id': vm_id,
                    'name': vm.name,
                    'resources': vm.resources,
                    'start_time': vm.start_time
                }
        
        except Exception as e:
            self.logger.error(f"Failed to get VM status for {vm_id}: {str(e)}")
            raise RuntimeError(f"Failed to get VM status: {str(e)}")

    async def list_vms(self) -> List[Dict]:
        """
        List all VMs.
        
        Returns:
            List of VM details
        """
        vm_list = []
        
        for vm_id, vm in self._active_vms.items():
            # Check if process is still running for running VMs
            if vm.status == "running" and vm.pid:
                if not self._is_process_running(vm.pid):
                    vm.status = "stopped"  # Update status if process is no longer running

import asyncio
import logging
import uuid
import time
import os
import json
from typing import Dict, List, Optional
import libvirt
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ..ai.circuits import LensRefractor

class VMController:
    """
    Controls virtual machine operations using libvirt instead of Docker.
    Integrates with VMIA neural AI through LensRefractor for resource optimization.
    """
    def __init__(self, uri: str = "qemu:///system"):
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self._conn = None
        self._active_vms: Dict[str, Dict] = {}
        self.lens_refractor = LensRefractor()
        self.vm_base_path = Path(os.path.expanduser("~/.micro_os/vm_images"))
        self.vm_base_path.mkdir(parents=True, exist_ok=True)
        
    async def _connect_libvirt(self):
        """Establish connection to libvirt daemon"""
        if self._conn is None or not self._conn.isAlive():
            try:
                self._conn = libvirt.open(self.uri)
                if not self._conn:
                    raise RuntimeError(f"Failed to open connection to {self.uri}")
                self.logger.info(f"Connected to libvirt daemon: {self.uri}")
            except libvirt.libvirtError as e:
                self.logger.error(f"Failed to connect to libvirt: {str(e)}")
                raise RuntimeError(f"Libvirt connection failed: {str(e)}")
                
    async def _ensure_connected(self):
        """Ensure we have a valid libvirt connection"""
        if self._conn is None or not self._conn.isAlive():
            await self._connect_libvirt()
    
    async def _generate_vm_xml(self, vm_name: str, memory_mb: int, vcpus: int, 
                             disk_path: str, network: str = "default") -> str:
        """Generate libvirt XML for VM definition"""
        root = ET.Element("domain", type="kvm")
        
        # Basic VM metadata
        name = ET.SubElement(root, "name")
        name.text = vm_name
        
        # Memory allocation
        memory = ET.SubElement(root, "memory", unit="MiB")
        memory.text = str(memory_mb)
        currentMemory = ET.SubElement(root, "currentMemory", unit="MiB")
        currentMemory.text = str(memory_mb)
        
        # CPU configuration
        vcpu = ET.SubElement(root, "vcpu", placement="static")
        vcpu.text = str(vcpus)
        
        # OS configuration
        os = ET.SubElement(root, "os")
        type_elem = ET.SubElement(os, "type", arch="x86_64", machine="pc-q35-5.2")
        type_elem.text = "hvm"
        boot = ET.SubElement(os, "boot", dev="hd")
        
        # Features
        features = ET.SubElement(root, "features")
        ET.SubElement(features, "acpi")
        ET.SubElement(features, "apic")
        ET.SubElement(features, "vmport", state="off")
        
        # CPU mode
        cpu = ET.SubElement(root, "cpu", mode="host-model")
        
        # Devices
        devices = ET.SubElement(root, "devices")
        
        # Disk
        disk = ET.SubElement(devices, "disk", type="file", device="disk")
        ET.SubElement(disk, "driver", name="qemu", type="qcow2")
        ET.SubElement(disk, "source", file=disk_path)
        ET.SubElement(disk, "target", dev="vda", bus="virtio")
        
        # Network
        interface = ET.SubElement(devices, "interface", type="network")
        ET.SubElement(interface, "source", network=network)
        ET.SubElement(interface, "model", type="virtio")
        
        # Console
        console = ET.SubElement(devices, "console", type="pty")
        ET.SubElement(console, "target", type="serial", port="0")
        
        # Graphics
        graphics = ET.SubElement(devices, "graphics", type="vnc", port="-1", autoport="yes", listen="0.0.0.0")
        ET.SubElement(graphics, "listen", type="address", address="0.0.0.0")
        
        # Convert to XML string
        return ET.tostring(root, encoding="unicode")
    
    async def _create_disk_image(self, base_image: str, vm_name: str, size_gb: int = 10) -> str:
        """Create a disk image for the VM from a base image"""
        disk_path = str(self.vm_base_path / f"{vm_name}.qcow2")
        
        # Check if base image exists or needs to be downloaded
        base_image_path = self.vm_base_path / base_image
        if not base_image_path.exists():
            self.logger.info(f"Base image {base_image} not found, downloading...")
            # This would download or prepare the base image
            # For simplicity, we'll just assume it exists or create a blank one
            await self._create_blank_image(str(base_image_path), 5)
        
        # Create a new disk based on the base image
        cmd = [
            "qemu-img", "create", 
            "-f", "qcow2", 
            "-b", str(base_image_path),
            disk_path,
            f"{size_gb}G"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Failed to create disk: {stderr.decode()}")
                
            return disk_path
        except Exception as e:
            self.logger.error(f"Error creating disk image: {str(e)}")
            raise RuntimeError(f"Disk creation failed: {str(e)}")
    
    async def _create_blank_image(self, path: str, size_gb: int = 5):
        """Create a blank disk image"""
        cmd = ["qemu-img", "create", "-f", "qcow2", path, f"{size_gb}G"]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Failed to create blank image: {stderr.decode()}")
        except Exception as e:
            self.logger.error(f"Error creating blank image: {str(e)}")
            raise RuntimeError(f"Blank image creation failed: {str(e)}")
    
    async def _optimize_resources(self, resource_requirements: Dict) -> Dict:
        """Use LensRefractor to optimize VM resource allocation"""
        # Default resource values
        defaults = {
            "memory_mb": 512,
            "vcpus": 1,
            "disk_gb": 10,
            "io_weight": 100,
            "network_bandwidth": 100  # Mbps
        }
        
        # If no requirements specified, use defaults
        if not resource_requirements:
            return defaults
            
        try:
            # Get optimized resource configuration from LensRefractor
            circuit_result = await self.lens_refractor.generate_circuit(resource_requirements)
            optimized_resources = circuit_result["resource_mapping"]
            
            # Apply optimized values while keeping defaults for missing values
            for key, value in defaults.items():
                if key not in optimized_resources:
                    optimized_resources[key] = value
                    
            return optimized_resources
        except Exception as e:
            self.logger.warning(f"Resource optimization failed, using defaults: {str(e)}")
            return defaults
    
    async def start_vm(self, image: str, vm_name: Optional[str] = None, 
                      resources: Optional[Dict] = None) -> Dict:
        """
        Start a new VM using the specified image and resources.
        
        Args:
            image: Base OS image to use
            vm_name: Optional name for the VM, will generate if not provided
            resources: Optional resource specifications
            
        Returns:
            Dictionary with VM details
        """
        try:
            await self._ensure_connected()
            
            # Generate a VM name if not provided
            if not vm_name:
                vm_name = f"micro-os-vm-{str(uuid.uuid4())[:8]}"
                
            # Optimize resources using neural AI
            optimized_resources = await self._optimize_resources(resources or {})
            
            # Create disk image
            disk_path = await self._create_disk_image(
                image, 
                vm_name, 
                int(optimized_resources.get("disk_gb", 10))
            )
            
            # Generate VM XML definition
            vm_xml = await self._generate_vm_xml(
                vm_name=vm_name,
                memory_mb=int(optimized_resources.get("memory_mb", 512)),
                vcpus=int(optimized_resources.get("vcpus", 1)),
                disk_path=disk_path,
                network="default"
            )
            
            # Define and start the VM
            domain = self._conn.defineXML(vm_xml)
            if not domain:
                raise RuntimeError(f"Failed to define VM from XML")
                
            domain.create()  # Start the VM
            
            # Store VM info
            vm_id = str(domain.UUIDString())
            self._active_vms[vm_id] = {
                "domain": domain,
                "name": vm_name,
                "resources": optimized_resources,
                "created_at": time.time(),
                "status": "running"
            }
            
            self.logger.info(f"Started VM {vm_name} with ID {vm_id}")
            
            # Return VM info
            return {
                "vm_id": vm_id,
                "name": vm_name,
                "status": "running",
                "resources": optimized_resources
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start VM: {str(e)}")
            raise RuntimeError(f"VM start failed: {str(e)}")
    
    async def stop_vm(self, vm_id: str) -> Dict:
        """
        Stop a running VM.
        
        Args:
            vm_id: ID of the VM to stop
            
        Returns:
            Dictionary with operation result
        """
        try:
            await self._ensure_connected()
            
            if vm_id not in self._active_vms:
                raise ValueError(f"VM with ID {vm_id} not found")
                
            vm_info = self._active_vms[vm_id]
            domain = vm_info["domain"]
            
            # Try graceful shutdown first
            domain.shutdown()
            
            # Wait for up to 30 seconds for graceful shutdown
            for _ in range(30):
                if not domain.isActive():
                    break
                await asyncio.sleep(1)
                
            # Force off if still running
            if domain.isActive():
                self.logger.warning(f"VM {vm_id} did not shut down gracefully, forcing off")
                domain.destroy()
            
            # Update status
            vm_info["status"] = "stopped"
            
            return {
                "vm_id": vm_id,
                "name": vm_info["name"],
                "status": "stopped"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop VM {vm_id}: {str(e)}")
            raise RuntimeError(f"VM stop failed: {str(e)}")
    
    async def get_vm_status(self, vm_id: str) -> Dict:
        """
        Get detailed status of a VM.
        
        Args:
            vm_id: ID of the VM
            
        Returns:
            Dictionary with VM status and details
        """
        try:
            await self._ensure_connected()
            
            if vm_id not in self._active_vms:
                return {"status": "not_found", "vm_id": vm_id}
                
            vm_info = self._active_vms[vm_id]
            domain = vm_info["domain"]
            
            # Get current state
            state, reason = domain.state()
            state_map = {
                libvirt.VIR_DOMAIN_NOSTATE: "no_state",
                libvirt.VIR_DOMAIN_RUNNING: "running",
                libvirt.VIR_DOMAIN_BLOCKED: "blocked",
                libvirt.VIR_DOMAIN_PAUSED: "paused",
                libvirt.VIR_DOMAIN_SHUTDOWN: "shutdown",
                libvirt.VIR_DOMAIN_SHUTOFF: "shutoff",
                libvirt.VIR_DOMAIN_CRASHED: "crashed",
                libvirt.VIR_DOMAIN_PMSUSPENDED: "suspended"
            }
            
            # Get stats
            stats = {}
            if state == libvirt.VIR_DOMAIN_RUNNING:
                # Memory stats
                try:
                    mem_stats = domain.memoryStats()
                    stats["memory"] = {
                        "total": mem_stats.get("actual", 0),
                        "available": mem_stats.get("unused", 0),
                        "used": mem_stats.get("actual", 0) - mem_stats.get("unused", 0)
                    }
                except:
                    stats["memory"] = {"error": "Could not retrieve memory stats"}
                
                # CPU stats
                try:
                    cpu_stats = domain.getCPUStats(True)
                    stats["cpu"] = {
                        "cpu_time": cpu_stats[0]["cpu_time"],
                        "system_time": cpu_stats[0]["system_time"],
                        "user_time": cpu_stats[0]["user_time"]
                    }
                except:
                    stats["cpu"] = {"error": "Could not retrieve CPU stats"}
            
            # Update status in our records
            vm_info["status"] = state_map.get(state, "unknown")
            
            return {
                "vm_id": vm_id,
                "name": vm_info["name"],
                "status": state_map.get(state, "unknown"),
                "stats": stats,
                "resources": vm_info["resources"],
                "created_at": vm_info["created_at"],
                "uptime": time.time() - vm_info["created_at"] if state == libvirt.VIR_DOMAIN_RUNNING else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get VM status: {str(e)}")
            raise RuntimeError(f"Failed to get VM status: {str(e)}")
    
    async def list_vms(self) -> List[Dict]:
        """
        List all VMs managed by this controller.
        
        Returns:
            List of VM information dictionaries
        """
        try:
            await self._ensure_connected()
            
            result = []
            
            # First, get all active domains
            for vm_id, vm_info in self._active_vms.items():
                domain = vm_info["domain"]
                
                try:
                    # Check if domain is still valid
                    if domain.isActive():
                        state = "running"
                    else:
                        state = "stopped"
                        
                    result.append({
                        "vm_id": vm_id,
                        "name": vm_info["name"],
                        "status": state,
                        "created_at": vm_info["created_at"]
                    })
                except libvirt.libvirtError:
                    # Domain may have been undefined or deleted
                    self.logger.warning(f"Domain {vm_id} no longer exists, removing from active VMs")
                    del self._active_vms[vm_id]
            
            # Also look for any domains that might exist but aren't in our active_vms
            try:
                all_domains = self._conn.listAllDomains()
                for domain in all_domains:
                    vm_id = str(domain.UUIDString())
                    if vm_id not in self._active_vms:
                        # This domain exists but isn't in our active_vms list
                        state, _ = domain.state()
                        state_map = {
                            libvirt.VIR_DOMAIN_RUNNING: "running",
                            libvirt.VIR_DOMAIN_SHUTOFF: "stopped"
                        }
                        
                        # Add to our active_vms
                        self._active_vms[vm_id] = {
                            "domain": domain,
                            "name": domain.name(),
                            "created_at": time.time(),  # Approximate
                            "status": state_map.get(state, "unknown"),
                            "resources": {}  # We don't know the resources
                        }
                        
                        result.append({
                            "vm_id": vm_id,
                            "name": domain.name(),
                            "status": state_map.get(state, "unknown"),
                            "created_at": time.time()  # Approximate
                        })
            except libvirt.libvirtError as e:
                self.logger.error(f"Error listing domains: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list VMs: {str(e)}")
            raise RuntimeError(f"Failed to list VMs: {str(e)}")

    async def debug_vm(self, vm_id: str) -> Dict:
        """
        Get debugging information for a VM.
        
        Args:
            vm_id: ID of the VM to debug
            
        Returns:
            Dictionary with debugging information
        """
        try:
            await self._ensure_connected()
            
            if vm_id not in self._active_vms:
                raise ValueError(f"VM with ID {vm_id} not found")
                
            vm_info = self._active_vms[vm_id]
            domain = vm_info["domain"]
            
            # Get detailed information
            result = {
                "vm_id": vm_id,
                "name": vm_info["name"],
                "status": vm_info["status"],
                "resources": vm_info["resources"],
                "created_at": vm_info["created_at"],
                "uptime": time.time() - vm_info["created_at"] if domain.isActive() else 0,
                "xml": domain.XMLDesc(),
                "logs": await self._get_vm_logs(vm_id)
            }
            
            # Get runtime stats if VM is running
            if domain.isActive():
                # Memory stats
                try:
                    mem_stats = domain.memoryStats()
                    result["memory_stats"] = {
                        "total": mem_stats.get("actual", 0),
                        "available": mem_stats.get("unused", 0),
                        "used": mem_stats.get("actual", 0) - mem_stats.get("unused", 0)
                    }
                except Exception as e:
                    result["memory_stats"] = {"error": str(e)}
                
                # CPU stats
                try:
                    cpu_stats = domain.getCPUStats(True)
                    result["cpu_stats"] = {
                        "cpu_time": cpu_stats[0]["cpu_time"],
                        "system_time": cpu_stats[0]["system_time"],
                        "user_time": cpu_stats[0]["user_time"]
                    }
                except Exception as e:
                    result["cpu_stats"] = {"error": str(e)}
                
                # Network stats
                try:
                    interfaces = domain.interfaceStats("vnet0")
                    result["network_stats"] = {
                        "rx_bytes": interfaces[0],
                        "rx_packets": interfaces[1],
                        "rx_errs": interfaces[2],
                        "rx_drop": interfaces[3],
                        "tx_bytes": interfaces[4],
                        "tx_packets": interfaces[5],
                        "tx_errs": interfaces[6],
                        "tx_drop": interfaces[7]
                    }
                except Exception as e:
                    result["network_stats"] = {"error": str(e)}
                
                # Block device stats
                try:
                    block_stats = domain.blockStats("vda")
                    result["block_stats"] = {
                        "rd_req": block_stats[0],
                        "rd_bytes": block_stats[1],
                        "wr_req": block_stats[2],
                        "wr_bytes": block_stats[3],
                        "errors": block_stats[4]
                    }
                except Exception as e:
                    result["block_stats"] = {"error": str(e)}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to debug VM {vm_id}: {str(e)}")
            raise RuntimeError(f"Debug VM failed: {str(e)}")
            
    async def _get_vm_logs(self, vm_id: str) -> str:
        """Get logs for a VM using libvirt's console functionality"""
        if vm_id not in self._active_vms:
            return "VM not found"
            
        vm_info = self._active_vms[vm_id]
        domain = vm_info["domain"]
        
        try:
            # Attempt to get console logs using virsh
            log_file = f"/tmp/vm_{vm_id}_console.log"
            cmd = ["virsh", "console", domain.name(), "--force"]
            
            # This is a simplified approach - in practice, getting console logs
            # from a libvirt domain requires more complex handling
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    return stdout.decode()
                else:
                    return f"Error getting logs: {stderr.decode()}"
            except Exception as e:
                return f"Failed to get console logs: {str(e)}"
                
        except Exception as e:
            return f"Error retrieving logs: {str(e)}"
            
    async def restart_vm(self, vm_id: str) -> Dict:
        """
        Restart a VM.
        
        Args:
            vm_id: ID of the VM to restart
            
        Returns:
            Dictionary with operation result
        """
        try:
            await self._ensure_connected()
            
            if vm_id not in self._active_vms:
                raise ValueError(f"VM with ID {vm_id} not found")
                
            vm_info = self._active_vms[vm_id]
            domain = vm_info["domain"]
            
            # Reboot the domain
            domain.reboot()
            
            return {
                "vm_id": vm_id,
                "name": vm_info["name"],
                "status": "restarting"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to restart VM {vm_id}: {str(e)}")
            raise RuntimeError(f"VM restart failed: {str(e)}")
            
    async def close(self):
        """Close the libvirt connection and clean up resources"""
        if self._conn:
            try:
                self._conn.close()
                self.logger.info("Closed libvirt connection")
            except Exception as e:
                self.logger.error(f"Error closing libvirt connection: {str(e)}")

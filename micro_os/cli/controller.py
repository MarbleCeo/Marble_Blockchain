import click
import asyncio
import json
from typing import Optional, Dict
import logging
from pathlib import Path
import sys

from ..vm.controller import VMController
from ..network.p2p_vm import P2PNetworkManager

class CLIController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vm_controller = VMController()
        self.network_manager = P2PNetworkManager()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('micro_os.log')
            ]
        )

@click.group()
@click.pass_context
def cli(ctx):
    """MicroOS CLI - VM and Network Management Interface"""
    ctx.obj = CLIController()

@cli.group()
def vm():
    """VM management commands"""
    pass

@vm.command('start')
@click.option('--image', required=True, help='Docker image to run')
@click.option('--name', help='Container name')
@click.option('--cpu-count', default=1, help='Number of CPUs')
@click.option('--memory', default='512m', help='Memory limit')
@click.option('--network', default='bridge', help='Network mode')
@click.pass_obj
def vm_start(controller: CLIController, image: str, name: Optional[str],
           cpu_count: int, memory: str, network: str):
    """Start a new VM instance"""
    try:
        resources = {
            'cpu_count': cpu_count,
            'mem_limit': memory,
            'network_mode': network
        }
        
        result = asyncio.run(controller.vm_controller.start_vm(
            image=image,
            container_name=name,
            resources=resources
        ))
        
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Error starting VM: {str(e)}", err=True)
        sys.exit(1)

@vm.command('stop')
@click.option('--container-id', required=True, help='Container ID to stop')
@click.pass_obj
def vm_stop(controller: CLIController, container_id: str):
    """Stop a running VM instance"""
    try:
        result = asyncio.run(controller.vm_controller.stop_vm(container_id))
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Error stopping VM: {str(e)}", err=True)
        sys.exit(1)

@vm.command('debug')
@click.option('--container-id', required=True, help='Container ID to debug')
@click.option('--tail', default=100, help='Number of log lines to show')
@click.pass_obj
def vm_debug(controller: CLIController, container_id: str, tail: int):
    """Debug a VM instance"""
    try:
        result = asyncio.run(controller.vm_controller.debug_vm(container_id))
        
        click.echo("\n=== VM Debug Information ===")
        click.echo(f"\nContainer ID: {container_id}")
        click.echo(f"Status: {result['status']}")
        
        click.echo("\n=== Last Logs ===")
        click.echo(result['logs'])
        
        click.echo("\n=== Container Inspection ===")
        click.echo(json.dumps(result['inspection'], indent=2))
        
    except Exception as e:
        click.echo(f"Error debugging VM: {str(e)}", err=True)
        sys.exit(1)

@vm.command('list')
@click.pass_obj
def vm_list(controller: CLIController):
    """List all VM instances"""
    try:
        result = asyncio.run(controller.vm_controller.list_vms())
        click.echo("\n=== Active VMs ===")
        for vm in result:
            click.echo(
                f"\nID: {vm['container_id']}\n"
                f"Name: {vm['name']}\n"
                f"Status: {vm['status']}"
            )
        
    except Exception as e:
        click.echo(f"Error listing VMs: {str(e)}", err=True)
        sys.exit(1)

@vm.command('status')
@click.option('--container-id', required=True, help='Container ID to check')
@click.pass_obj
def vm_status(controller: CLIController, container_id: str):
    """Get detailed status of a VM instance"""
    try:
        result = asyncio.run(controller.vm_controller.get_vm_status(container_id))
        click.echo("\n=== VM Status ===")
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Error getting VM status: {str(e)}", err=True)
        sys.exit(1)

@cli.group()
def network():
    """Network management commands"""
    pass

@network.command('start')
@click.option('--host', default='0.0.0.0', help='Host IP to bind')
@click.option('--port', default=8888, help='Port to listen on')
@click.pass_obj
def network_start(controller: CLIController, host: str, port: int):
    """Start P2P network node"""
    try:
        controller.network_manager = P2PNetworkManager(host_ip=host, port=port)
        result = asyncio.run(controller.network_manager.start())
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Error starting network: {str(e)}", err=True)
        sys.exit(1)

@network.command('connect')
@click.option('--peer', required=True, help='Peer address to connect to')
@click.pass_obj
def network_connect(controller: CLIController, peer: str):
    """Connect to a peer"""
    try:
        result = asyncio.run(controller.network_manager.connect_to_peer(peer))
        status = "success" if result else "failed"
        click.echo(f"Connection to peer {peer}: {status}")
        
    except Exception as e:
        click.echo(f"Error connecting to peer: {str(e)}", err=True)
        sys.exit(1)

@network.command('peers')
@click.pass_obj
def network_peers(controller: CLIController):
    """List connected peers"""
    try:
        peers = asyncio.run(controller.network_manager.discover_peers())
        click.echo("\n=== Connected Peers ===")
        for peer in peers:
            click.echo(
                f"\nPeer ID: {peer['peer_id']}\n"
                f"Last Seen: {peer['last_seen']}\n"
                f"State: {json.dumps(peer['state'], indent=2)}"
            )
        
    except Exception as e:
        click.echo(f"Error listing peers: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()


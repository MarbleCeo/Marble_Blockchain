import docker
from typing import Dict, List, Optional
import logging
import json

class VMController:
    def __init__(self):
        self.client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        self._active_containers: Dict[str, docker.models.containers.Container] = {}

    async def start_vm(self, image: str, container_name: Optional[str] = None, 
                    resources: Optional[Dict] = None) -> Dict:
        try:
            # Default resource constraints
            default_resources = {
                'cpu_count': 1,
                'mem_limit': '512m',
                'network_mode': 'bridge'
            }
            
            if resources:
                default_resources.update(resources)

            container = self.client.containers.run(
                image,
                name=container_name,
                detach=True,
                **default_resources
            )
            
            container_id = container.id
            self._active_containers[container_id] = container
            
            return {
                'container_id': container_id,
                'status': container.status,
                'name': container.name
            }
        
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to start VM: {str(e)}")
            raise RuntimeError(f"VM start failed: {str(e)}")

    async def stop_vm(self, container_id: str) -> Dict:
        try:
            if container_id in self._active_containers:
                container = self._active_containers[container_id]
                container.stop()
                del self._active_containers[container_id]
                return {'status': 'stopped', 'container_id': container_id}
            else:
                raise ValueError(f"Container {container_id} not found")
        
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to stop VM: {str(e)}")
            raise RuntimeError(f"VM stop failed: {str(e)}")

    async def get_vm_status(self, container_id: str) -> Dict:
        try:
            container = self._active_containers.get(container_id)
            if not container:
                return {'status': 'not_found', 'container_id': container_id}
            
            container.reload()  # Refresh container info
            stats = container.stats(stream=False)  # Get current stats
            
            return {
                'status': container.status,
                'container_id': container_id,
                'name': container.name,
                'cpu_usage': stats['cpu_stats']['cpu_usage']['total_usage'],
                'memory_usage': stats['memory_stats'].get('usage', 0),
                'network': stats['networks'] if 'networks' in stats else {}
            }
        
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to get VM status: {str(e)}")
            raise RuntimeError(f"Failed to get VM status: {str(e)}")

    async def list_vms(self) -> List[Dict]:
        return [
            {
                'container_id': container_id,
                'name': container.name,
                'status': container.status
            }
            for container_id, container in self._active_containers.items()
        ]

    async def debug_vm(self, container_id: str) -> Dict:
        try:
            container = self._active_containers.get(container_id)
            if not container:
                raise ValueError(f"Container {container_id} not found")
            
            logs = container.logs(tail=100).decode('utf-8')
            inspection = container.inspect()
            
            return {
                'container_id': container_id,
                'logs': logs,
                'inspection': inspection,
                'status': container.status
            }
        
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to debug VM: {str(e)}")
            raise RuntimeError(f"VM debug failed: {str(e)}")


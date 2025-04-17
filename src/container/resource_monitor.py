#!/usr/bin/env python3
"""
Resource Monitor for Container System

This module provides real-time monitoring of system resources (CPU, memory, disk, network)
to support intelligent container scheduling and scaling decisions.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
import psutil
from pathlib import Path

# Configure logging
logger = logging.getLogger("ResourceMonitor")

# Define constants
DEFAULT_CHECK_INTERVAL = 5       # 5 seconds between resource checks
DEFAULT_MEMORY_THRESHOLD = 0.85  # 85% memory usage triggers alert
DEFAULT_CPU_THRESHOLD = 0.80     # 80% CPU usage triggers alert
DEFAULT_DISK_THRESHOLD = 0.90    # 90% disk usage triggers alert
DEFAULT_HISTORY_SIZE = 100       # Max number of resource metrics to store in history

class ResourceMonitor:
    """
    Monitors system resources and containers resource usage.
    
    Provides real-time metrics and alerts for resource constraints.
    Supports monitoring of CPU, memory, disk, and network usage.
    """
    
    def __init__(self, check_interval: int = DEFAULT_CHECK_INTERVAL):
        """
        Initialize the resource monitor.
        
        Args:
            check_interval: Interval in seconds between resource checks
        """
        self.check_interval: int = check_interval
        self._stop_event: threading.Event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.resource_history: List[Dict[str, Any]] = []
        self.history_max_size: int = DEFAULT_HISTORY_SIZE
        
        # Initialize resource thresholds
        self.memory_threshold: float = DEFAULT_MEMORY_THRESHOLD
        self.cpu_threshold: float = DEFAULT_CPU_THRESHOLD
        self.disk_threshold: float = DEFAULT_DISK_THRESHOLD
        
        # Resource alerts storage
        self.resource_alerts: List[Dict[str, Any]] = []
        self.max_alerts: int = 100
        
        logger.info("ResourceMonitor initialized with check interval: %d seconds", check_interval)
    
    def start(self) -> None:
        """
        Start the resource monitoring thread.
        
        Begins periodic collection of system metrics at the specified interval.
        Does nothing if monitoring is already running.
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring thread is already running")
            return
            
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self) -> None:
        """
        Stop the resource monitoring thread.
        
        Gracefully stops the monitoring thread and waits for it to terminate.
        Does nothing if monitoring is not running.
        """
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring thread is not running")
            return
            
        self._stop_event.set()
        self._monitoring_thread.join(timeout=10)
        logger.info("Resource monitoring stopped")
    
    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a listener that will be called with resource metrics.
        
        Args:
            callback: Function to call with resource metrics. The function
                     should accept a dictionary of resource metrics.
        """
        self.listeners.append(callback)
        logger.debug("Added resource metrics listener")
    
    def remove_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove a listener.
        
        Args:
            callback: Function to remove from listeners
        """
        if callback in self.listeners:
            self.listeners.remove(callback)
            logger.debug("Removed resource metrics listener")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current system resource metrics.
        
        Returns:
            Dictionary containing current resource metrics including CPU, memory,
            disk, and network usage.
        """
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.1),
                    'count': psutil.cpu_count(),
                    'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
                    'load_avg': self._get_load_average()
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used,
                    'free': psutil.virtual_memory().free,
                    'swap': {
                        'total': psutil.swap_memory().total,
                        'used': psutil.swap_memory().used,
                        'percent': psutil.swap_memory().percent
                    }
                },
                'disk': self._get_disk_metrics(),
                'network': self._get_network_metrics()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics in case of error
            return {
                'timestamp': time.time(),
                'error': str(e),
                'cpu': {'percent': 0},
                'memory': {'percent': 0},
                'disk': {'partitions': []},
                'network': {'bytes_sent': 0, 'bytes_recv': 0}
            }
    
    def _get_load_average(self) -> List[float]:
        """
        Get system load average if available.
        
        Returns:
            List of 1, 5, and 15 minute load averages or empty list if not available
        """
        try:
            return [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        except (AttributeError, OSError):
            # getloadavg might not be available on all platforms
            return []
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """
        Get detailed disk usage metrics.
        
        Returns:
            Dictionary containing disk metrics including partition usage and I/O statistics
        """
        disk_metrics = {
            'partitions': [],
            'io': {
                'read_count': 0,
                'write_count': 0,
                'read_bytes': 0,
                'write_bytes': 0,
                'read_time': 0,
                'write_time': 0
            }
        }
        
        # Get partition information
        try:
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_metrics['partitions'].append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent
                    })
                except (PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not access disk partition {partition.mountpoint}: {e}")
        except Exception as e:
            logger.error(f"Error retrieving disk partitions: {e}")
        
        # Get I/O statistics
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                disk_metrics['io'] = {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_time': getattr(io_counters, 'read_time', 0),
                    'write_time': getattr(io_counters, 'write_time', 0)
                }
        except (AttributeError, FileNotFoundError, OSError) as e:
            logger.warning(f"Could not access disk I/O statistics: {e}")
            
        return disk_metrics
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """
        Get detailed network usage metrics.
        
        Returns:
            Dictionary containing network metrics including bytes sent/received
            and packet information
        """
        network_metrics = {
            'bytes_sent': 0,
            'bytes_recv': 0,
            'packets_sent': 0,
            'packets_recv': 0,
            'interfaces': {}
        }
        
        try:
            # Get overall network I/O
            net_io = psutil.net_io_counters()
            network_metrics.update({
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            })
            
            # Get per-interface statistics
            net_if_stats = psutil.net_if_stats()
            net_io_counters_per_nic = psutil.net_io_counters(pernic=True)
            
            for interface, stats in net_if_stats.items():
                if interface in net_io_counters_per_nic:
                    counters = net_io_counters_per_nic[interface]
                    network_metrics['interfaces'][interface] = {
                        'isup': stats.isup,
                        'speed': getattr(stats, 'speed', 0),
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv,
                        'packets_sent': counters.packets_sent,
                        'packets_recv': counters.packets_recv
                    }
        except (AttributeError, OSError) as e:
            logger.warning(f"Could not access network statistics: {e}")
            
        return network_metrics
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that collects metrics at regular intervals.
        
        This method runs in a separate thread and periodically collects system
        resource metrics, checks thresholds, and notifies listeners.
        """
        logger.info("Resource monitoring loop started")
        last_metrics = None
        
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.get_current_metrics()
                
                # Calculate rates if we have previous metrics
                if last_metrics is not None:
                    time_diff = metrics['timestamp'] - last_metrics['timestamp']
                    if time_diff > 0:
                        self._calculate_rates(metrics, last_metrics, time_diff)
                
                # Store current metrics for next iteration
                last_metrics = metrics
                
                # Store in history with size limit
                self.resource_history.append(metrics)
                if len(self.resource_history) > self.history_max_size:
                    self.resource_history.pop(0)
                
                # Notify listeners
                for listener in self.listeners:
                    try:
                        listener(metrics)
                    except Exception as e:
                        logger.error(f"Error in resource listener callback: {e}")
                
                # Check for threshold violations
                self._check_thresholds(metrics)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
            
            # Wait for next check interval or until stopped
            self._stop_event.wait(self.check_interval)
    
    def _calculate_rates(self, current_metrics: Dict[str, Any], 
                         previous_metrics: Dict[str, Any], 
                         time_diff: float) -> None:
        """
        Calculate rate metrics between two measurements.
        
        Args:
            current_metrics: Current resource metrics
            previous_metrics: Previous resource metrics
            time_diff: Time difference in seconds between measurements
        """
        # Network rates
        if 'network' in current_metrics and 'network' in previous_metrics:
            network = current_metrics['network']
            prev_network = previous_metrics['network']
            
            network['bytes_sent_per_sec'] = (network['bytes_sent'] - prev_network['bytes_sent']) / time_diff
            network['bytes_recv_per_sec'] = (network['bytes_recv'] - prev_network['bytes_recv']) / time_diff
            network['packets_sent_per_sec'] = (network['packets_sent'] - prev_network['packets_sent']) / time_diff
            network['packets_recv_per_sec'] = (network['packets_recv'] - prev_network['packets_recv']) / time_diff
            
            # Calculate per-interface rates
            for interface in network.get('interfaces', {}):
                if interface in prev_network.get('interfaces', {}):
                    current_if = network['interfaces'][interface]
                    prev_if = prev_network['interfaces'][interface]
                    
                    current_if['bytes_sent_per_sec'] = (current_if['bytes_sent'] - prev_if['bytes_sent']) / time_diff
                    current_if['bytes_recv_per_sec'] = (current_if['bytes_recv'] - prev_if['bytes_recv']) / time_diff
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """
        Check if any resource metrics exceed defined thresholds.
        
        Args:
            metrics: The resource metrics to check
        """
        alerts = []
        
        # Check CPU usage
        if metrics['cpu']['percent'] > self.cpu_threshold * 100:
            message = f"CPU usage threshold exceeded: {metrics['cpu']['percent']}%"
            logger.warning(message)
            alerts.append({
                'resource': 'cpu',
                'level': 'warning' if metrics['cpu']['percent'] < 95 else 'critical',
                'message': message,
                'value': metrics['cpu']['percent'],
                'threshold': self.cpu_threshold * 100,
                'timestamp': metrics['timestamp']
            })
        
        # Check memory usage
        if metrics['memory']['percent'] > self.memory_threshold * 100:
            message = f"Memory usage threshold exceeded: {metrics['memory']['percent']}%"
            logger.warning(message)
            alerts.append({
                'resource': 'memory',
                'level': 'warning' if metrics['memory']['percent'] < 95 else 'critical',
                'message': message,
                'value': metrics['memory']['percent'],
                'threshold': self.memory_threshold * 100,
                'timestamp': metrics['timestamp']
            })
        
        # Check disk usage
        for partition in metrics['disk']['partitions']:
            if partition['percent'] > self.disk_threshold * 100:
                message = f"Disk usage threshold exceeded on {partition['mountpoint']}: {partition['percent']}%"
                logger.warning(message)
                alerts.append({
                    'resource': 'disk',
                    'level': 'warning' if partition['percent'] < 95 else 'critical',
                    'message': message,
                    'value': partition['percent'],
                    'threshold': self.disk_threshold * 100,
                    'partition': partition['mountpoint'],
                    'timestamp': metrics['timestamp']
                })
        
        # Store alerts if any were generated
        if alerts:
            # Add all new alerts to the resource_alerts list
            self.resource_alerts.extend(alerts)
            
            # Enforce max alerts limit by removing oldest alerts if needed
            if len(self.resource_alerts) > self.max_alerts:
                self.resource_alerts = self.resource_alerts[-self.max_alerts:]
            
            # Notify about alerts
            self._send_alert_notifications(alerts)
    
    def _send_alert_notifications(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Send notifications for resource alerts.
        
        This method handles sending notifications to appropriate channels
        based on alert severity and type.
        
        Args:
            alerts: List of alert dictionaries to send notifications for
        """
        for alert in alerts:
            # Log the alert
            log_message = f"RESOURCE ALERT: {alert['level'].upper()} - {alert['message']}"
            
            if alert['level'] == 'critical':
                logger.critical(log_message)
                # In a real system, this might send emails, SMS, or trigger pager duty
                self._handle_critical_alert(alert)
            else:
                logger.warning(log_message)
                
    def _handle_critical_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle critical resource alerts that require immediate attention.
        
        Args:
            alert: The critical alert to handle
        """
        # In a production system, this could trigger:
        # 1. Emergency scaling operations
        # 2. Task priority adjustments
        # 3. External notification systems (email, SMS)
        
        # For now, we just log it and add extra metadata for escalation
        alert['escalated'] = True
        alert['escalation_time'] = time.time()
        
        # Here you would typically integrate with external notification systems
        logger.critical(f"ESCALATED: {alert['message']} - Immediate action required!")
        
    def get_recent_alerts(self, count: int = 10, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent resource alerts, optionally filtered by level.
        
        Args:
            count: Maximum number of alerts to return
            level: Optional filter for alert level ('warning' or 'critical')
            
        Returns:
            List of recent alert dictionaries
        """
        if level:
            filtered_alerts = [a for a in self.resource_alerts if a['level'] == level]
            return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)[:count]
        else:
            return sorted(self.resource_alerts, key=lambda x: x['timestamp'], reverse=True)[:count]

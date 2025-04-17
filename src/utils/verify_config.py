#!/usr/bin/env python3
"""
Configuration verification utility for the application.

This script checks various components of the application setup:
1. Configuration file structure and validity
2. Database connections and schema
3. Network connectivity and configurations
4. Directory structure and permissions

Usage:
    python verify_config.py [config_path]

Arguments:
    config_path: Optional path to the configuration file. 
                 If not provided, uses default location.

Exit codes:
    0: All checks passed
    1: Configuration file issues
    2: Database connection issues
    3: Network configuration issues
    4: Directory structure issues
    5: Other errors
"""

import os
import sys
import json
import socket
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ConfigVerifier')


class ConfigVerificationError(Exception):
    """Exception raised for configuration verification errors."""
    pass


class ConfigVerifier:
    """
    Utility class for verifying application configuration and setup.
    
    This class provides methods to check:
    - Configuration file existence and structure
    - Database connections and schema
    - Network configuration and connectivity
    - Directory structure and permissions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration verifier.
        
        Args:
            config_path: Optional path to the configuration file.
                        If not provided, uses default location.
        """
        self.config_path = config_path or os.path.join('config', 'default_config.json')
        self.config: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def verify_all(self) -> bool:
        """
        Run all verification checks.
        
        Returns:
            bool: True if all checks passed, False otherwise.
        """
        try:
            logger.info("Starting configuration verification...")
            
            # Check configuration file
            config_result = self.verify_config_file()
            if not config_result:
                return False
                
            # Check directory structure
            dir_result = self.verify_directory_structure()
            
            # Check database connection
            db_result = self.verify_database()
            
            # Check network configuration
            net_result = self.verify_network_config()
            
            # Report results
            all_passed = config_result and dir_result and db_result and net_result
            
            if all_passed:
                logger.info("All verification checks passed successfully!")
            else:
                logger.error("Verification failed with errors:")
                for error in self.errors:
                    logger.error(f" - {error}")
                    
                if self.warnings:
                    logger.warning("Warnings:")
                    for warning in self.warnings:
                        logger.warning(f" - {warning}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}")
            self.errors.append(f"Unexpected error: {str(e)}")
            return False
    
    def verify_config_file(self) -> bool:
        """
        Verify the configuration file exists and has valid structure.
        
        Returns:
            bool: True if configuration file is valid, False otherwise.
        """
        logger.info(f"Verifying configuration file: {self.config_path}")
        
        # Check if file exists
        if not os.path.exists(self.config_path):
            self.errors.append(f"Configuration file not found: {self.config_path}")
            logger.error(f"Configuration file not found: {self.config_path}")
            return False
            
        # Check if file is readable
        if not os.access(self.config_path, os.R_OK):
            self.errors.append(f"Configuration file is not readable: {self.config_path}")
            logger.error(f"Configuration file is not readable: {self.config_path}")
            return False
            
        # Try to parse JSON
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in configuration file: {e}")
            logger.error(f"Invalid JSON in configuration file: {e}")
            return False
            
        # Verify required sections
        required_sections = ['database', 'network', 'directories', 'application']
        missing_sections = [s for s in required_sections if s not in self.config]
        
        if missing_sections:
            self.errors.append(f"Missing required sections in config file: {', '.join(missing_sections)}")
            logger.error(f"Missing required sections in config file: {', '.join(missing_sections)}")
            return False
            
        logger.info("Configuration file verified successfully")
        return True
        
    def verify_directory_structure(self) -> bool:
        """
        Verify the application directory structure.
        
        Returns:
            bool: True if directory structure is valid, False otherwise.
        """
        if 'directories' not in self.config:
            self.errors.append("Missing 'directories' section in configuration")
            return False
            
        logger.info("Verifying directory structure...")
        
        directories = self.config['directories']
        all_valid = True
        
        required_dirs = ['data', 'logs', 'blockchain', 'temp']
        
        # Check that all required directories are specified
        for dir_name in required_dirs:
            if dir_name not in directories:
                self.errors.append(f"Missing required directory in config: {dir_name}")
                all_valid = False
                
        # Check if directories exist and are accessible
        for dir_name, dir_path in directories.items():
            dir_obj = Path(dir_path)
            
            if not dir_obj.exists():
                try:
                    # Try to create directory if it doesn't exist
                    logger.info(f"Creating directory: {dir_path}")
                    dir_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.errors.append(f"Failed to create directory {dir_path}: {e}")
                    all_valid = False
                    continue
            
            # Check if directory is writable
            if not os.access(dir_path, os.W_OK):
                self.errors.append(f"Directory not writable: {dir_path}")
                all_valid = False
                
            # Check if directory is readable
            if not os.access(dir_path, os.R_OK):
                self.errors.append(f"Directory not readable: {dir_path}")
                all_valid = False
                
        if all_valid:
            logger.info("Directory structure verified successfully")
        else:
            logger.error("Directory structure verification failed")
            
        return all_valid
        
    def verify_database(self) -> bool:
        """
        Verify database connection and schema.
        
        Returns:
            bool: True if database connection is successful, False otherwise.
        """
        if 'database' not in self.config:
            self.errors.append("Missing 'database' section in configuration")
            return False
            
        logger.info("Verifying database connection...")
        
        db_config = self.config['database']
        
        # Check required database configuration
        if 'engine' not in db_config:
            self.errors.append("Missing 'engine' in database configuration")
            return False
            
        if 'path' not in db_config and db_config['engine'] == 'sqlite':
            self.errors.append("Missing 'path' in SQLite database configuration")
            return False
            
        # Test database connection based on engine
        engine = db_config['engine'].lower()
        
        if engine == 'sqlite':
            return self._verify_sqlite_database(db_config)
        elif engine in ('mysql', 'postgresql'):
            self.warnings.append(f"Verification for {engine} not implemented")
            return True
        else:
            self.errors.append(f"Unsupported database engine: {engine}")
            return False
            
    def _verify_sqlite_database(self, db_config: Dict[str, Any]) -> bool:
        """
        Verify SQLite database connection and structure.
        
        Args:
            db_config: The database configuration dictionary.
            
        Returns:
            bool: True if SQLite database is valid, False otherwise.
        """
        db_path = db_config['path']
        db_dir = os.path.dirname(db_path)
        
        # Check if database directory exists
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                self.errors.append(f"Failed to create database directory {db_dir}: {e}")
                return False
                
        # Try to connect to database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Verify schema if required
            required_tables = db_config.get('required_tables', [])
            
            if required_tables:
                # Get existing tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                # Check if required tables exist
                missing_tables = [table for table in required_tables if table not in existing_tables]
                
                if missing_tables:
                    self.warnings.append(f"Missing required tables: {', '.join(missing_tables)}")
                    logger.warning(f"Missing required tables: {', '.join(missing_tables)}")
                    
            conn.close()
            logger.info("SQLite database verified successfully")
            return True
            
        except sqlite3.Error as e:
            self.errors.append(f"SQLite database error: {e}")
            logger.error(f"SQLite database error: {e}")
            return False
            
    def verify_network_config(self) -> bool:
        """
        Verify network configuration and connectivity.
        
        Returns:
            bool: True if network configuration is valid, False otherwise.
        """
        if 'network' not in self.config:
            self.errors.append("Missing 'network' section in configuration")
            return False
            
        logger.info("Verifying network configuration...")
        
        network_config = self.config['network']
        all_valid = True
        
        # Check required network configuration
        if 'host' not in network_config:
            self.errors.append("Missing 'host' in network configuration")
            all_valid = False
            
        if 'port' not in network_config:
            self.errors.append("Missing 'port' in network configuration")
            all_valid = False
            
        if not all_valid:
            return False
            
        # Verify port is valid
        try:
            port = int(network_config['port'])
            if port < 0 or port > 65535:
                self.errors.append(f"Invalid port number: {port}")
                all_valid = False
        except (ValueError, TypeError):
            self.errors.append(f"Invalid port value: {network_config['port']}")
            all_valid = False
            
        # Test if specified port is available
        if all_valid and network_config.get('test_port_availability', True):
            try:
                host = network_config['host']
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2)
                
                # Check if we can bind to this port (it's available)
                try:
                    test_socket.bind((host, port))
                    logger.info(f"Port {port} is available")
                except OSError:
                    self.warnings.append(f"Port {port} is already in use")
                    logger.warning(f"Port {port} is already in use")
                    
                test_socket.close()
                
            except socket.error as e:
                self.errors.append(f"Network socket error: {e}")
                all_valid = False
                
        # Test internet connectivity if required
        if all_valid and network_config.get('test_internet', False):
            test_hosts = network_config.get('test_hosts', ['8.8.8.8', '1.1.1.1'])
            connected = False
            
            for test_host in test_hosts:
                try:
                    # Try to connect to a reliable external host
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(3)
                    test_socket.connect((test_host, 53))  # DNS port
                    test_socket.close()
                    connected = True
                    break
                except socket.error:
                    continue
                    
            if not connected:
                self.warnings.append("Internet connectivity test failed")
                logger.warning("Internet connectivity test failed")
                
        if all_valid:
            logger.info("Network configuration verified successfully")
            
        return all_valid


def main():
    """
    Main function to run the configuration verification.
    
    Returns:
        int: Exit code
    """
    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    # Create the verifier
    verifier = ConfigVerifier(config_path)
    
    try:
        # Run verification
        if verifier.verify_config_file():
            # Continue with other verifications
            dir_result = verifier.verify_directory_structure()
            db_result = verifier.verify_database()
            net_result = verifier.verify_network_config()
            
            # Determine exit code
            if not dir_result:
                return 4
            elif not db_result:
                return 2
            elif not net_result:
                return 3
            else:
                return 0
        else:
            return 1
            
    except ConfigVerificationError as e:
        logger.error(f"Verification error: {e}")
        return 5
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())


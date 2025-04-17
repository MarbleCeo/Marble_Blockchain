"""
Initialize Configuration for P2P Blockchain application.

This script handles first-time setup of the application configuration,
creating the necessary directories and files, and guiding the user
through initial configuration.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config_manager import ConfigManager, ConfigError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'config'
)
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, 'default_config.json')

USER_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.p2p_blockchain')
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, 'config.json')

def create_user_directories() -> None:
    """
    Create necessary directories for user configuration and data.
    
    Creates:
        - User configuration directory
        - Data directory for blockchain database
        - Log directory
    """
    directories = [
        USER_CONFIG_DIR,
        os.path.join(USER_CONFIG_DIR, 'data'),
        os.path.join(USER_CONFIG_DIR, 'logs'),
        os.path.join(USER_CONFIG_DIR, 'backups')
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

def prompt_for_custom_settings() -> Dict[str, Any]:
    """
    Prompt the user for custom configuration settings.
    
    Returns:
        Dictionary with user-specified configuration values
    """
    custom_config = {}
    
    print("\n===== P2P Blockchain Application - Initial Setup =====\n")
    print("Press Enter to accept default values or enter custom values.\n")
    
    # Basic application settings
    nickname = input("Enter your nickname (default 'User'): ").strip()
    if nickname:
        custom_config['user.default_nickname'] = nickname
    
    # Network settings
    port = input("Enter TCP port for P2P connections (default 0 for random): ").strip()
    if port and port.isdigit():
        custom_config['network.tcp_port'] = int(port)
    
    # Blockchain settings
    difficulty = input("Enter mining difficulty (default 4): ").strip()
    if difficulty and difficulty.isdigit():
        custom_config['blockchain.difficulty'] = int(difficulty)
    
    mining_reward = input("Enter mining reward (default 50.0): ").strip()
    if mining_reward:
        try:
            custom_config['blockchain.mining_reward'] = float(mining_reward)
        except ValueError:
            logger.warning(f"Invalid mining reward value: {mining_reward}, using default")
    
    # GUI settings
    theme = input("Enter GUI theme (system/light/dark, default 'system'): ").strip().lower()
    if theme in ['system', 'light', 'dark']:
        custom_config['gui.theme'] = theme

    return custom_config

def load_and_customize_config(default_config_path: str, custom_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the default configuration and apply user customizations.
    
    Args:
        default_config_path: Path to the default configuration file
        custom_settings: Dictionary with user-specified settings
        
    Returns:
        Combined configuration dictionary
        
    Raises:
        ConfigError: If the default configuration cannot be loaded
    """
    try:
        with open(default_config_path, 'r') as f:
            config = json.load(f)
            
        # Apply custom settings using dot notation
        for path, value in custom_settings.items():
            parts = path.split('.')
            target = config
            
            # Navigate to the parent of the target field
            for i, part in enumerate(parts[:-1]):
                if part not in target:
                    target[part] = {}
                target = target[part]
                
            # Set the value
            target[parts[-1]] = value
            
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error loading default configuration: {e}")
        raise ConfigError(f"Invalid default configuration file: {e}")
    except (OSError, IOError) as e:
        logger.error(f"Error accessing default configuration: {e}")
        raise ConfigError(f"Could not read default configuration: {e}")

def save_user_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save the configuration to the user's config file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration
        
    Raises:
        ConfigError: If the configuration cannot be saved
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    except (OSError, IOError) as e:
        logger.error(f"Error saving configuration: {e}")
        raise ConfigError(f"Could not save configuration: {e}")

def verify_configuration(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify that the configuration is valid and contains all required fields.
    
    Args:
        config: Configuration dictionary to verify
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Create a temporary ConfigManager to validate the configuration
    try:
        # Write config to a temporary file for validation
        temp_path = os.path.join(USER_CONFIG_DIR, 'temp_config.json')
        with open(temp_path, 'w') as f:
            json.dump(config, f)
            
        # Use ConfigManager to validate
        config_manager = ConfigManager(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        return True, None
    except ConfigError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error during configuration validation: {e}"

def initialize_config(interactive: bool = True, custom_settings: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the application configuration.
    
    Creates necessary directories, loads default configuration,
    applies user customizations if interactive, and saves the 
    configuration to the user's config directory.
    
    Args:
        interactive: Whether to prompt for user input
        custom_settings: Pre-defined custom settings (used in non-interactive mode)
        
    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Create user directories
        create_user_directories()
        
        # Load default config
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            logger.error(f"Default configuration not found at {DEFAULT_CONFIG_PATH}")
            print(f"ERROR: Default configuration file not found at {DEFAULT_CONFIG_PATH}")
            return False
        
        # Get custom settings
        settings = {}
        if interactive:
            settings = prompt_for_custom_settings()
        elif custom_settings:
            settings = custom_settings
            
        # Load and customize configuration
        config = load_and_customize_config(DEFAULT_CONFIG_PATH, settings)
        
        # Verify configuration
        is_valid, error = verify_configuration(config)
        if not is_valid:
            logger.error(f"Invalid configuration: {error}")
            print(f"ERROR: Invalid configuration: {error}")
            return False
            
        # Save user configuration
        save_user_config(config, USER_CONFIG_PATH)
        
        logger.info("Configuration initialization completed successfully")
        print("\nConfiguration initialized successfully!")
        print(f"Configuration saved to: {USER_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error during configuration initialization: {e}")
        print(f"\nERROR: Failed to initialize configuration: {e}")
        return False

def parse_arguments():
    """
    Parse command-line arguments for the initialization script.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Initialize P2P Blockchain application configuration"
    )
    
    parser.add_argument(
        "--non-interactive", 
        action="store_true", 
        help="Run in non-interactive mode without prompting for user input"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to a custom JSON file with configuration to use instead of prompting"
    )
    
    parser.add_argument(
        "--nickname", 
        type=str, 
        help="Set the default nickname"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        help="Set the TCP port for P2P connections"
    )
    
    parser.add_argument(
        "--difficulty", 
        type=int, 
        help="Set the mining difficulty"
    )
    
    parser.add_argument(
        "--mining-reward", 
        type=float, 
        help="Set the mining reward"
    )
    
    parser.add_argument(
        "--theme", 
        choices=["system", "light", "dark"],
        help="Set the GUI theme (system/light/dark)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the initialization script."""
    args = parse_arguments()
    
    # Handle non-interactive mode with a provided config file
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_settings = json.load(f)
            success = initialize_config(interactive=False, custom_settings=custom_settings)
            sys.exit(0 if success else 1)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading custom configuration file: {e}")
            print(f"ERROR: Could not load configuration from {args.config}: {e}")
            sys.exit(1)
    
    # Extract command-line arguments for non-interactive mode
    if args.non_interactive:
        custom_settings = {}
        
        if args.nickname:
            custom_settings['user.default_nickname'] = args.nickname
            
        if args.port is not None:
            custom_settings['network.tcp_port'] = args.port
            
        if args.difficulty is not None:
            custom_settings['blockchain.difficulty'] = args.difficulty
            
        if args.mining_reward is not None:
            custom_settings['blockchain.mining_reward'] = args.mining_reward
            
        if args.theme:
            custom_settings['gui.theme'] = args.theme
            
        if args.log_level:
            custom_settings['application.log_level'] = args.log_level
            
        success = initialize_config(interactive=False, custom_settings=custom_settings)
        sys.exit(0 if success else 1)
    
    # Interactive mode
    success = initialize_config(interactive=True)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

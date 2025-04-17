# Installation Guide

This guide will walk you through the installation process for the Blockchain P2P Network application.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- Internet connection for P2P functionality

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/blockchain-p2p.git
cd blockchain-p2p
```

### 2. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize Configuration

```bash
python -m src.utils.initialize_config
```

This will create default configuration files in the `config/` directory.

### 4. Verify Installation

Run the verification script to ensure all components are properly installed:

```bash
python -m src.utils.verify_installation
```

If successful, you should see:
```
✓ All dependencies installed
✓ Database connection successful
✓ Network connection available
✓ Configuration files initialized
```

## Troubleshooting

### Common Issues

#### Missing Dependencies

If you encounter errors related to missing packages:

```bash
pip install -r requirements.txt --upgrade
```

#### Database Initialization Failure

If the database fails to initialize:

```bash
# Remove existing database file
rm config/blockchain.db
# Reinitialize
python -m src.utils.initialize_config
```

#### Network Connection Issues

If you cannot connect to peers:

1. Check your internet connection
2. Verify firewall settings allow outbound connections
3. Try connecting to default bootstrap peers:
   ```bash
   python -m src.networking.test_connection
   ```

### Getting Help

If you continue to experience issues:

1. Check the [troubleshooting documentation](troubleshooting.md)
2. Open an issue on the GitHub repository
3. Contact support at support@blockchain-p2p.example.com


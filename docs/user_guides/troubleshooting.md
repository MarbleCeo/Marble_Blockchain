# Troubleshooting Guide

This guide helps you diagnose and fix common issues with the Blockchain P2P Network application.

## Common Issues and Solutions

### Application Won't Start

#### Symptoms:
- Application crashes immediately after launch
- Error messages about missing modules

#### Solutions:
1. Verify Python version (3.8+ required):
   ```bash
   python --version
   ```

2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. Check for conflicting packages:
   ```bash
   pip check
   ```

### Cannot Connect to Network

#### Symptoms:
- "Connection Failed" error messages
- Empty peers list
- Status indicator shows "Disconnected"

#### Solutions:
1. Check internet connection

2. Verify bootstrap peers are accessible:
   ```bash
   python -m src.utils.network_test
   ```

3. Check if ports are blocked by firewall:
   ```bash
   # On Linux/Mac
   nc -zv bootstrap-peer.example.com 4001
   
   # On Windows
   Test-NetConnection -ComputerName bootstrap-peer.example.com -Port 4001
   ```

4. Try connecting with explicit peers:
   ```bash
   python -m src.gui.clientgui_merged --peers "/ip4/public-peer.example.com/tcp/4001/p2p/QmPeerID"
   ```

### Blockchain Synchronization Issues

#### Symptoms:
- Transactions not appearing
- Balance incorrect
- Mining doesn't work

#### Solutions:
1. Reset local blockchain (will download from peers):
   ```bash
   python -m src.utils.blockchain_tools reset
   ```

2. Force resynchronization:
   ```bash
   python -m src.utils.blockchain_tools sync --force
   ```

3. Check blockchain integrity:
   ```bash
   python -m src.utils.blockchain_tools validate
   ```

### GUI Glitches

#### Symptoms:
- Interface freezes
- Elements not rendering properly
- Input fields not responding

#### Solutions:
1. Restart application with debug logging:
   ```bash
   python -m src.gui.clientgui_merged --debug
   ```

2. Clear cached UI settings:
   ```bash
   python -m src.utils.clear_settings
   ```

3. Update PyQt6:
   ```bash
   pip install PyQt6 PyQt6-QScintilla --upgrade
   ```

## Advanced Troubleshooting

### Database Issues

To repair corrupted database:
```bash
python -m src.utils.database_repair
```

To


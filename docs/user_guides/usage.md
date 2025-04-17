# Usage Guide

This guide explains how to use the Blockchain P2P Network application.

## Starting the Application

### GUI Application

To start the graphical user interface:

```bash
python -m src.gui.clientgui_merged
```

### Command Line Client

For terminal-based usage:

```bash
python -m src.networking.client_merged --nickname YourName
```

## GUI Overview

The application interface consists of three main tabs:

1. **Chat** - For sending and receiving messages with connected peers
2. **Transactions** - For creating and monitoring blockchain transactions
3. **Network** - For managing peer connections and network status

### Chat Tab

- **Send Message**: Type your message in the input field and click "Send"
- **Message History**: View all messages from connected peers
- **Nickname**: Change your display name by clicking "Set Nickname"

### Transactions Tab

- **New Transaction**: Create a transaction by filling in recipient and amount
- **Transaction History**: View all transactions on the blockchain
- **Balance**: See your current account balance
- **Mining**: Start/stop mining to earn rewards

### Network Tab

- **Connect**: Connect to the network using bootstrap peers
- **Disconnect**: Disconnect from the network
- **Peers List**: View currently connected peers
- **Connection Status**: Monitor network connection status

## Blockchain Operations

### Creating Transactions

1. Navigate to the "Transactions" tab
2. Enter the recipient's address
3. Enter the amount to send
4. Click "Send Transaction"

### Mining

1. Navigate to the "Transactions" tab
2. Click "Start Mining"
3. The application will mine pending transactions
4. Mining rewards will be credited to your account

### Viewing Transaction History

1. Navigate to the "Transactions" tab
2. The transaction history panel shows all confirmed transactions
3. Your current balance is displayed at the top

## Network Management

### Connecting to Peers

1. Navigate to the "Network" tab
2. Enter bootstrap peer addresses (or use defaults)
3. Click "Connect"

### Managing Connections

- The peers list shows all connected peers
- Connection status indicator shows network health
- Automatic reconnection attempts are made if connection is lost

## Advanced Features

### Custom Bootstrap Peers

To connect to specific bootstrap peers:

```bash
python -m src.gui.clientgui_merged --peers "/ip4/127.0.0.1/tcp/4001/p2p/PeerID1" "/ip4/127.0.0.1/tcp/4002/p2p/PeerID2"
```

### Running a Full Node

To run a dedicated node without GUI:

```bash
python -m src.networking.server_merged --port 4001
```

### Blockchain Export/Import

Export blockchain to file:
```bash
python -m src.utils.blockchain_tools export blockchain.json
```

Import blockchain from file:
```bash
python -m src.utils.blockchain_tools import blockchain.json
```


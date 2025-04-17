# Architecture Overview

## System Components

The blockchain P2P network consists of four main components:

1. **Blockchain Core** - Handles blockchain operations, transaction validation, and mining
2. **P2P Network Layer** - Manages peer discovery, message routing, and connection maintenance
3. **Client Logic** - Bridges the UI and network layers, handles message formatting and processing
4. **GUI Interface** - Provides user interaction capabilities through a PyQt6 interface

## Component Interactions

```
┌─────────────┐         ┌───────────────┐         ┌─────────────┐
│    GUI      │◄───────►│ Client Logic  │◄───────►│  Network    │
│  Interface  │         │               │         │    Layer    │
└─────────────┘         └───────┬───────┘         └──────┬──────┘
                                │                         │
                                ▼                         ▼
                        ┌───────────────┐         ┌───────────────┐
                        │  Blockchain   │         │ Peer Discovery│
                        │     Core      │         │ & Connection  │
                        └───────────────┘         └───────────────┘
```

## Data Flow

1. User interacts with GUI to send messages or create transactions
2. Client logic processes input and formats appropriate messages
3. Network layer distributes messages to connected peers
4. Receiving clients process incoming messages and update UI
5. Blockchain transactions are validated and added to pending transactions
6. Mining operations create new blocks and update the blockchain
7. Database persistence ensures blockchain state is maintained

## Technical Details

### Blockchain Implementation

- Block structure includes:
  - Index
  - Timestamp
  - Previous hash
  - Proof of work
  - Transactions list

- Transaction structure includes:
  - Sender
  - Recipient
  - Amount
  - Timestamp
  - Signature

### P2P Network

- Uses libp2p for peer discovery and connection
- Implements custom message protocol for blockchain operations
- Handles NAT traversal for wide-area networking
- Implements heartbeat for connection maintenance

### Database Schema

```
Blocks Table:
- id (INTEGER PRIMARY KEY)
- index (INTEGER)
- timestamp (REAL)
- previous_hash (TEXT)
- proof (INTEGER)
- data (TEXT) - JSON serialized transactions

Transactions Table:
- id (INTEGER PRIMARY KEY)
- sender (TEXT)
- recipient (TEXT)
- amount (REAL)
- timestamp (REAL)
- signature (TEXT)
- block_id (INTEGER) - Foreign key to blocks
```


# Marble Blockchain

A high-performance blockchain implementation with integrated DEX (Decentralized Exchange) and cross-bridge support.

## Overview

Marble Blockchain is a cutting-edge blockchain platform that combines the benefits of traditional blockchain technology with advanced features like:

- Built-in Decentralized Exchange (DEX)
- Cross-chain bridge capabilities
- AI-powered block analysis
- Parallel validation mechanisms
- Interactive command-line interface

The project consists of a Python-based blockchain core, a FastAPI backend, and a modified Raydium V3 frontend for the DEX.

## Installation and Setup

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm 8+

### Backend Setup

1. Clone the repository
   ```
   git clone https://github.com/yourusername/marble-blockchain.git
   cd marble-blockchain
   ```

2. Install Python dependencies
   ```
   pip install fastapi uvicorn pydantic asyncio cmd2 deepseek-ai
   ```

3. Run the FastAPI backend
   ```
   uvicorn marble_dex_app:app --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory
   ```
   cd raydium-ui-master
   ```

2. Install dependencies
   ```
   npm install
   ```

3. Start the frontend development server
   ```
   npm start
   ```
   
The frontend will be available at `http://localhost:3000`

## CLI Usage Guide

The Marble CLI provides an interactive shell to interact with the blockchain and DEX.

### Starting the CLI

```
python marble_cli.py
```

### Available Commands

#### Start the Blockchain
Initialize the blockchain and P2P node.
```
marble> start
Blockchain initialized. P2P node started.
```

#### Send Tokens
Add a transaction to the blockchain.
```
marble> send 0x1a2b3c4d5e6f 100 MTOKEN
Transaction added to the blockchain.
Transaction ID: 0x7890abcdef1234
```

#### Check Balance
View token balance for an address.
```
marble> balance 0x1a2b3c4d5e6f
Address: 0x1a2b3c4d5e6f
MTOKEN: 500
ETH: 2.5
USDC: 1000
```

#### Analyze Block
Use DeepSeek AI to analyze a block.
```
marble> analyze 15
AI Analysis Result for Block #15:
- Valid structure: Yes
- Suspicious transactions: None
- Performance metrics: 120 TPS
- Optimization recommendations: Increase block size
```

#### Validate Block
Trigger parallel validation of a block.
```
marble> validate 15 10
Validating Block #15 with 10 parallel processes...
Validation successful. Block confirmed.
```

#### Check Status
View blockchain and DEX status.
```
marble> status
Blockchain Status:
- Current Height: 26
- Pending Transactions: 8
- Network Nodes: 5

DEX Status:
- Active Trading Pairs: 12
- 24h Volume: 15,000 USDC
- Active Orders: 56
```

## Features

### Blockchain Core
- Secure transaction handling
- Block validation mechanisms
- Distributed consensus algorithm
- Token support for multiple assets

### Decentralized Exchange (DEX)
- Raydium V3-based trading interface
- Order book and liquidity pools
- Token swaps with minimal slippage
- Charting and trading analytics

### Cross-Chain Bridge
- Mock Solana bridge implementation
- Asset transfer between chains
- Transaction verification across bridges
- Bridge status monitoring

### AI Integration
- DeepSeek AI for block analysis
- Anomaly detection in transactions
- Performance optimization recommendations
- Security vulnerability scanning

### Validation Framework
- Parallel block validation
- Custom validation rules
- Resource allocation for validation
- Performance metrics tracking

## Architecture

The Marble Blockchain project follows a modular architecture:

1. **Core Layer** (`blockchain_core.py`)
   - Blockchain implementation
   - Transaction handling
   - Block validation logic

2. **API Layer** (`marble_dex_app.py`)
   - FastAPI endpoints for blockchain interaction
   - DEX functionality
   - Bridge operations
   - AI analysis integration

3. **Frontend** (Modified Raydium V3)
   - User interface for DEX operations
   - Wallet connection
   - Trading functionality
   - Block exploration

4. **CLI** (`marble_cli.py`)
   - Command-line interface
   - Interactive shell for blockchain operations
   - Administrative functions

## Development Environment Setup

### Required Tools
- Visual Studio Code or PyCharm
- Git
- Postman (for API testing)
- Python 3.9+
- Node.js 16+

### Development Workflow
1. Clone the repository
2. Set up virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies
   ```
   pip install -r requirements-dev.txt
   ```
4. Run tests
   ```
   pytest
   ```
5. For frontend development, use hot reload
   ```
   cd raydium-ui-master
   npm run dev
   ```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Nuxt Minimal Starter

Look at the [Nuxt documentation](https://nuxt.com/docs/getting-started/introduction) to learn more.

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build

# bun
bun run build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview

# bun
bun run preview
```

Check out the [deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information.

// Base URL for all API calls
const BASE_URL = 'http://localhost:8000';

// Types for API responses
interface Block {
  id: string;
  timestamp: number;
  data: any;
  // Add more properties as needed
}

interface Transaction {
  id: string;
  amount: number;
  sender: string;
  recipient: string;
  // Add more properties as needed
}

interface BridgeRequest {
  token_mint: string;
  amount: number;
}

interface BridgeResponse {
  status: string;
  // Add more properties as needed
}

interface DexConfig {
  pools: any[];
  fees: any;
  version: string;
  // Add more properties as needed
}

interface ValidateBlockRequest {
  block_id: string;
  coins: number;
}

interface ValidateBlockResponse {
  status: string;
  power: number;
  // Add more properties as needed
}

interface BlockAnalysis {
  block_id: string;
  analysis: any;
  // Add more properties as needed
}

/**
 * Fetches all blocks from the blockchain
 * @returns Promise with blocks data
 */
// Using enhanced implementations below

/**
 * Fetches DEX configuration
 * @returns Promise with DEX configuration
 */
export const fetchDexConfig = async (): Promise<DexConfig> => {
  try {
    const response = await fetch(`${BASE_URL}/dex_config`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch DEX config: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching DEX config:', error);
    throw error;
  }
};

/**
 * Validates a block with provided coins
 * @param blockId The ID of the block to validate
 * @param coins The number of coins to use for validation
 * @returns Promise with validation result
 */
export const fetchValidateBlock = async (blockId: string, coins: number): Promise<ValidateBlockResponse> => {
  try {
    const response = await fetch(`${BASE_URL}/validate_block`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        block_id: blockId,
        coins: coins
      } as ValidateBlockRequest),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to validate block: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error validating block:', error);
    throw error;
  }
};

/**
 * Analyzes a specific block
 * @param blockId The ID of the block to analyze
 * @returns Promise with block analysis
 */
export const fetchAnalyzeBlock = async (blockId: string): Promise<BlockAnalysis> => {
  try {
    const response = await fetch(`${BASE_URL}/analyze_block/${blockId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to analyze block: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error analyzing block:', error);
    throw error;
  }
};

import { toast } from 'react-toastify';

// API endpoints
const API_BASE_URL = 'http://localhost:8000';
const BLOCKS_ENDPOINT = `${API_BASE_URL}/blocks`;
const TRANSACTIONS_ENDPOINT = `${API_BASE_URL}/transactions`;
const BRIDGE_ENDPOINT = `${API_BASE_URL}/bridge`;
const TRADE_ENDPOINT = `${API_BASE_URL}/trade`;
const STATUS_ENDPOINT = `${API_BASE_URL}/status`;

// Type definitions
export interface Block {
  index: number;
  timestamp: number;
  transactions: Transaction[];
  previous_hash: string;
  hash: string;
  proof: number;
}

export interface Transaction {
  sender: string;
  recipient: string;
  amount: number;
  token: string;
  timestamp: number;
  signature?: string;
}

export interface BridgeRequest {
  token_mint: string;
  amount: number;
}

export interface BridgeResponse {
  status: string;
  amount: number;
}

export interface TradeRequest {
  sender: string;
  recipient: string;
  amount: number;
  token: string;
}

export interface TradeResponse {
  status: string;
  token: string;
  amount: number;
}

export interface StatusResponse {
  status: string;
}

// Error handling
const handleApiError = (error: unknown, fallbackMessage: string): never => {
  console.error('API Error:', error);
  const errorMessage = error instanceof Error ? error.message : fallbackMessage;
  toast.error(errorMessage);
  throw new Error(errorMessage);
};

/**
 * Fetches the current blockchain status from the backend
 * @returns The status response from the API
 */
export const fetchStatus = async (): Promise<StatusResponse> => {
  try {
    const response = await fetch(STATUS_ENDPOINT);
    
    if (!response.ok) {
      throw new Error(`Status API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    return handleApiError(error, 'Failed to fetch blockchain status');
  }
};

/**
 * Fetches the current blocks from the blockchain
 * @returns An array of block objects
 */
export const fetchBlocks = async (): Promise<Block[]> => {
  try {
    const response = await fetch(BLOCKS_ENDPOINT);
    
    if (!response.ok) {
      throw new Error(`Blocks API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    return handleApiError(error, 'Failed to fetch blockchain blocks');
  }
};

/**
 * Fetches pending transactions from the blockchain
 * @returns An array of transaction objects
 */
export const fetchTransactions = async (): Promise<Transaction[]> => {
  try {
    const response = await fetch(TRANSACTIONS_ENDPOINT);
    
    if (!response.ok) {
      throw new Error(`Transactions API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    return handleApiError(error, 'Failed to fetch pending transactions');
  }
};

/**
 * Bridges tokens from Solana to Marble blockchain
 * @param token_mint The mint address of the token to bridge
 * @param amount The amount of tokens to bridge
 * @returns The bridge response from the API
 */
export const fetchBridge = async (token_mint: string, amount: number): Promise<BridgeResponse> => {
  try {
    if (!token_mint || token_mint.trim() === '') {
      throw new Error('Token mint address is required');
    }
    
    if (!amount || amount <= 0) {
      throw new Error('Amount must be greater than zero');
    }
    
    const response = await fetch(BRIDGE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token_mint, amount } as BridgeRequest),
    });
    
    if (!response.ok) {
      throw new Error(`Bridge API error: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    toast.success(`Successfully bridged ${amount} tokens to Marble`);
    return result;
  } catch (error) {
    return handleApiError(error, 'Failed to bridge tokens');
  }
};

/**
 * Executes a trade on the Marble blockchain
 * @param sender The address of the sender
 * @param recipient The address of the recipient
 * @param amount The amount of tokens to trade
 * @param token The token identifier (MT1 or MT2)
 * @returns The trade response from the API
 */
export const fetchTrade = async (
  sender: string,
  recipient: string,
  amount: number,
  token: string
): Promise<TradeResponse> => {
  try {
    // Validate inputs
    if (!sender || sender.trim() === '') {
      throw new Error('Sender address is required');
    }
    
    if (!recipient || recipient.trim() === '') {
      throw new Error('Recipient address is required');
    }
    
    if (!amount || amount <= 0) {
      throw new Error('Amount must be greater than zero');
    }
    
    if (!token || (token !== 'MT1' && token !== 'MT2')) {
      throw new Error('Invalid token: must be MT1 or MT2');
    }
    
    const response = await fetch(TRADE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sender, recipient, amount, token } as TradeRequest),
    });
    
    if (!response.ok) {
      throw new Error(`Trade API error: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    toast.success(`Successfully queued trade of ${amount} ${token}`);
    return result;
  } catch (error) {
    return handleApiError(error, 'Failed to execute trade');
  }
};


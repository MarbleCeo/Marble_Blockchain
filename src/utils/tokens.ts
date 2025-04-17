import { PublicKey } from '@solana/web3.js';

/**
 * Interface representing a token in the Marble DEX system
 */
export interface Token {
  name: string;
  symbol: string;
  mint: string;
  decimals: number;
  logoURI?: string;
}

/**
 * MT1 token definition with mock mint address
 */
export const MT1: Token = {
  name: 'Marble Token 1',
  symbol: 'MT1',
  mint: 'mock_mint_1',
  decimals: 6,
  logoURI: '/images/tokens/mt1.png'
};

/**
 * MT2 token definition with mock mint address
 */
export const MT2: Token = {
  name: 'Marble Token 2',
  symbol: 'MT2',
  mint: 'mock_mint_2',
  decimals: 6,
  logoURI: '/images/tokens/mt2.png'
};

/**
 * List of all available tokens
 */
export const TOKENS: Token[] = [MT1, MT2];

/**
 * Get a token by its mint address
 * @param mint The mint address of the token
 * @returns The token or undefined if not found
 */
export function getTokenByMint(mint: string): Token | undefined {
  return TOKENS.find(token => token.mint === mint);
}

/**
 * Get a token by its symbol
 * @param symbol The symbol of the token
 * @returns The token or undefined if not found
 */
export function getTokenBySymbol(symbol: string): Token | undefined {
  return TOKENS.find(token => token.symbol === symbol);
}

/**
 * Format token amount based on its decimals
 * @param amount Raw token amount
 * @param token Token object
 * @returns Formatted amount as string
 */
export function formatTokenAmount(amount: number, token: Token): string {
  return (amount / Math.pow(10, token.decimals)).toFixed(token.decimals);
}

/**
 * Convert display amount to raw amount
 * @param displayAmount Amount as seen by user
 * @param token Token object
 * @returns Raw amount as number
 */
export function toRawAmount(displayAmount: number, token: Token): number {
  return displayAmount * Math.pow(10, token.decimals);
}

/**
 * Create a public key from a mint string
 * @param mintStr Mint address as string
 * @returns PublicKey object
 */
export function mintToPublicKey(mintStr: string): PublicKey {
  try {
    return new PublicKey(mintStr);
  } catch (error) {
    console.error('Invalid mint address:', error);
    throw new Error(`Invalid mint address: ${mintStr}`);
  }
}

export interface Token {
  name: string;
  symbol: string;
  mint: string;
  decimals: number;
  logoURI?: string;
}

export const TOKENS: Record<string, Token> = {
  SOL: {
    name: 'Solana',
    symbol: 'SOL',
    mint: 'So11111111111111111111111111111111111111112',
    decimals: 9,
    logoURI: '/assets/tokens/sol.png'
  },
  USDC: {
    name: 'USD Coin',
    symbol: 'USDC',
    mint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    decimals: 6,
    logoURI: '/assets/tokens/usdc.png'
  },
  // Added Marble tokens
  MT1: {
    name: 'Marble Token 1',
    symbol: 'MT1',
    mint: 'YOUR_TOKEN_MINT_1',
    decimals: 6,
    logoURI: '/assets/tokens/mt1.png'
  },
  MT2: {
    name: 'Marble Token 2',
    symbol: 'MT2',
    mint: 'YOUR_TOKEN_MINT_2',
    decimals: 6,
    logoURI: '/assets/tokens/mt2.png'
  }
};

export function getTokenBySymbol(symbol: string): Token | undefined {
  return TOKENS[symbol];
}

export function getTokenByMint(mint: string): Token | undefined {
  return Object.values(TOKENS).find(token => token.mint === mint);
}

export const SUPPORTED_TOKEN_LIST = Object.values(TOKENS);


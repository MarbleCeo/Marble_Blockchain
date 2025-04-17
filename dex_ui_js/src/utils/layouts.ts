import { PublicKey } from '@solana/web3.js'
import { blob, struct, u8, nu64 } from 'buffer-layout'

export function getBigNumber(num: string | number): number {
  return parseFloat(num.toString())
}

// Basic account layout
export const ACCOUNT_LAYOUT = struct([
  // mint
  blob(32, 'mint'),
  // owner
  blob(32, 'owner'),
  // amount
  nu64('amount'),
  // delegate option
  u8('delegateOption'),
  // delegate
  blob(32, 'delegate'),
  // state
  u8('state'),
  // is native option
  u8('isNativeOption'),
  // is native
  nu64('isNative'),
  // delegated amount
  nu64('delegatedAmount'),
  // close authority option
  u8('closeAuthorityOption'),
  // close authority
  blob(32, 'closeAuthority')
])

// Basic mint layout
export const MINT_LAYOUT = struct([
  // mint authority option
  u8('mintAuthorityOption'),
  // mint authority
  blob(32, 'mintAuthority'),
  // supply
  nu64('supply'),
  // decimals
  u8('decimals'),
  // is initialized
  u8('isInitialized'),
  // freeze authority option
  u8('freezeAuthorityOption'),
  // freeze authority
  blob(32, 'freezeAuthority')
])

import { bool, publicKey, struct, u32, u64, u8 } from '@project-serum/borsh'

// https://github.com/solana-labs/solana-program-library/blob/master/token/js/client/token.js#L210
export const ACCOUNT_LAYOUT = struct([
  publicKey('mint'),
  publicKey('owner'),
  u64('amount'),
  u32('delegateOption'),
  publicKey('delegate'),
  u8('state'),
  u32('isNativeOption'),
  u64('isNative'),
  u64('delegatedAmount'),
  u32('closeAuthorityOption'),
  publicKey('closeAuthority')
])

export const MINT_LAYOUT = struct([
  u32('mintAuthorityOption'),
  publicKey('mintAuthority'),
  u64('supply'),
  u8('decimals'),
  bool('initialized'),
  u32('freezeAuthorityOption'),
  publicKey('freezeAuthority')
])

export function getBigNumber(num: any) {
  return num === undefined || num === null ? 0 : parseFloat(num.toString())
}

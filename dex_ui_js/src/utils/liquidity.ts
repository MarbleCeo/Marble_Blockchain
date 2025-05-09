import { Connection, PublicKey, Transaction, TransactionInstruction } from '@solana/web3.js'
import { Token } from '@solana/spl-token'
import { nu64, struct } from 'buffer-layout'
import { LIQUIDITY_POOL_PROGRAM_ID_V4, TOKEN_PROGRAM_ID } from './ids'
import { TokenAmount } from './safe-math'
import { LiquidityPoolInfo } from './pools'
import { findAssociatedTokenAddress, sendTransaction } from './web3'

export const AMM_INFO_LAYOUT = struct([
  nu64('status'),
  nu64('nonce'),
  nu64('orderNum'),
  nu64('depth'),
  nu64('coinDecimals'),
  nu64('pcDecimals'),
  nu64('state'),
  nu64('resetFlag'),
  nu64('minSize'),
  nu64('volMaxCutRatio'),
  nu64('amountWaveRatio'),
  nu64('coinLotSize'),
  nu64('pcLotSize'),
  nu64('minPriceMultiplier'),
  nu64('maxPriceMultiplier'),
  nu64('systemDecimalsValue')
])

export const AMM_INFO_LAYOUT_V4 = struct([
  nu64('status'),
  nu64('nonce'),
  nu64('orderNum'),
  nu64('depth'),
  nu64('coinDecimals'),
  nu64('pcDecimals'),
  nu64('state'),
  nu64('resetFlag'),
  nu64('fee'),
  nu64('minSize'),
  nu64('volMaxCutRatio'),
  nu64('amountWaveRatio'),
  nu64('coinLotSize'),
  nu64('pcLotSize'),
  nu64('minPriceMultiplier'),
  nu64('maxPriceMultiplier'),
  nu64('systemDecimalsValue'),
  nu64('minSeparateNumerator'),
  nu64('minSeparateDenominator'),
  nu64('tradeFeeNumerator'),
  nu64('tradeFeeDenominator'),
  nu64('pnlNumerator'),
  nu64('pnlDenominator'),
  nu64('swapFeeNumerator'),
  nu64('swapFeeDenominator')
])

export async function getOutAmount(
  connection: Connection,
  poolInfo: LiquidityPoolInfo,
  fromCoinMint: string,
  toCoinMint: string,
  amount: string,
  slippage: number
): Promise<TokenAmount | null> {
  const mintEquals = (fromCoinMint === poolInfo.coin.mintAddress && toCoinMint === poolInfo.pc.mintAddress) ||
                   (fromCoinMint === poolInfo.pc.mintAddress && toCoinMint === poolInfo.coin.mintAddress)
  if (!mintEquals) return null

  const { ammId } = poolInfo

  // Fetch pool state
  const ammAccount = await connection.getAccountInfo(new PublicKey(ammId))
  if (!ammAccount) return null

  const poolState = AMM_INFO_LAYOUT_V4.decode(ammAccount.data)
  
  // Calculate output amount based on pool state
  const fromAmount = new TokenAmount(amount)
  const supply = new TokenAmount(poolState.supply.toString())
  const reserves = new TokenAmount(poolState.coinReserves.toString())
  
  if (fromAmount.gt(reserves)) return null

  const outputAmount = fromAmount
    .mul(supply)
    .div(reserves)
    .mul(new TokenAmount((1 - slippage).toString()))

  return outputAmount
}

export async function addLiquidity(
  connection: Connection,
  wallet: any,
  poolInfo: LiquidityPoolInfo,
  fromCoinAccount: string, 
  toCoinAccount: string,
  fromAmount: string,
  toAmount: string
): Promise<string> {
  const transaction = new Transaction()
  const signers: any[] = []

  const userCoinTokenAccount = new PublicKey(fromCoinAccount)
  const userPcTokenAccount = new PublicKey(toCoinAccount)
  const ammId = new PublicKey(poolInfo.ammId)
  const poolCoinTokenAccount = new PublicKey(poolInfo.poolCoinTokenAccount)
  const poolPcTokenAccount = new PublicKey(poolInfo.poolPcTokenAccount)
  const lpMintAddress = new PublicKey(poolInfo.lp.mintAddress)

  const userLpTokenAccount = await findAssociatedTokenAddress(
    wallet.publicKey,
    lpMintAddress
  )

  // Add instructions to transaction
  transaction.add(
    // Transfer tokens to pool
    Token.createTransferInstruction(
      TOKEN_PROGRAM_ID,
      userCoinTokenAccount,
      poolCoinTokenAccount,
      wallet.publicKey,
      [],
      fromAmount
    ),
    Token.createTransferInstruction(
      TOKEN_PROGRAM_ID,
      userPcTokenAccount,
      poolPcTokenAccount,
      wallet.publicKey,
      [],
      toAmount
    ),
    // Mint LP tokens to user
    Token.createMintToInstruction(
      TOKEN_PROGRAM_ID,
      lpMintAddress,
      userLpTokenAccount,
      wallet.publicKey,
      [],
      fromAmount
    )
  )

  return await sendTransaction(connection, wallet, transaction, signers)
}
146|

export async function getOutAmountStable(
  poolInfo: LiquidityPoolInfo,
  fromCoinMint: string, 
  toCoinMint: string,
  amount: string,
  slippage: number
): Promise<TokenAmount | null> {
  if (poolInfo.version !== 5) return null

  const mintEquals = (fromCoinMint === poolInfo.coin.mintAddress && toCoinMint === poolInfo.pc.mintAddress) ||
                  (fromCoinMint === poolInfo.pc.mintAddress && toCoinMint === poolInfo.coin.mintAddress)
  if (!mintEquals) return null

  const fromAmount = new TokenAmount(amount)
  const multiplier = new TokenAmount((1 - slippage).toString())
  
  // Simplified stable swap calculation
  return fromAmount.mul(multiplier)
}

export async function removeLiquidity(
  connection: Connection,
  wallet: any,
  poolInfo: LiquidityPoolInfo,
  lpAccount: string,
  coinAccount: string,
  pcAccount: string,
  amount: string
): Promise<string> {
  const transaction = new Transaction()
  const signers: any[] = []

  const userLpTokenAccount = new PublicKey(lpAccount)
  const userCoinTokenAccount = new PublicKey(coinAccount)
  const userPcTokenAccount = new PublicKey(pcAccount)
  const poolCoinTokenAccount = new PublicKey(poolInfo.poolCoinTokenAccount)
  const poolPcTokenAccount = new PublicKey(poolInfo.poolPcTokenAccount)
  const ammId = new PublicKey(poolInfo.ammId)

  // Add instructions to transaction
  transaction.add(
    // Burn LP tokens
    Token.createBurnInstruction(
      TOKEN_PROGRAM_ID,
      userLpTokenAccount,
      wallet.publicKey,
      [],
      amount
    ),
    // Transfer tokens from pool to user
    Token.createTransferInstruction(
      TOKEN_PROGRAM_ID,
      poolCoinTokenAccount,
      userCoinTokenAccount,
      ammId,
      [],
      amount
    ),
    Token.createTransferInstruction(
      TOKEN_PROGRAM_ID,
      poolPcTokenAccount,
      userPcTokenAccount,
      ammId,
      [],
      amount
    )
  )

  return await sendTransaction(connection, wallet, transaction, signers)
}

import { publicKey, u128, u64 } from '@project-serum/borsh'
import { closeAccount } from '@project-serum/serum/lib/token-instructions'
import { Connection, PublicKey, Transaction, TransactionInstruction } from '@solana/web3.js'
import BigNumber from 'bignumber.js'
// @ts-ignore
import { nu64, struct, u8, seq } from 'buffer-layout'

import { TOKEN_PROGRAM_ID } from '@/utils/ids'
import {
  getLpMintByTokenMintAddresses,
  getPoolByLpMintAddress,
  getPoolByTokenMintAddresses,
  LIQUIDITY_POOLS,
  LiquidityPoolInfo
} from '@/utils/pools'
import { TokenAmount } from '@/utils/safe-math'
import { LP_TOKENS, NATIVE_SOL, TokenInfo, TOKENS } from '@/utils/tokens'
import {
  commitment,
  createAssociatedTokenAccountIfNotExist,
  createTokenAccountIfNotExist,
  getMultipleAccounts,
  sendTransaction
} from '@/utils/web3'
import { getBigNumber, MINT_LAYOUT } from './layouts'
import { getStablePrice } from './stable'

export { getLpMintByTokenMintAddresses, getPoolByLpMintAddress, getPoolByTokenMintAddresses }

export function getPrice(poolInfo: LiquidityPoolInfo, coinBase = true) {
  const { coin, pc } = poolInfo

  if (!coin.balance || !pc.balance) {
    return new BigNumber(0)
  }

  if (poolInfo.version === 5) {
    if (!poolInfo.modelData) return new BigNumber(0)
    const x = poolInfo.coin.balance?.toEther()
    const y = poolInfo.pc.balance?.toEther()
    if (!x || !y) return new BigNumber(0)

    if (coinBase) {
      return getStablePrice(poolInfo.modelData, x.toNumber(), y.toNumber(), true)
    } else {
      return getStablePrice(poolInfo.modelData, x.toNumber(), y.toNumber(), false)
    }
  } else if (coinBase) {
    return pc.balance.toEther().dividedBy(coin.balance.toEther())
  } else {
    return coin.balance.toEther().dividedBy(pc.balance.toEther())
  }
}

export function getOutAmount(
  poolInfo: LiquidityPoolInfo,
  amount: string,
  fromCoinMint: string,
  toCoinMint: string,
  slippage: number
) {
  const { coin, pc } = poolInfo

  const price = getPrice(poolInfo)
  const fromAmount = new BigNumber(amount)
  let outAmount = new BigNumber(0)

  const percent = new BigNumber(100).plus(slippage).dividedBy(100)

  if (!coin.balance || !pc.balance) {
    return { outAmount, outMinAmount: outAmount }
  }

  if (fromCoinMint === coin.mintAddress && toCoinMint === pc.mintAddress) {
    // outcoin is pc
    outAmount = fromAmount.multipliedBy(price)
  } else if (fromCoinMint === pc.mintAddress && toCoinMint === coin.mintAddress) {
    // outcoin is coin
    outAmount = fromAmount.dividedBy(price)
  }

  return {
    outAmount: outAmount.multipliedBy(percent),
    outMinAmount: outAmount
  }
}

export function getOutAmountStable(
  poolInfo: any,
  amount: string,
  fromCoinMint: string,
  toCoinMint: string,
  slippage: number
) {
  const { coin, pc } = poolInfo

  const x = poolInfo.coin.balance?.toEther()
  const y = poolInfo.pc.balance?.toEther()
  if (!x || !y) return { outAmount: new BigNumber(0), outMinAmount: new BigNumber(0) }

  const price = y.dividedBy(x).toNumber()
  //  getStablePrice(currentK.toNumber(), x.toNumber(), y.toNumber(), true)
  const fromAmount = new BigNumber(amount)
  let outAmount = new BigNumber(0)

  const percent = new BigNumber(100).plus(slippage).dividedBy(100)

  if (!coin.balance || !pc.balance) {
    return { outAmount, outMinAmount: outAmount }
  }

  if (fromCoinMint === coin.mintAddress && toCoinMint === pc.mintAddress) {
    // outcoin is pc
    outAmount = fromAmount.multipliedBy(price)
  } else if (fromCoinMint === pc.mintAddress && toCoinMint === coin.mintAddress) {
    // outcoin is coin
    outAmount = fromAmount.dividedBy(price)
  }

  return {
    outAmount: outAmount.multipliedBy(percent),
    outMinAmount: outAmount
  }
}

/* eslint-disable */
export async function addLiquidity(
  connection: Connection | undefined | null,
  wallet: any | undefined | null,
  poolInfo: LiquidityPoolInfo | undefined | null,
  fromCoinAccount: string | undefined | null,
  toCoinAccount: string | undefined | null,
  lpAccount: string | undefined | null,
  fromCoin: TokenInfo | undefined | null,
  toCoin: TokenInfo | undefined | null,
  fromAmount: string | undefined | null,
  toAmount: string | undefined | null,
  fixedCoin: string
): Promise<string> {
  if (!connection || !wallet) throw new Error('Miss connection')
  if (!poolInfo || !fromCoin || !toCoin) {
    throw new Error('Miss pool infomations')
  }
  if (!fromCoinAccount || !toCoinAccount) {
    throw new Error('Miss account infomations')
  }
  if (!fromAmount || !toAmount) {
    throw new Error('Miss amount infomations')
  }

  const transaction = new Transaction()
  const signers: any = []

  const owner = wallet.publicKey

  const userAccounts = [new PublicKey(fromCoinAccount), new PublicKey(toCoinAccount)]
  const userAmounts = [fromAmount, toAmount]

  if (poolInfo.coin.mintAddress === toCoin.mintAddress && poolInfo.pc.mintAddress === fromCoin.mintAddress) {
    userAccounts.reverse()
    userAmounts.reverse()
  }

  const userCoinTokenAccount = userAccounts[0]
  const userPcTokenAccount = userAccounts[1]
  const coinAmount = getBigNumber(new TokenAmount(userAmounts[0], poolInfo.coin.decimals, false).wei)
  const pcAmount = getBigNumber(new TokenAmount(userAmounts[1], poolInfo.pc.decimals, false).wei)

  let wrappedCoinSolAccount
  if (poolInfo.coin.mintAddress === NATIVE_SOL.mintAddress) {
    wrappedCoinSolAccount = await createTokenAccountIfNotExist(
      connection,
      wrappedCoinSolAccount,
      owner,
      TOKENS.WSOL.mintAddress,
      coinAmount + 1e7,
      transaction,
      signers
    )
  }
  let wrappedSolAccount
  if (poolInfo.pc.mintAddress === NATIVE_SOL.mintAddress) {
    wrappedSolAccount = await createTokenAccountIfNotExist(
      connection,
      wrappedSolAccount,
      owner,
      TOKENS.WSOL.mintAddress,
      pcAmount + 1e7,
      transaction,
      signers
    )
  }

  let userLpTokenAccount = await createAssociatedTokenAccountIfNotExist(
    lpAccount,
    owner,
    poolInfo.lp.mintAddress,
    transaction
  )

  transaction.add(
    poolInfo.version === 5
      ? addLiquidityInstructionStable(
        new PublicKey(poolInfo.programId),
        new PublicKey(poolInfo.ammId),
        new PublicKey(poolInfo.ammAuthority),
        new PublicKey(poolInfo.ammOpenOrders),
        new PublicKey(poolInfo.ammTargetOrders),
        new PublicKey(poolInfo.lp.mintAddress),
        new PublicKey(poolInfo.poolCoinTokenAccount),
        new PublicKey(poolInfo.poolPcTokenAccount),
        new PublicKey(poolInfo.modelDataAccount ?? ''),
        new PublicKey(poolInfo.serumMarket),
        wrappedCoinSolAccount ? wrappedCoinSolAccount : userCoinTokenAccount,
        wrappedSolAccount ? wrappedSolAccount : userPcTokenAccount,
        userLpTokenAccount,
        owner,

        coinAmount,
        pcAmount,
        fixedCoin === poolInfo.coin.mintAddress ? 0 : 1
      )
      : poolInfo.version === 4
        ? addLiquidityInstructionV4(
          new PublicKey(poolInfo.programId),

          new PublicKey(poolInfo.ammId),
          new PublicKey(poolInfo.ammAuthority),
          new PublicKey(poolInfo.ammOpenOrders),
          new PublicKey(poolInfo.ammTargetOrders),
          new PublicKey(poolInfo.lp.mintAddress),
          new PublicKey(poolInfo.poolCoinTokenAccount),
          new PublicKey(poolInfo.poolPcTokenAccount),

          new PublicKey(poolInfo.serumMarket),
          new PublicKey(poolInfo.serumEventQueue!),

          wrappedCoinSolAccount ? wrappedCoinSolAccount : userCoinTokenAccount,
          wrappedSolAccount ? wrappedSolAccount : userPcTokenAccount,
          userLpTokenAccount,
          owner,

          coinAmount,
          pcAmount,
          fixedCoin === poolInfo.coin.mintAddress ? 0 : 1
        )
        : addLiquidityInstruction(
          new PublicKey(poolInfo.programId),

          new PublicKey(poolInfo.ammId),
          new PublicKey(poolInfo.ammAuthority),
          new PublicKey(poolInfo.ammOpenOrders),
          new PublicKey(poolInfo.ammQuantities),
          new PublicKey(poolInfo.lp.mintAddress),
          new PublicKey(poolInfo.poolCoinTokenAccount),
          new PublicKey(poolInfo.poolPcTokenAccount),

          new PublicKey(poolInfo.serumMarket),

          wrappedCoinSolAccount ? wrappedCoinSolAccount : userCoinTokenAccount,
          wrappedSolAccount ? wrappedSolAccount : userPcTokenAccount,
          userLpTokenAccount,
          owner,

          coinAmount,
          pcAmount,
          fixedCoin === poolInfo.coin.mintAddress ? 0 : 1
        )
  )

  if (wrappedCoinSolAccount) {
    transaction.add(
      closeAccount({
        source: wrappedCoinSolAccount,
        destination: owner,
        owner: owner
      })
    )
  }
  if (wrappedSolAccount) {
    transaction.add(
      closeAccount({
        source: wrappedSolAccount,
        destination: owner,
        owner: owner
      })
    )
  }

  return await sendTransaction(connection, wallet, transaction, signers)
}

export async function removeLiquidity(
  connection: Connection | undefined | null,
  wallet: any | undefined | null,
  poolInfo: LiquidityPoolInfo | undefined | null,
  lpAccount: string | undefined | null,
  fromCoinAccount: string | undefined | null,
  toCoinAccount: string | undefined | null,
  amount: string | undefined | null
) {
  if (!connection || !wallet) throw new Error('Miss connection')
  if (!poolInfo) throw new Error('Miss pool infomations')

  if (!lpAccount) throw new Error('Miss account infomations')

  if (!amount) throw new Error('Miss amount infomations')

  const transaction = new Transaction()
  const signers: any = []

  const owner = wallet.publicKey

  const lpAmount = getBigNumber(new TokenAmount(amount, poolInfo.lp.decimals, false).wei)

  let needCloseFromTokenAccount = false
  let newFromTokenAccount
  if (poolInfo.coin.mintAddress === NATIVE_SOL.mintAddress) {
    newFromTokenAccount = await createTokenAccountIfNotExist(
      connection,
      newFromTokenAccount,
      owner,
      TOKENS.WSOL.mintAddress,
      null,
      transaction,
      signers
    )
    needCloseFromTokenAccount = true
  } else {
    newFromTokenAccount = await createAssociatedTokenAccountIfNotExist(
      fromCoinAccount,
      owner,
      poolInfo.coin.mintAddress,
      transaction
    )
  }

  let needCloseToTokenAccount = false
  let newToTokenAccount
  if (poolInfo.pc.mintAddress === NATIVE_SOL.mintAddress) {
    newToTokenAccount = await createTokenAccountIfNotExist(
      connection,
      newToTokenAccount,
      owner,
      TOKENS.WSOL.mintAddress,
      null,
      transaction,
      signers
    )
    needCloseToTokenAccount = true
  } else {
    newToTokenAccount = await createAssociatedTokenAccountIfNotExist(
      toCoinAccount,
      owner,
      poolInfo.pc.mintAddress === NATIVE_SOL.mintAddress ? TOKENS.WSOL.mintAddress : poolInfo.pc.mintAddress,
      transaction
    )
  }

  transaction.add(
    poolInfo.version === 5
      ? removeLiquidityInstructionStable(
        new PublicKey(poolInfo.programId),

        new PublicKey(poolInfo.ammId),
        new PublicKey(poolInfo.ammAuthority),
        new PublicKey(poolInfo.ammOpenOrders),
        new PublicKey(poolInfo.ammTargetOrders),
        new PublicKey(poolInfo.lp.mintAddress),
        new PublicKey(poolInfo.poolCoinTokenAccount),
        new PublicKey(poolInfo.poolPcTokenAccount),
        new PublicKey(poolInfo.modelDataAccount ?? ''),
        new PublicKey(poolInfo.serumProgramId),
        new PublicKey(poolInfo.serumMarket),
        new PublicKey(poolInfo.serumCoinVaultAccount),
        new PublicKey(poolInfo.serumPcVaultAccount),
        new PublicKey(poolInfo.serumVaultSigner),

        new PublicKey(lpAccount),
        newFromTokenAccount,
        newToTokenAccount,
        owner,

        poolInfo,

        lpAmount
      )
      : poolInfo.version === 4
        ? removeLiquidityInstructionV4(
          new PublicKey(poolInfo.programId),

          new PublicKey(poolInfo.ammId),
          new PublicKey(poolInfo.ammAuthority),
          new PublicKey(poolInfo.ammOpenOrders),
          new PublicKey(poolInfo.ammTargetOrders),
          new PublicKey(poolInfo.lp.mintAddress),
          new PublicKey(poolInfo.poolCoinTokenAccount),
          new PublicKey(poolInfo.poolPcTokenAccount),
          new PublicKey(poolInfo.poolWithdrawQueue),
          new PublicKey(poolInfo.poolTempLpTokenAccount),

          new PublicKey(poolInfo.serumProgramId),
          new PublicKey(poolInfo.serumMarket),
          new PublicKey(poolInfo.serumCoinVaultAccount),
          new PublicKey(poolInfo.serumPcVaultAccount),
          new PublicKey(poolInfo.serumVaultSigner),

          new PublicKey(lpAccount),
          newFromTokenAccount,
          newToTokenAccount,
          owner,

          poolInfo,

          lpAmount
        )
        : removeLiquidityInstruction(
          new PublicKey(poolInfo.programId),

          new PublicKey(poolInfo.ammId),
          new PublicKey(poolInfo.ammAuthority),
          new PublicKey(poolInfo.ammOpenOrders),
          new PublicKey(poolInfo.ammQuantities),
          new PublicKey(poolInfo.lp.mintAddress),
          new PublicKey(poolInfo.poolCoinTokenAccount),
          new PublicKey(poolInfo.poolPcTokenAccount),
          new PublicKey(poolInfo.poolWithdrawQueue),
          new PublicKey(poolInfo.poolTempLpTokenAccount),

          new PublicKey(poolInfo.serumProgramId),
          new PublicKey(poolInfo.serumMarket),
          new PublicKey(poolInfo.serumCoinVaultAccount),
          new PublicKey(poolInfo.serumPcVaultAccount),
          new PublicKey(poolInfo.serumVaultSigner),

          new PublicKey(lpAccount),
          newFromTokenAccount,
          newToTokenAccount,
          owner,

          lpAmount
        )
  )

  if (needCloseFromTokenAccount) {
    transaction.add(
      closeAccount({
        source: newFromTokenAccount,
        destination: owner,
        owner: owner
      })
    )
  }
  if (needCloseToTokenAccount) {
    transaction.add(
      closeAccount({
        source: newToTokenAccount,
        destination: owner,
        owner: owner
      })
    )
  }

  return await sendTransaction(connection, wallet, transaction, signers)
}

export function addLiquidityInstruction(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammQuantities: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  // serum
  serumMarket: PublicKey,
  // user
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userLpTokenAccount: PublicKey,
  userOwner: PublicKey,

  maxCoinAmount: number,
  maxPcAmount: number,
  fixedFromCoin: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('maxCoinAmount'), nu64('maxPcAmount'), nu64('fixedFromCoin')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: false },
    { pubkey: ammQuantities, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: serumMarket, isSigner: false, isWritable: false },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false }
  ]

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 3,
      maxCoinAmount,
      maxPcAmount,
      fixedFromCoin
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export function addLiquidityInstructionV4(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammTargetOrders: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  // serum
  serumMarket: PublicKey,
  marketEventQueue: PublicKey,
  // user
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userLpTokenAccount: PublicKey,
  userOwner: PublicKey,

  maxCoinAmount: number,
  maxPcAmount: number,
  fixedFromCoin: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('maxCoinAmount'), nu64('maxPcAmount'), nu64('fixedFromCoin')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: false },
    { pubkey: ammTargetOrders, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: serumMarket, isSigner: false, isWritable: false },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false },
    { pubkey: marketEventQueue, isSigner: false, isWritable: false }
  ]

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 3,
      maxCoinAmount,
      maxPcAmount,
      fixedFromCoin
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export function removeLiquidityInstruction(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammQuantities: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  poolWithdrawQueue: PublicKey,
  poolTempLpTokenAccount: PublicKey,
  // serum
  serumProgramId: PublicKey,
  serumMarket: PublicKey,
  serumCoinVaultAccount: PublicKey,
  serumPcVaultAccount: PublicKey,
  serumVaultSigner: PublicKey,
  // user
  userLpTokenAccount: PublicKey,
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userOwner: PublicKey,

  amount: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('amount')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: true },
    { pubkey: ammQuantities, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolWithdrawQueue, isSigner: false, isWritable: true },
    { pubkey: poolTempLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: serumProgramId, isSigner: false, isWritable: false },
    { pubkey: serumMarket, isSigner: false, isWritable: true },
    { pubkey: serumCoinVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumPcVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumVaultSigner, isSigner: false, isWritable: false },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false }
  ]

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 4,
      amount: amount
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export function removeLiquidityInstructionV4(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammTargetOrders: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  poolWithdrawQueue: PublicKey,
  poolTempLpTokenAccount: PublicKey,
  // serum
  serumProgramId: PublicKey,
  serumMarket: PublicKey,
  serumCoinVaultAccount: PublicKey,
  serumPcVaultAccount: PublicKey,
  serumVaultSigner: PublicKey,
  // user
  userLpTokenAccount: PublicKey,
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userOwner: PublicKey,

  poolInfo: LiquidityPoolInfo,

  amount: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('amount')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: true },
    { pubkey: ammTargetOrders, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolWithdrawQueue, isSigner: false, isWritable: true },
    { pubkey: poolTempLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: serumProgramId, isSigner: false, isWritable: false },
    { pubkey: serumMarket, isSigner: false, isWritable: true },
    { pubkey: serumCoinVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumPcVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumVaultSigner, isSigner: false, isWritable: false },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false }
  ]

  if (poolInfo.serumEventQueue) {
    keys.push({ pubkey: new PublicKey(poolInfo.serumEventQueue), isSigner: false, isWritable: true })
  }
  if (poolInfo.serumBids) {
    keys.push({ pubkey: new PublicKey(poolInfo.serumBids), isSigner: false, isWritable: true })
  }
  if (poolInfo.serumAsks) {
    keys.push({ pubkey: new PublicKey(poolInfo.serumAsks), isSigner: false, isWritable: true })
  }

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 4,
      amount: amount
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export function addLiquidityInstructionStable(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammTargetOrders: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  modelDataAccount: PublicKey,
  // serum
  serumMarket: PublicKey,
  // user
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userLpTokenAccount: PublicKey,
  userOwner: PublicKey,

  maxCoinAmount: number,
  maxPcAmount: number,
  fixedFromCoin: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('maxCoinAmount'), nu64('maxPcAmount'), nu64('fixedFromCoin')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: false },
    { pubkey: ammTargetOrders, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: modelDataAccount, isSigner: false, isWritable: false },
    { pubkey: serumMarket, isSigner: false, isWritable: false },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false }
  ]

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 3,
      maxCoinAmount,
      maxPcAmount,
      fixedFromCoin
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export function removeLiquidityInstructionStable(
  programId: PublicKey,
  // tokenProgramId: PublicKey,
  // amm
  ammId: PublicKey,
  ammAuthority: PublicKey,
  ammOpenOrders: PublicKey,
  ammTargetOrders: PublicKey,
  lpMintAddress: PublicKey,
  poolCoinTokenAccount: PublicKey,
  poolPcTokenAccount: PublicKey,
  modelDataAccount: PublicKey,
  // serum
  serumProgramId: PublicKey,
  serumMarket: PublicKey,
  serumCoinVaultAccount: PublicKey,
  serumPcVaultAccount: PublicKey,
  serumVaultSigner: PublicKey,
  // user
  userLpTokenAccount: PublicKey,
  userCoinTokenAccount: PublicKey,
  userPcTokenAccount: PublicKey,
  userOwner: PublicKey,

  poolInfo: LiquidityPoolInfo,

  amount: number
): TransactionInstruction {
  const dataLayout = struct([u8('instruction'), nu64('amount')])

  const keys = [
    { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
    { pubkey: ammId, isSigner: false, isWritable: true },
    { pubkey: ammAuthority, isSigner: false, isWritable: false },
    { pubkey: ammOpenOrders, isSigner: false, isWritable: true },
    { pubkey: ammTargetOrders, isSigner: false, isWritable: true },
    { pubkey: lpMintAddress, isSigner: false, isWritable: true },
    { pubkey: poolCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: poolPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: modelDataAccount, isSigner: false, isWritable: false },
    { pubkey: serumProgramId, isSigner: false, isWritable: false },
    { pubkey: serumMarket, isSigner: false, isWritable: true },
    { pubkey: serumCoinVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumPcVaultAccount, isSigner: false, isWritable: true },
    { pubkey: serumVaultSigner, isSigner: false, isWritable: false },
    { pubkey: userLpTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userCoinTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userPcTokenAccount, isSigner: false, isWritable: true },
    { pubkey: userOwner, isSigner: true, isWritable: false }
  ]

  if (poolInfo.serumEventQueue && poolInfo.serumBids && poolInfo.serumAsks) {
    keys.push({ pubkey: new PublicKey(poolInfo.serumEventQueue), isSigner: false, isWritable: true })
    keys.push({ pubkey: new PublicKey(poolInfo.serumBids), isSigner: false, isWritable: true })
    keys.push({ pubkey: new PublicKey(poolInfo.serumAsks), isSigner: false, isWritable: true })
  }

  const data = Buffer.alloc(dataLayout.span)
  dataLayout.encode(
    {
      instruction: 4,
      amount: amount
    },
    data
  )

  return new TransactionInstruction({
    keys,
    programId,
    data
  })
}

export const AMM_INFO_LAYOUT = struct([
  u64('status'),
  u64('nonce'),
  u64('orderNum'),
  u64('depth'),
  u64('coinDecimals'),
  u64('pcDecimals'),
  u64('state'),
  u64('resetFlag'),
  u64('fee'),
  u64('minSize'),
  u64('volMaxCutRatio'),
  u64('pnlRatio'),
  u64('amountWaveRatio'),
  u64('coinLotSize'),
  u64('pcLotSize'),
  u64('minPriceMultiplier'),
  u64('maxPriceMultiplier'),
  u64('needTakePnlCoin'),
  u64('needTakePnlPc'),
  u64('totalPnlX'),
  u64('totalPnlY'),
  u64('systemDecimalsValue'),
  publicKey('poolCoinTokenAccount'),
  publicKey('poolPcTokenAccount'),
  publicKey('coinMintAddress'),
  publicKey('pcMintAddress'),
  publicKey('lpMintAddress'),
  publicKey('ammOpenOrders'),
  publicKey('serumMarket'),
  publicKey('serumProgramId'),
  publicKey('ammTargetOrders'),
  publicKey('ammQuantities'),
  publicKey('poolWithdrawQueue'),
  publicKey('poolTempLpTokenAccount'),
  publicKey('ammOwner'),
  publicKey('pnlOwner')
])

export const AMM_INFO_LAYOUT_V3 = struct([
  u64('status'),
  u64('nonce'),
  u64('orderNum'),
  u64('depth'),
  u64('coinDecimals'),
  u64('pcDecimals'),
  u64('state'),
  u64('resetFlag'),
  u64('fee'),
  u64('min_separate'),
  u64('minSize'),
  u64('volMaxCutRatio'),
  u64('pnlRatio'),
  u64('amountWaveRatio'),
  u64('coinLotSize'),
  u64('pcLotSize'),
  u64('minPriceMultiplier'),
  u64('maxPriceMultiplier'),
  u64('needTakePnlCoin'),
  u64('needTakePnlPc'),
  u64('totalPnlX'),
  u64('totalPnlY'),
  u64('poolTotalDepositPc'),
  u64('poolTotalDepositCoin'),
  u64('systemDecimalsValue'),
  publicKey('poolCoinTokenAccount'),
  publicKey('poolPcTokenAccount'),
  publicKey('coinMintAddress'),
  publicKey('pcMintAddress'),
  publicKey('lpMintAddress'),
  publicKey('ammOpenOrders'),
  publicKey('serumMarket'),
  publicKey('serumProgramId'),
  publicKey('ammTargetOrders'),
  publicKey('ammQuantities'),
  publicKey('poolWithdrawQueue'),
  publicKey('poolTempLpTokenAccount'),
  publicKey('ammOwner'),
  publicKey('pnlOwner'),
  publicKey('srmTokenAccount')
])

export const AMM_INFO_LAYOUT_V4 = struct([
  u64('status'),
  u64('nonce'),
  u64('orderNum'),
  u64('depth'),
  u64('coinDecimals'),
  u64('pcDecimals'),
  u64('state'),
  u64('resetFlag'),
  u64('minSize'),
  u64('volMaxCutRatio'),
  u64('amountWaveRatio'),
  u64('coinLotSize'),
  u64('pcLotSize'),
  u64('minPriceMultiplier'),
  u64('maxPriceMultiplier'),
  u64('systemDecimalsValue'),
  // Fees
  u64('minSeparateNumerator'),
  u64('minSeparateDenominator'),
  u64('tradeFeeNumerator'),
  u64('tradeFeeDenominator'),
  u64('pnlNumerator'),
  u64('pnlDenominator'),
  u64('swapFeeNumerator'),
  u64('swapFeeDenominator'),
  // OutPutData
  u64('needTakePnlCoin'),
  u64('needTakePnlPc'),
  u64('totalPnlPc'),
  u64('totalPnlCoin'),

  u64('poolOpenTime'),
  u64('punishPcAmount'),
  u64('punishCoinAmount'),
  u64('orderbookToInitTime'),

  u128('swapCoinInAmount'),
  u128('swapPcOutAmount'),
  u64('swapCoin2PcFee'),
  u128('swapPcInAmount'),
  u128('swapCoinOutAmount'),
  u64('swapPc2CoinFee'),

  publicKey('poolCoinTokenAccount'),
  publicKey('poolPcTokenAccount'),
  publicKey('coinMintAddress'),
  publicKey('pcMintAddress'),
  publicKey('lpMintAddress'),
  publicKey('ammOpenOrders'),
  publicKey('serumMarket'),
  publicKey('serumProgramId'),
  publicKey('ammTargetOrders'),
  publicKey('poolWithdrawQueue'),
  publicKey('poolTempLpTokenAccount'),
  publicKey('ammOwner'),
  publicKey('pnlOwner')
])

export const AMM_INFO_LAYOUT_STABLE = struct([
  u64('accountType'),
  u64('status'),
  u64('nonce'),
  u64('orderNum'),
  u64('depth'),
  u64('coinDecimals'),
  u64('pcDecimals'),
  u64('state'),
  u64('resetFlag'),
  u64('minSize'),
  u64('volMaxCutRatio'),
  u64('amountWaveRatio'),
  u64('coinLotSize'),
  u64('pcLotSize'),
  u64('minPriceMultiplier'),
  u64('maxPriceMultiplier'),
  u64('systemDecimalsValue'),
  u64('abortTradeFactor'),
  u64('priceTickMultiplier'),
  u64('priceTick'),
  // Fees
  u64('minSeparateNumerator'),
  u64('minSeparateDenominator'),
  u64('tradeFeeNumerator'),
  u64('tradeFeeDenominator'),
  u64('pnlNumerator'),
  u64('pnlDenominator'),
  u64('swapFeeNumerator'),
  u64('swapFeeDenominator'),
  // OutPutData
  u64('needTakePnlCoin'),
  u64('needTakePnlPc'),
  u64('totalPnlPc'),
  u64('totalPnlCoin'),
  u64('poolOpenTime'),
  u64('punishPcAmount'),
  u64('punishCoinAmount'),
  u64('orderbookToInitTime'),
  u128('swapCoinInAmount'),
  u128('swapPcOutAmount'),
  u128('swapPcInAmount'),
  u128('swapCoinOutAmount'),
  u64('swapCoin2PcFee'),
  u64('swapPc2CoinFee'),

  publicKey('poolCoinTokenAccount'),
  publicKey('poolPcTokenAccount'),
  publicKey('coinMintAddress'),
  publicKey('pcMintAddress'),
  publicKey('lpMintAddress'),
  publicKey('modelDataAccount'),
  publicKey('ammOpenOrders'),
  publicKey('serumMarket'),
  publicKey('serumProgramId'),
  publicKey('ammTargetOrders'),
  publicKey('ammOwner'),
  seq(u64('padding'), 64, 'padding')
])

export async function getLpMintInfo(conn: any, mintAddress: string, coin: any, pc: any): Promise<TokenInfo> {
  let lpInfo = Object.values(LP_TOKENS).find((item) => item.mintAddress === mintAddress)
  if (!lpInfo) {
    const mintAll = await getMultipleAccounts(conn, [new PublicKey(mintAddress)], commitment)
    if (mintAll !== null) {
      const data = Buffer.from(mintAll[0]?.account.data ?? '')
      const mintLayoutData = MINT_LAYOUT.decode(data)
      lpInfo = {
        symbol: 'unknown',
        name: 'unknown',
        coin,
        pc,
        mintAddress: mintAddress,
        decimals: mintLayoutData.decimals
      }
    }
  }
  return lpInfo
}

export async function getLpMintListDecimals(
  conn: any,
  mintAddressInfos: string[]
): Promise<{ [name: string]: number }> {
  const reLpInfoDict: { [name: string]: number } = {}
  const mintList = [] as PublicKey[]
  mintAddressInfos.forEach((item) => {
    let lpInfo = Object.values(LP_TOKENS).find((itemLpToken) => itemLpToken.mintAddress === item)
    if (!lpInfo) {
      mintList.push(new PublicKey(item))
      lpInfo = {
        decimals: null
      }
    }
    reLpInfoDict[item] = lpInfo.decimals
  })

  const mintAll = await getMultipleAccounts(conn, mintList, commitment)
  for (let mintIndex = 0; mintIndex < mintAll.length; mintIndex += 1) {
    const itemMint = mintAll[mintIndex]
    if (itemMint) {
      const mintLayoutData = MINT_LAYOUT.decode(Buffer.from(itemMint.account.data))
      reLpInfoDict[mintList[mintIndex].toString()] = mintLayoutData.decimals
    }
  }
  const reInfo: { [name: string]: number } = {}
  for (const key of Object.keys(reLpInfoDict)) {
    if (reLpInfoDict[key] !== null) {
      reInfo[key] = reLpInfoDict[key]
    }
  }
  return reInfo
}

export function getLiquidityInfoSimilar(
  ammIdOrMarket: string | undefined,
  from: string | undefined,
  to: string | undefined
) {
  // const fromCoin = from === NATIVE_SOL.mintAddress ? TOKENS.WSOL.mintAddress : from
  // const toCoin = to === NATIVE_SOL.mintAddress ? TOKENS.WSOL.mintAddress : to
  const fromCoin = from === TOKENS.WSOL.mintAddress ? NATIVE_SOL.mintAddress : from
  const toCoin = to === TOKENS.WSOL.mintAddress ? NATIVE_SOL.mintAddress : to
  const knownLiquidity = LIQUIDITY_POOLS.find((item) => {
    if (fromCoin !== undefined && toCoin != undefined && fromCoin === toCoin) {
      return false
    }
    if (ammIdOrMarket !== undefined && !(item.ammId === ammIdOrMarket || item.serumMarket === ammIdOrMarket)) {
      return false
    }
    if (fromCoin && item.pc.mintAddress !== fromCoin && item.coin.mintAddress !== fromCoin) {
      return false
    }
    if (toCoin && item.pc.mintAddress !== toCoin && item.coin.mintAddress !== toCoin) {
      return false
    }
    if (ammIdOrMarket || (fromCoin && toCoin)) {
      return true
    }
    return false
  })
  return knownLiquidity
}

export function getLiquidityInfo(from: string, to: string) {
  const fromCoin = from === TOKENS.WSOL.mintAddress ? NATIVE_SOL.mintAddress : from
  const toCoin = to === TOKENS.WSOL.mintAddress ? NATIVE_SOL.mintAddress : to
  return LIQUIDITY_POOLS.filter(
    (item) =>
      item.version === 4 &&
      ((item.coin.mintAddress === fromCoin && item.pc.mintAddress === toCoin) ||
        (item.coin.mintAddress === toCoin && item.pc.mintAddress === fromCoin))
  )
}

export function getQueryVariable(variable: string) {
  var query = window.location.search.substring(1)
  var vars = query.split('&')
  for (var i = 0; i < vars.length; i++) {
    var pair = vars[i].split('=')
    if (pair[0] == variable) {
      return pair[1]
    }
  }
  return undefined
}

import { Connection } from '@solana/web3.js'
import { TokenInfo } from '../utils/tokens'

export interface NuxtApiInstance {
  getConfig(): Promise<any>
  getEpochInfo(rpc: string): Promise<any>
  getCompaign(params: { campaignId?: number; address: string; referral: string }): Promise<any>
  postCompaign(params: { campaignId?: number; address: string; task: string; result?: string; sign?: string }): Promise<any>
  getCompaignWinners(params: { campaignId: number }): Promise<any>
}

export interface RouterInfoItem {
  ammId: string
  label: string
  inputMint: string
  outputMint: string
  inAmount: string
  outAmount: string
  minInAmount?: string
  minOutAmount?: string
  priceImpact: string
  fee: string
  currentPrice: string
}

export interface RouterInfo {
  keys: string[]
  items: { [ammId: string]: RouterInfoItem }
  routes: string[][]
}

export interface PairInfo {
  ammId: string
  base: TokenInfo
  quote: TokenInfo
  lpMint: string
  name: string
}

export interface CampaignInfo {
  campaignAddress: string 
  tasks: Record<string, number>
}

export interface CampaignWinners {
  total: number
  data: Array<{
    address: string
    amount: number
  }>
}

export interface Rpc {
  endpoint: string
  weight: number
}


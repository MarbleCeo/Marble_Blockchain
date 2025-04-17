import { Connection } from '@solana/web3.js'

export const useSolanaStore = defineStore('solana', {
  state: () => ({
    connection: null as Connection | null,
    selectedRpc: '',
    isConnected: false,
    error: null as string | null
  }),
  
  actions: {
    async connect(rpcUrl: string) {
      try {
        const connection = new Connection(rpcUrl, 'confirmed')
        // Test connection
        await connection.getSlot()
        
        this.connection = connection
        this.selectedRpc = rpcUrl
        this.isConnected = true
        this.error = null
        
        return true
      } catch (err) {
        this.error = 'Failed to connect to RPC node'
        console.error('Connection error:', err)
        return false
      }
    },
    
    disconnect() {
      this.connection = null
      this.selectedRpc = ''
      this.isConnected = false
    }
  },
  
  getters: {
    getConnection: (state) => state.connection,
    getSelectedRpc: (state) => state.selectedRpc,
    connectionStatus: (state) => state.isConnected
  }
})


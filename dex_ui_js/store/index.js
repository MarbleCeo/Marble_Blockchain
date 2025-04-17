export const state = () => ({
  availableRpcNodes: [
    { name: 'Tatum RPC', url: 'https://solana-mainnet.gateway.tatum.io' },
    { name: 'Default RPC', url: 'https://rpc.example.com' },
    { name: 'Public RPC', url: 'https://api.mainnet-beta.solana.com' }
  ],
  currentRpcNode: { name: 'Tatum RPC', url: 'https://solana-mainnet.gateway.tatum.io' }
})

export const mutations = {
  setRpcNode(state, node) {
    state.currentRpcNode = node
  }
}

export const actions = {
  updateRpcNode({ commit }, node) {
    // Here you could add any additional logic like
    // saving the preference to localStorage or making API calls
    
    commit('setRpcNode', node)
    
    // Example of updating a web3 provider or API endpoint
    // if (this.$web3) {
    //   this.$web3.setProvider(new Web3.providers.HttpProvider(node.url))
    // }
  }
}

export const getters = {
  getCurrentRpcNode: state => state.currentRpcNode,
  getAvailableRpcNodes: state => state.availableRpcNodes
}


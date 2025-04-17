import { ref } from 'vue'

const API_BASE_URL = 'http://localhost:8000'

/**
 * Fetches the current status of the Marble DEX API
 */
export function fetchStatus() {
  const status = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const execute = async () => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(`${API_BASE_URL}/status`)
      if (!response.ok) {
        throw new Error(`Error fetching status: ${response.statusText}`)
      }
      status.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error occurred'
      console.error(error.value)
    } finally {
      loading.value = false
    }
  }

  return {
    status,
    error,
    loading,
    execute
  }
}

/**
 * Fetches the DEX configuration
 */
export function fetchDexConfig() {
  const config = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const execute = async () => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(`${API_BASE_URL}/dex_config`)
      if (!response.ok) {
        throw new Error(`Error fetching DEX config: ${response.statusText}`)
      }
      config.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error occurred'
      console.error(error.value)
    } finally {
      loading.value = false
    }
  }

  return {
    config,
    error,
    loading,
    execute
  }
}

/**
 * Validates a blockchain block
 * @param blockId - ID of the block to validate
 * @param coins - Amount of coins for validation
 */
export function fetchValidateBlock() {
  const result = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const execute = async (blockId: string, coins: number) => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(`${API_BASE_URL}/validate_block`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          block_id: blockId,
          coins: coins
        })
      })
      
      if (!response.ok) {
        throw new Error(`Error validating block: ${response.statusText}`)
      }
      
      result.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error occurred'
      console.error(error.value)
    } finally {
      loading.value = false
    }
  }

  return {
    result,
    error,
    loading,
    execute
  }
}

/**
 * Analyzes a blockchain block
 * @param blockId - ID of the block to analyze
 */
export function fetchAnalyzeBlock() {
  const analysis = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const execute = async (blockId: string) => {
    loading.value = true
    error.value = null
    
    try {
      const response = await fetch(`${API_BASE_URL}/analyze_block/${blockId}`)
      
      if (!response.ok) {
        throw new Error(`Error analyzing block: ${response.statusText}`)
      }
      
      analysis.value = await response.json()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error occurred'
      console.error(error.value)
    } finally {
      loading.value = false
    }
  }

  return {
    analysis,
    error,
    loading,
    execute
  }
}


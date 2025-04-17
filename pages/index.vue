<template>
  <div class="container">
    <div class="hero card mb-3">
      <h1 class="hero-title">Welcome to Marble DEX</h1>
      <p class="hero-subtitle">High-performance blockchain with DEX and cross-bridge support</p>
      <img src="/logo.png" alt="Marble DEX Logo" class="hero-logo" />
    </div>

    <div class="dashboard grid mb-3">
      <div class="stats-card card">
        <h2>Network Status</h2>
        <div v-if="loading" class="loading">Loading...</div>
        <div v-else class="stats">
          <div class="stat-item">
            <span class="stat-label">Status:</span>
            <span class="stat-value">{{ statusData?.status || 'Unknown' }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Block Height:</span>
            <span class="stat-value">{{ statusData?.block_height || 0 }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Transaction Count:</span>
            <span class="stat-value">{{ statusData?.tx_count || 0 }}</span>
          </div>
        </div>
      </div>

      <div class="actions-card card">
        <h2>Block Actions</h2>
        <div class="action-inputs">
          <div class="input-group mb-2">
            <label for="block-id">Block ID:</label>
            <input
              id="block-id"
              v-model="blockId"
              type="number"
              min="0"
              placeholder="Enter Block ID"
              class="input"
            />
          </div>
          <div class="input-group mb-2" v-if="showValidateInput">
            <label for="coins">Coins:</label>
            <input
              id="coins"
              v-model="coins"
              type="number"
              min="0"
              placeholder="Enter Coins"
              class="input"
            />
          </div>
        </div>
        <div class="action-buttons">
          <button @click="handleValidateClick" class="button mr-2">
            Validate Block
          </button>
          <button @click="handleAnalyzeClick" class="button">
            Analyze Block
          </button>
        </div>
      </div>
    </div>

    <div class="result-card card" v-if="result">
      <h2>Result</h2>
      <pre class="result-data">{{ JSON.stringify(result, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useApi } from '~/composables/useApi';

const { fetchStatus, fetchValidateBlock, fetchAnalyzeBlock } = useApi();

const statusData = ref(null);
const loading = ref(true);
const result = ref(null);
const blockId = ref('');
const coins = ref('');
const showValidateInput = ref(false);

onMounted(async () => {
  try {
    statusData.value = await fetchStatus();
  } catch (error) {
    console.error('Failed to fetch status:', error);
  } finally {
    loading.value = false;
  }
});

const handleValidateClick = async () => {
  if (!blockId.value) {
    alert('Please enter a Block ID');
    return;
  }
  
  showValidateInput.value = true;
  
  if (!coins.value) return;
  
  try {
    result.value = await fetchValidateBlock(blockId.value, coins.value);
  } catch (error) {
    console.error('Failed to validate block:', error);
    result.value = { error: 'Failed to validate block' };
  }
};

const handleAnalyzeClick = async () => {
  if (!blockId.value) {
    alert('Please enter a Block ID');
    return;
  }
  
  showValidateInput.value = false;
  
  try {
    result.value = await fetchAnalyzeBlock(blockId.value);
  } catch (error) {
    console.error('Failed to analyze block:', error);
    result.value = { error: 'Failed to analyze block' };
  }
};
</script>

<style scoped>
.hero {
  text-align: center;
  padding: 2rem;
}

.hero-title {
  font-size: 2.5rem;
  color: var(--primary-color);
  margin-bottom: 0.5rem;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: var(--secondary-color);
  margin-bottom: 1.5rem;
}

.hero-logo {
  max-width: 150px;
  margin: 0 auto;
}

.dashboard {
  margin: 2rem 0;
}

.stats-card, .actions-card {
  height: 100%;
}

.stats {
  margin-top: 1rem;
}

.stat-item {
  margin-bottom: 0.5rem;
  display: flex;
  justify-content: space-between;
}

.stat-label {
  font-weight: 600;
}

.stat-value {
  color: var(--primary-color);
}

.action-inputs {
  margin-bottom: 1.5rem;
}

.input-group {
  display: flex;
  flex-direction: column;
}

.input-group label {
  margin-bottom: 0.3rem;
  font-weight: 500;
}

.input {
  padding: 0.5rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 1rem;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
}

.mr-2 {
  margin-right: 0.5rem;
}

.result-card {
  background-color: #f8f8f8;
}

.result-data {
  background-color: #f1f1f1;
  padding: 1rem;
  border-radius: 4px;
  overflow-x: auto;
  font-family: monospace;
}

.loading {
  display: flex;
  justify-content: center;
  padding: 1rem;
  color: var(--secondary-color);
}
</style>


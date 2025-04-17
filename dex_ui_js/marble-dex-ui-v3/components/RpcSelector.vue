<template>
  <UModal v-model="isOpen" :ui="{ width: 'sm:max-w-md' }" prevent-close>
    <div class="p-6">
      <h2 class="text-xl font-bold mb-4 spatial-red p-2 rounded">Select RPC Node</h2>
      <div class="space-y-4">
        <URadio
          v-model="selectedRpc"
          v-for="rpc in rpcNodes"
          :key="rpc.url"
          :label="rpc.name"
          :value="rpc.url"
          class="block"
        />
        <p v-if="error" class="text-red-500 text-sm mt-2">{{ error }}</p>
      </div>
      <div class="mt-6 flex justify-end space-x-4">
        <UButton
          label="Connect"
          :loading="loading"
          :disabled="!selectedRpc"
          @click="connect"
          class="spatial-red"
        />
      </div>
    </div>
  </UModal>
</template>

<script setup lang="ts">
const solanaStore = useSolanaStore()
const isOpen = ref(true)
const loading = ref(false)
const selectedRpc = ref('')
const error = ref('')

const rpcNodes = [
  { name: 'Tatum.io', url: 'https://solana-mainnet.gateway.tatum.io', weight: 50 },
  { name: 'Mainnet Beta', url: 'https://api.mainnet-beta.solana.com', weight: 20 },
  { name: 'Project Serum', url: 'https://solana-api.projectserum.com', weight: 15 },
  { name: 'Ankr', url: 'https://rpc.ankr.com/solana', weight: 15 }
]

const connect = async () => {
  if (!selectedRpc.value) return
  loading.value = true
  error.value = ''
  
  try {
    const success = await solanaStore.connect(selectedRpc.value)
    if (success) {
      isOpen.value = false
    } else {
      error.value = 'Failed to connect to RPC node'
    }
  } catch (err) {
    error.value = 'An error occurred while connecting'
    console.error(err)
  } finally {
    loading.value = false
  }
}

// Prevent closing if not connected
watch(() => isOpen.value, (newValue) => {
  if (!solanaStore.isConnected) {
    isOpen.value = true
  }
})
</script>


<template>
  <UModal v-model="isOpen" :ui="{ width: 'sm:max-w-md' }">
    <div class="p-6">
      <h2 class="text-xl font-bold mb-4">Select RPC Node</h2>
      <div class="space-y-4">
        <URadio
          v-model="selectedRpc"
          v-for="rpc in rpcNodes"
          :key="rpc.url"
          :label="rpc.name"
          :value="rpc.url"
          class="block"
        />
      </div>
      <div class="mt-6 flex justify-end">
        <UButton
          label="Connect"
          :loading="loading"
          @click="connect"
          class="spatial-red"
        />
      </div>
    </div>
  </UModal>
</template>

<script setup lang="ts">
const isOpen = ref(true)
const loading = ref(false)
const selectedRpc = ref('')

const rpcNodes = [
  { name: 'Tatum.io', url: 'https://solana-mainnet.gateway.tatum.io', weight: 50 },
  { name: 'Mainnet Beta', url: 'https://api.mainnet-beta.solana.com', weight: 20 },
  { name: 'Project Serum', url: 'https://solana-api.projectserum.com', weight: 15 },
  { name: 'Ankr', url: 'https://rpc.ankr.com/solana', weight: 15 }
]

const connect = async () => {
  if (!selectedRpc.value) return
  loading.value = true
  // Add connection logic here
  setTimeout(() => {
    loading.value = false
    isOpen.value = false
  }, 1000)
}
</script>


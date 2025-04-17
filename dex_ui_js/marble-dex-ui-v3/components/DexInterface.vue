<template>
  <div class="min-h-screen bg-gray-900">
    <nav class="border-b border-gray-800 spatial-red">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center">
            <h1 class="text-2xl font-bold text-white">Marble DEX</h1>
          </div>
          <div class="flex items-center space-x-4">
            <UButton
              v-if="solanaStore.isConnected"
              :label="truncatedRpc"
              icon="i-heroicons-server"
              class="spatial-red"
            />
            <UButton
              v-else
              label="Connect RPC"
              icon="i-heroicons-server"
              class="spatial-red"
              @click="openRpcSelector"
            />
          </div>
        </div>
      </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Swap Panel -->
        <div class="bg-gray-800 rounded-lg p-6 spatial-red bg-opacity-10">
          <h2 class="text-xl font-semibold mb-6">Swap</h2>
          <div class="space-y-4">
            <div class="bg-gray-900 rounded-lg p-4">
              <label class="block text-sm font-medium text-gray-400">From</label>
              <div class="mt-1 flex items-center space-x-3">
                <UButton
                  label="Select Token"
                  icon="i-heroicons-chevron-down"
                  class="spatial-red"
                />
                <UInput
                  type="number"
                  placeholder="0.0"
                  class="flex-1"
                />
              </div>
            </div>

            <div class="flex justify-center">
              <UButton
                icon="i-heroicons-arrow-down"
                class="spatial-red rounded-full"
                square
              />
            </div>

            <div class="bg-gray-900 rounded-lg p-4">
              <label class="block text-sm font-medium text-gray-400">To</label>
              <div class="mt-1 flex items-center space-x-3">
                <UButton
                  label="Select Token"
                  icon="i-heroicons-chevron-down"
                  class="spatial-red"
                />
                <UInput
                  type="number"
                  placeholder="0.0"
                  class="flex-1"
                />
              </div>
            </div>

            <UButton
              label="Swap"
              class="w-full spatial-red"
              :disabled="!solanaStore.isConnected"
            />
          </div>
        </div>

        <!-- Info Panel -->
        <div class="bg-gray-800 rounded-lg p-6 spatial-red bg-opacity-10">
          <h2 class="text-xl font-semibold mb-6">Market Info</h2>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div>
                <p class="text-sm text-gray-400">Price Impact</p>
                <p class="text-lg font-medium">--</p>
              </div>
              <div>
                <p class="text-sm text-gray-400">Minimum Received</p>
                <p class="text-lg font-medium">--</p>
              </div>
              <div>
                <p class="text-sm text-gray-400">Liquidity Provider Fee</p>
                <p class="text-lg font-medium">--</p>
              </div>
              <div>
                <p class="text-sm text-gray-400">Route</p>
                <p class="text-lg font-medium">Direct</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
const solanaStore = useSolanaStore()

const truncatedRpc = computed(() => {
  const rpc = solanaStore.getSelectedRpc
  if (!rpc) return ''
  return rpc.replace(/^https?:\/\//, '').slice(0, 20) + '...'
})

const openRpcSelector = () => {
  // Implementation will be added
}
</script>


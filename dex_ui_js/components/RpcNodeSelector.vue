<template>
  <div class="rpc-node-selector">
    <button 
      @click="toggleDropdown" 
      class="rpc-selector-button"
      :class="{ 'active': isOpen }"
    >
      {{ selectedNode.name }}
      <span class="dropdown-arrow">▼</span>
    </button>
    <div v-if="isOpen" class="rpc-dropdown">
      <div 
        v-for="(node, index) in rpcNodes" 
        :key="index" 
        @click="selectNode(node)"
        class="rpc-option"
        :class="{ 'active': selectedNode.url === node.url }"
      >
        {{ node.name }}
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

// Define TypeScript interface for RPC nodes
interface RpcNode {
  name: string;
  url: string;
}

// Define emits
const emit = defineEmits<{
  (e: 'node-changed', node: RpcNode): void
}>()

// State with refs
const isOpen = ref(false)
const selectedNode = ref<RpcNode>({ name: 'Default RPC', url: 'https://rpc.example.com' })
const rpcNodes = ref<RpcNode[]>([
  { name: 'Default RPC', url: 'https://rpc.example.com' },
  { name: 'Backup RPC 1', url: 'https://rpc1.example.com' },
  { name: 'Backup RPC 2', url: 'https://rpc2.example.com' },
  { name: 'Local Node', url: 'http://localhost:8545' }
])

// Methods
const toggleDropdown = () => {
  isOpen.value = !isOpen.value
}

const selectNode = (node: RpcNode) => {
  selectedNode.value = node
  isOpen.value = false
  // Emit an event that can be caught by parent components
  emit('node-changed', node)
  
  // Here you would typically update your web3 provider or API endpoint
  console.log(`RPC node changed to: ${node.name} (${node.url})`)
}

const handleClickOutside = (event: MouseEvent) => {
  const target = event.target as Element
  const element = document.querySelector('.rpc-node-selector')
  if (element && !element.contains(target)) {
    isOpen.value = false
  }
}

// Lifecycle hooks
onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.rpc-node-selector {
  position: relative;
  display: inline-block;
}

.rpc-selector-button {
  background-color: #2c3e50;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background-color 0.2s;
}

.rpc-selector-button:hover, .rpc-selector-button.active {
  background-color: #1a2633;
}

.dropdown-arrow {
  font-size: 10px;
  transition: transform 0.2s ease;
}

.rpc-selector-button.active .dropdown-arrow {
  transform: rotate(180deg);
}

.rpc-dropdown {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 4px;
  background-color: white;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  border-radius: 4px;
  width: 180px;
  z-index: 10;
  overflow: hidden;
}

.rpc-option {
  padding: 10px 16px;
  cursor: pointer;
  transition: background-color 0.2s;
  color: #333;
}

.rpc-option:hover, .rpc-option.active {
  background-color: #f5f5f5;
}

.rpc-option.active {
  font-weight: bold;
  color: #2c3e50;
}
</style>


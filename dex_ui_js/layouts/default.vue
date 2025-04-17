<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <!-- Logo or other header content can go here -->
        <h1 class="site-title">Marble DEX</h1>
      </div>
      <div class="header-right">
        <RpcNodeSelector @node-changed="handleRpcNodeChange" />
      </div>
    </header>
    <main class="app-content">
      <Nuxt />
    </main>
    <footer class="app-footer">
      <p>© {{ new Date().getFullYear() }} Marble DEX</p>
    </footer>
  </div>
</template>

<script>
import RpcNodeSelector from '~/components/RpcNodeSelector.vue'

export default {
  components: {
    RpcNodeSelector
  },
  methods: {
    handleRpcNodeChange(node) {
      // Handle RPC node change at the layout level
      console.log('RPC node changed in layout:', node)
      
      // Here you would typically store the selected node in Vuex or another global state management
      // this.$store.dispatch('updateRpcNode', node)
    }
  }
}
</script>

<style>
/* Global styles */
body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  background-color: #f7f9fc;
  color: #333;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background-color: #ffffff;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.site-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: #2c3e50;
}

.app-content {
  flex: 1;
  padding: 2rem;
}

.app-footer {
  padding: 1rem;
  background-color: #f0f2f5;
  text-align: center;
  color: #666;
  font-size: 0.85rem;
}

.header-left, .header-right {
  display: flex;
  align-items: center;
}
</style>


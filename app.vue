<template>
  <div class="app">
    <header class="app-header">
      <div class="logo-container">
        <img src="/logo.png" alt="Marble DEX Logo" class="logo" />
        <h1 class="app-title">Marble DEX</h1>
      </div>
      <nav class="main-nav">
        <NuxtLink to="/" class="nav-link">Home</NuxtLink>
        <NuxtLink to="/trade" class="nav-link">Trade</NuxtLink>
        <NuxtLink to="/bridge" class="nav-link">Bridge</NuxtLink>
        <NuxtLink to="/stake" class="nav-link">Stake</NuxtLink>
      </nav>
    </header>
    <main class="app-content">
      <NuxtPage />
    </main>
    <footer class="app-footer">
      <p>© {{ new Date().getFullYear() }} Marble DEX - High-performance blockchain with DEX and cross-bridge support</p>
    </footer>
  </div>
</template>

<style>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #ffffff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.logo-container {
  display: flex;
  align-items: center;
}

.logo {
  height: 40px;
  margin-right: 1rem;
}

.app-title {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--primary-color);
  margin: 0;
}

.main-nav {
  display: flex;
  gap: 1.5rem;
}

.nav-link {
  color: #333;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s;
}

.nav-link:hover {
  color: var(--primary-color);
}

.app-content {
  flex: 1;
  padding: 2rem;
}

.app-footer {
  padding: 1rem;
  text-align: center;
  background-color: #ffffff;
  border-top: 1px solid #eaeaea;
}
</style>


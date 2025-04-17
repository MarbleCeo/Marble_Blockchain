// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },
  modules: [
    // Add any required modules here
  ],
  app: {
    head: {
      // Head configuration will be handled by mintbutton module
    }
  },
  // Enable TypeScript
  typescript: {
    strict: true
  },
  // Development server configuration
  devServer: {
    port: 3000
  }
})


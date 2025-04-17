// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  ssr: false,
  compatibilityDate: '2024-11-01',
  modules: [
    '@nuxt/ui',
    '@pinia/nuxt',
    '@vueuse/nuxt'
  ],
  css: ['~/assets/css/main.css'],
  ui: {
    global: true,
    icons: ['heroicons'],
    primary: 'red',
    colors: {
      primary: {
        50: '#fff1f1',
        100: '#ffe1e1',
        200: '#ffc7c7',
        300: '#ffa0a0',
        400: '#ff6b6b',
        500: '#ff3b3b',
        600: '#ff0000',
        700: '#db0000',
        800: '#b80000',
        900: '#920000',
        950: '#500000'
      }
    }
  },
  pinia: {
    autoImports: [
      'defineStore',
      'storeToRefs'
    ]
  },
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
  app: {
    head: {
      title: 'Marble DEX',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' }
      ]
    }
  },
  devtools: { enabled: true }
})

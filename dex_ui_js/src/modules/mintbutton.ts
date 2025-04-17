import { defineNuxtModule } from '@nuxt/kit'

export default defineNuxtModule({
  setup(options, nuxt) {
    nuxt.options.app.head = nuxt.options.app.head || {}
    nuxt.options.app.head.script = nuxt.options.app.head.script || []
    nuxt.options.app.head.link = nuxt.options.app.head.link || []

    nuxt.options.app.head.script.push(
      {
        src: 'https://unpkg.com/@magiceden/mintbutton@latest/umd/runtime-main.js',
        tagPosition: 'head'
      },
      {
        src: 'https://unpkg.com/@magiceden/mintbutton@latest/umd/main.js',
        tagPosition: 'head'
      },
      {
        src: 'https://unpkg.com/@magiceden/mintbutton@latest/umd/2.js',
        tagPosition: 'head'
      }
    )

    nuxt.options.app.head.link.push({
      rel: 'stylesheet',
      href: 'https://unpkg.com/@magiceden/mintbutton@latest/umd/main.css'
    })
  }
})

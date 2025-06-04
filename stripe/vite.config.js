import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

export default defineConfig({
  base: '/',
  plugins: [react()],
  build: {
    outDir: 'build'
  },
  server: {
    proxy: {
      '/create-payment-intent': {
        target: 'http://localhost:4242',
        changeOrigin: true
      }
    }
  }
})

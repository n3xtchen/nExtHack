import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import svgr from 'vite-plugin-svgr';

export default defineConfig({
  base: '/',
  plugins: [react(), svgr()],
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

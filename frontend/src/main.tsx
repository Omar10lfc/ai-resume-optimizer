import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { initTheme } from './lib/theme'
import App from './App.tsx'

// Applied before the first paint so the page never flashes the wrong theme.
// Bundled module code — no inline scripts, so MV3 CSP stays intact.
initTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

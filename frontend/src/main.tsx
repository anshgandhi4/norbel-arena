import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import LeaderboardPage from './LeaderboardPage'
import './styles.css'

function normalizePath(pathname: string): string {
  if (pathname.length > 1 && pathname.endsWith('/')) {
    return pathname.slice(0, -1)
  }
  return pathname
}

const path = normalizePath(window.location.pathname)
const RootComponent = path === '/leaderboard' ? LeaderboardPage : App

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootComponent />
  </React.StrictMode>
)

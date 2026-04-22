import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './App.css'

import Home from './pages/Home.jsx'
import Leaderboard from './pages/Leaderboard.jsx'
import About from './pages/About.jsx'
import Submit from './pages/Submit.jsx'

const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  { path: '/leaderboard', element: <Leaderboard /> },
  { path: '/about', element: <About /> },
  { path: '/submit', element: <Submit /> },
])

export default function App() {
  return <RouterProvider router={router} />
}

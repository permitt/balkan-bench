import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './App.css'

import Home from './pages/Home.jsx'
import Leaderboard from './pages/Leaderboard.jsx'
import About from './pages/About.jsx'
import Submit from './pages/Submit.jsx'
import Acknowledgements from './pages/Acknowledgements.jsx'

const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  { path: '/leaderboard', element: <Leaderboard /> },
  { path: '/about', element: <About /> },
  { path: '/submit', element: <Submit /> },
  { path: '/acknowledgements', element: <Acknowledgements /> },
])

export default function App() {
  return <RouterProvider router={router} />
}

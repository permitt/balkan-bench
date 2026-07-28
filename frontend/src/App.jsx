import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom'
import Shell from './components/Shell.jsx'
import Home from './pages/Home.jsx'
import Leaderboard from './pages/Leaderboard.jsx'
import About from './pages/About.jsx'
import Submit from './pages/Submit.jsx'

const router = createBrowserRouter([
  {
    element: <Shell />,
    children: [
      { path: '/', element: <Home /> },
      { path: '/leaderboard', element: <Leaderboard /> },
      { path: '/about', element: <About /> },
      { path: '/submit', element: <Submit /> },
      { path: '/acknowledgements', element: <Navigate to="/about#acknowledgements" replace /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}

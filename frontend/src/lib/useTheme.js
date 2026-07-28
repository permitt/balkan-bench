import { useEffect, useState } from 'react'

const KEY = 'bb-theme'

function apply(theme) {
  if (theme === 'light' || theme === 'dark') {
    document.documentElement.dataset.theme = theme
  } else {
    delete document.documentElement.dataset.theme
  }
}

export function useTheme() {
  const [theme, setThemeState] = useState(() => localStorage.getItem(KEY) ?? 'auto')

  useEffect(() => { apply(theme) }, [theme])

  const setTheme = (next) => {
    if (next === 'auto') localStorage.removeItem(KEY)
    else localStorage.setItem(KEY, next)
    setThemeState(next)
  }

  return { theme, setTheme }
}

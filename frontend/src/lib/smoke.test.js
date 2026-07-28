import React from 'react'
import { render, screen } from '@testing-library/react'

test('harness renders JSX', () => {
  render(React.createElement('div', null, 'hello'))
  expect(screen.getByText('hello')).toBeInTheDocument()
})

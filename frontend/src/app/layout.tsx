import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Interpoletor - 5D Regression System',
  description: 'Machine Learning system for 5-dimensional regression with neural networks',
  keywords: ['machine learning', 'neural network', 'regression', 'pytorch', 'data science'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      colors: {
        surface: {
          DEFAULT: '#ffffff',
          dim: '#f8fafc',
          container: '#f1f5f9',
          'container-high': '#e2e8f0',
          'container-highest': '#cbd5e1',
          dark: '#0f172a',
          'dark-dim': '#1e293b',
          'dark-container': '#334155',
          'dark-container-high': '#475569',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};

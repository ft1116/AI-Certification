/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ['./src/**/*.{js,jsx,ts,tsx}'],
    theme: {
      extend: {
        colors: {
          'dark-bg': '#1a202c',
          'panel-bg': '#2d3748',
          'accent-blue': '#63b3ed',
          'text-light': '#e2e8f0',
        },
      },
    },
    plugins: [],
  }

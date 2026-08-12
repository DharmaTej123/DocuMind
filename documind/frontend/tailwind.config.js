/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#14161F",
          800: "#1B1E29",
          700: "#232733",
          600: "#2E3340",
        },
        paper: "#EEF1F6",
        surface: "#FFFFFF",
        signal: {
          DEFAULT: "#0F8B8D",
          soft: "#E3F4F3",
        },
        attention: {
          DEFAULT: "#E7A33E",
          soft: "#FBF0DC",
        },
        muted: "#6B7280",
        border: "#DDE2EA",
        danger: "#C1443A",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        md: "6px",
        lg: "10px",
      },
      boxShadow: {
        panel: "0 1px 2px rgba(20,22,31,0.06), 0 8px 24px rgba(20,22,31,0.04)",
      },
    },
  },
  plugins: [],
};

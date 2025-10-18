import { createTheme, ThemeOptions } from '@mui/material/styles';

// Custom color palette
const colors = {
  primary: '#1976d2',      // Blue - Primary actions
  secondary: '#dc004e',    // Red - Critical actions
  success: '#4caf50',      // Green - Success states
  warning: '#ff9800',      // Orange - Warnings
  error: '#f44336',        // Red - Errors
  info: '#2196f3',         // Light blue - Info
  background: {
    default: '#0a1929',    // Dark background
    paper: '#1e2936',      // Card background
    elevated: '#2d3843',   // Elevated surfaces
  },
  text: {
    primary: '#ffffff',
    secondary: 'rgba(255, 255, 255, 0.7)',
    disabled: 'rgba(255, 255, 255, 0.38)',
  }
};

// Typography configuration
const typography = {
  fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  h1: {
    fontSize: '2.5rem',
    fontWeight: 600,
    lineHeight: 1.2,
  },
  h2: {
    fontSize: '2rem',
    fontWeight: 600,
    lineHeight: 1.3,
  },
  h3: {
    fontSize: '1.75rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  h4: {
    fontSize: '1.5rem',
    fontWeight: 500,
    lineHeight: 1.4,
  },
  h5: {
    fontSize: '1.25rem',
    fontWeight: 500,
    lineHeight: 1.5,
  },
  h6: {
    fontSize: '1rem',
    fontWeight: 500,
    lineHeight: 1.6,
  },
  body1: {
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  body2: {
    fontSize: '0.875rem',
    lineHeight: 1.6,
  },
  caption: {
    fontSize: '0.75rem',
    lineHeight: 1.4,
  },
  button: {
    fontSize: '0.875rem',
    fontWeight: 500,
    textTransform: 'none' as const,
  },
};

// Spacing configuration
const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

// Component overrides
const componentOverrides = {
  MuiCssBaseline: {
    styleOverrides: {
      body: {
        backgroundColor: colors.background.default,
        color: colors.text.primary,
      },
    },
  },
  MuiCard: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.paper,
        borderRadius: 12,
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        border: `1px solid rgba(255, 255, 255, 0.1)`,
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        textTransform: 'none',
        fontWeight: 500,
        padding: '8px 16px',
      },
      contained: {
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        '&:hover': {
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.15)',
        },
      },
    },
  },
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          backgroundColor: colors.background.elevated,
          borderRadius: 8,
        },
      },
    },
  },
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 6,
        fontWeight: 500,
      },
    },
  },
  MuiTabs: {
    styleOverrides: {
      root: {
        borderBottom: `1px solid rgba(255, 255, 255, 0.1)`,
      },
      indicator: {
        backgroundColor: colors.primary,
        height: 3,
        borderRadius: '3px 3px 0 0',
      },
    },
  },
  MuiTab: {
    styleOverrides: {
      root: {
        textTransform: 'none',
        fontWeight: 500,
        minHeight: 48,
        '&.Mui-selected': {
          color: colors.primary,
        },
      },
    },
  },
  MuiTableHead: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.elevated,
      },
    },
  },
  MuiTableRow: {
    styleOverrides: {
      root: {
        '&:nth-of-type(even)': {
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
        },
        '&:hover': {
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
        },
      },
    },
  },
  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundColor: colors.background.paper,
        borderRight: `1px solid rgba(255, 255, 255, 0.1)`,
      },
    },
  },
  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.paper,
        borderBottom: `1px solid rgba(255, 255, 255, 0.1)`,
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundColor: colors.background.paper,
      },
    },
  },
};

// Create theme options
const themeOptions: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: {
      main: colors.primary,
      light: '#42a5f5',
      dark: '#1565c0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: colors.secondary,
      light: '#ff5983',
      dark: '#9a0036',
      contrastText: '#ffffff',
    },
    success: {
      main: colors.success,
      light: '#81c784',
      dark: '#388e3c',
      contrastText: '#ffffff',
    },
    warning: {
      main: colors.warning,
      light: '#ffb74d',
      dark: '#f57c00',
      contrastText: '#000000',
    },
    error: {
      main: colors.error,
      light: '#e57373',
      dark: '#d32f2f',
      contrastText: '#ffffff',
    },
    info: {
      main: colors.info,
      light: '#64b5f6',
      dark: '#1976d2',
      contrastText: '#ffffff',
    },
    background: {
      default: colors.background.default,
      paper: colors.background.paper,
    },
    text: {
      primary: colors.text.primary,
      secondary: colors.text.secondary,
      disabled: colors.text.disabled,
    },
    divider: 'rgba(255, 255, 255, 0.1)',
  },
  typography,
  spacing: (factor: number) => `${factor * 8}px`,
  shape: {
    borderRadius: 8,
  },
  components: componentOverrides as any, // Type assertion to bypass version compatibility issue
};

// Create and export theme
export const theme = createTheme(themeOptions);

// Export color palette and spacing for use in components
export { colors, spacing };

// Status color mapping
export const statusColors = {
  success: colors.success,
  warning: colors.warning,
  error: colors.error,
  info: colors.info,
  primary: colors.primary,
  secondary: colors.secondary,
};

// Chart color palette
export const chartColors = [
  colors.primary,
  colors.secondary,
  colors.success,
  colors.warning,
  colors.info,
  '#9c27b0', // Purple
  '#ff5722', // Deep orange
  '#607d8b', // Blue grey
  '#795548', // Brown
  '#e91e63', // Pink
];

// Breakpoints
export const breakpoints = {
  xs: 0,
  sm: 600,
  md: 900,
  lg: 1200,
  xl: 1536,
};

// Z-index scale
export const zIndex = {
  appBar: 1100,
  drawer: 1200,
  modal: 1300,
  snackbar: 1400,
  tooltip: 1500,
};

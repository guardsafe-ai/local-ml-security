import React from 'react';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  Typography,
  Divider,
  Box,
  Tooltip,
  useTheme,
} from '@mui/material';
import {
  Dashboard,
  ModelTraining,
  Storage,
  Analytics,
  Business,
  Security,
  Monitor,
  Memory,
  Science,
  Settings,
  ExpandLess,
  ExpandMore,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { styled } from '@mui/material/styles';

interface SidebarProps {
  open: boolean;
}

const StyledListItemButton = styled(ListItemButton)(({ theme }) => ({
  minHeight: 48,
  justifyContent: 'center',
  px: 2.5,
  '&.Mui-selected': {
    backgroundColor: theme.palette.primary.main + '20',
    '&:hover': {
      backgroundColor: theme.palette.primary.main + '30',
    },
  },
}));

const StyledListItemText = styled(ListItemText)(({ theme }) => ({
  '& .MuiListItemText-primary': {
    fontSize: '0.875rem',
    fontWeight: 500,
  },
}));

const navigationItems = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: <Dashboard />,
    path: '/',
  },
  {
    id: 'models',
    label: 'Models',
    icon: <ModelTraining />,
    path: '/models',
  },
  {
    id: 'training',
    label: 'Training',
    icon: <Science />,
    path: '/training',
  },
  {
    id: 'data',
    label: 'Data',
    icon: <Storage />,
    path: '/data',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: <Analytics />,
    path: '/analytics',
  },
  {
    id: 'business',
    label: 'Business',
    icon: <Business />,
    path: '/business',
  },
  {
    id: 'privacy',
    label: 'Privacy',
    icon: <Security />,
    path: '/privacy',
  },
  {
    id: 'monitoring',
    label: 'Monitoring',
    icon: <Monitor />,
    path: '/monitoring',
  },
  {
    id: 'cache',
    label: 'Cache',
    icon: <Memory />,
    path: '/cache',
  },
  {
    id: 'experiments',
    label: 'Experiments',
    icon: <Science />,
    path: '/experiments',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: <Settings />,
    path: '/settings',
  },
];

const Sidebar: React.FC<SidebarProps> = ({ open }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo/Brand */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'flex-start' : 'center',
          minHeight: 64,
        }}
      >
        {open ? (
          <Typography variant="h6" fontWeight={600} color="primary">
            ML Security
          </Typography>
        ) : (
          <Typography variant="h6" fontWeight={600} color="primary">
            MS
          </Typography>
        )}
      </Box>

      <Divider />

      {/* Navigation */}
      <List sx={{ flexGrow: 1, px: 1, py: 2 }}>
        {navigationItems.map((item) => {
          const active = isActive(item.path);
          
          return (
            <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
              {open ? (
                <StyledListItemButton
                  selected={active}
                  onClick={() => handleNavigation(item.path)}
                  sx={{
                    borderRadius: 1,
                    '&.Mui-selected': {
                      backgroundColor: theme.palette.primary.main + '20',
                      '&:hover': {
                        backgroundColor: theme.palette.primary.main + '30',
                      },
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 40,
                      color: active ? 'primary.main' : 'text.secondary',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <StyledListItemText
                    primary={item.label}
                    sx={{
                      '& .MuiListItemText-primary': {
                        color: active ? 'primary.main' : 'text.primary',
                        fontWeight: active ? 600 : 500,
                      },
                    }}
                  />
                </StyledListItemButton>
              ) : (
                <Tooltip title={item.label} placement="right" arrow>
                  <StyledListItemButton
                    selected={active}
                    onClick={() => handleNavigation(item.path)}
                    sx={{
                      borderRadius: 1,
                      justifyContent: 'center',
                      '&.Mui-selected': {
                        backgroundColor: theme.palette.primary.main + '20',
                        '&:hover': {
                          backgroundColor: theme.palette.primary.main + '30',
                        },
                      },
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        minWidth: 'auto',
                        color: active ? 'primary.main' : 'text.secondary',
                      }}
                    >
                      {item.icon}
                    </ListItemIcon>
                  </StyledListItemButton>
                </Tooltip>
              )}
            </ListItem>
          );
        })}
      </List>

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 2 }}>
        {open && (
          <Typography variant="caption" color="text.secondary" textAlign="center">
            v1.0.0
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default Sidebar;

import React, { useState } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  ModelTraining as ModelIcon,
  Analytics as AnalyticsIcon,
  Security as SecurityIcon,
  Business as BusinessIcon,
  Privacy as PrivacyIcon,
  Monitor as MonitorIcon,
  Science as ScienceIcon,
  Settings as SettingsIcon,
  ExpandLess,
  ExpandMore,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

interface NavigationMenuProps {
  open: boolean;
  onClose: () => void;
}

const NavigationMenu: React.FC<NavigationMenuProps> = ({ open, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  const handleItemClick = (path: string) => {
    navigate(path);
    onClose();
  };

  const handleExpandClick = (item: string) => {
    setExpandedItems(prev => 
      prev.includes(item) 
        ? prev.filter(i => i !== item)
        : [...prev, item]
    );
  };

  const menuItems = [
    {
      title: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/',
      children: [
        { title: 'Overview', path: '/' },
        { title: 'Enhanced Dashboard', path: '/enhanced-dashboard' },
      ]
    },
    {
      title: 'Models',
      icon: <ModelIcon />,
      path: '/models',
      children: [
        { title: 'Model Registry', path: '/models/registry' },
        { title: 'Model Management', path: '/models/management' },
        { title: 'Model Testing', path: '/models/testing' },
        { title: 'Model Performance', path: '/models/performance' },
      ]
    },
    {
      title: 'Training',
      icon: <ModelIcon />,
      path: '/training',
      children: [
        { title: 'Training Queue', path: '/training/queue' },
        { title: 'Data Management', path: '/training/data' },
        { title: 'Data Augmentation', path: '/training/augmentation' },
        { title: 'Training Config', path: '/training/config' },
      ]
    },
    {
      title: 'Analytics',
      icon: <AnalyticsIcon />,
      path: '/analytics',
      children: [
        { title: 'Analytics Dashboard', path: '/analytics' },
        { title: 'Performance Analytics', path: '/analytics/performance' },
        { title: 'Drift Detection', path: '/analytics/drift' },
        { title: 'Advanced Analytics', path: '/analytics/advanced' },
      ]
    },
    {
      title: 'Red Team',
      icon: <SecurityIcon />,
      path: '/red-team',
      children: [
        { title: 'Red Team Dashboard', path: '/red-team' },
        { title: 'Attack Simulation', path: '/red-team/simulation' },
        { title: 'Vulnerability Assessment', path: '/red-team/vulnerability' },
        { title: 'Compliance Testing', path: '/red-team/compliance' },
      ]
    },
    {
      title: 'Business Metrics',
      icon: <BusinessIcon />,
      path: '/business',
      children: [
        { title: 'Business Metrics', path: '/business/metrics' },
        { title: 'Cost Analysis', path: '/business/cost' },
        { title: 'Resource Utilization', path: '/business/resources' },
        { title: 'Performance KPIs', path: '/business/kpis' },
      ]
    },
    {
      title: 'Data Privacy',
      icon: <PrivacyIcon />,
      path: '/privacy',
      children: [
        { title: 'Data Privacy', path: '/privacy' },
        { title: 'Data Classification', path: '/privacy/classification' },
        { title: 'Privacy Compliance', path: '/privacy/compliance' },
      ]
    },
    {
      title: 'Monitoring',
      icon: <MonitorIcon />,
      path: '/monitoring',
      children: [
        { title: 'System Monitoring', path: '/monitoring' },
        { title: 'Performance Monitoring', path: '/monitoring/performance' },
        { title: 'Alert Management', path: '/monitoring/alerts' },
      ]
    },
    {
      title: 'Experiments',
      icon: <ScienceIcon />,
      path: '/experiments',
      children: [
        { title: 'MLflow Experiments', path: '/experiments/mlflow' },
        { title: 'Model Experiments', path: '/experiments/models' },
      ]
    },
    {
      title: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
    },
  ];

  const renderMenuItem = (item: any, level: number = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.title);
    const isActive = location.pathname === item.path || 
      (item.children && item.children.some((child: any) => location.pathname === child.path));

    return (
      <React.Fragment key={item.title}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => hasChildren ? handleExpandClick(item.title) : handleItemClick(item.path)}
            sx={{
              pl: 2 + level * 2,
              backgroundColor: isActive ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
              '&:hover': {
                backgroundColor: 'rgba(25, 118, 210, 0.04)',
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText 
              primary={item.title}
              primaryTypographyProps={{
                fontSize: level > 0 ? '0.875rem' : '0.95rem',
                fontWeight: isActive ? 600 : 400,
              }}
            />
            {hasChildren && (
              isExpanded ? <ExpandLess /> : <ExpandMore />
            )}
          </ListItemButton>
        </ListItem>
        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children.map((child: any) => renderMenuItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <Drawer
      variant="temporary"
      anchor="left"
      open={open}
      onClose={onClose}
      sx={{
        width: 280,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 280,
          boxSizing: 'border-box',
          backgroundColor: '#fafafa',
        },
      }}
    >
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 600, color: '#1976d2' }}>
            ML Security Dashboard
          </Typography>
          <IconButton onClick={onClose} size="small">
            <MenuIcon />
          </IconButton>
        </Box>
      </Box>
      
      <List sx={{ pt: 1 }}>
        {menuItems.map((item) => renderMenuItem(item))}
      </List>
      
      <Divider sx={{ mt: 'auto' }} />
      
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Enterprise ML Security Platform
        </Typography>
      </Box>
    </Drawer>
  );
};

export default NavigationMenu;

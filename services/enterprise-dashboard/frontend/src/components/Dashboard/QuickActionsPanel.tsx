import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Button,
  ButtonGroup,
  Chip,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Science,
  Analytics,
  Security,
  Settings,
  PlayArrow,
  Stop,
  Refresh,
  Download,
  Upload,
  Assessment,
  Timeline,
  Memory,
  CloudSync,
  MoreVert,
} from '@mui/icons-material';

interface QuickAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  variant: 'contained' | 'outlined' | 'text';
  onClick: () => void;
  disabled?: boolean;
  tooltip?: string;
}

interface QuickActionsPanelProps {
  onAction?: (actionId: string) => void;
  loading?: boolean;
}

const QuickActionsPanel: React.FC<QuickActionsPanelProps> = ({
  onAction,
  loading = false,
}) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleAction = (actionId: string) => {
    onAction?.(actionId);
    handleMenuClose();
  };

  const primaryActions: QuickAction[] = [
    {
      id: 'run_security_scan',
      label: 'Security Scan',
      icon: <Security />,
      color: 'error',
      variant: 'contained',
      onClick: () => handleAction('run_security_scan'),
      tooltip: 'Run comprehensive security scan',
    },
    {
      id: 'start_training',
      label: 'Start Training',
      icon: <PlayArrow />,
      color: 'primary',
      variant: 'contained',
      onClick: () => handleAction('start_training'),
      tooltip: 'Start new model training job',
    },
    {
      id: 'generate_report',
      label: 'Generate Report',
      icon: <Assessment />,
      color: 'secondary',
      variant: 'outlined',
      onClick: () => handleAction('generate_report'),
      tooltip: 'Generate system performance report',
    },
  ];

  const secondaryActions: QuickAction[] = [
    {
      id: 'refresh_data',
      label: 'Refresh Data',
      icon: <Refresh />,
      color: 'primary',
      variant: 'text',
      onClick: () => handleAction('refresh_data'),
      tooltip: 'Refresh all dashboard data',
    },
    {
      id: 'export_data',
      label: 'Export Data',
      icon: <Download />,
      color: 'secondary',
      variant: 'text',
      onClick: () => handleAction('export_data'),
      tooltip: 'Export dashboard data',
    },
    {
      id: 'import_config',
      label: 'Import Config',
      icon: <Upload />,
      color: 'secondary',
      variant: 'text',
      onClick: () => handleAction('import_config'),
      tooltip: 'Import configuration',
    },
  ];

  const menuActions: QuickAction[] = [
    {
      id: 'configure_alerts',
      label: 'Configure Alerts',
      icon: <Settings />,
      color: 'primary',
      variant: 'text',
      onClick: () => handleAction('configure_alerts'),
    },
    {
      id: 'view_logs',
      label: 'View Logs',
      icon: <Timeline />,
      color: 'secondary',
      variant: 'text',
      onClick: () => handleAction('view_logs'),
    },
    {
      id: 'system_status',
      label: 'System Status',
      icon: <Memory />,
      color: 'info',
      variant: 'text',
      onClick: () => handleAction('system_status'),
    },
    {
      id: 'sync_data',
      label: 'Sync Data',
      icon: <CloudSync />,
      color: 'success',
      variant: 'text',
      onClick: () => handleAction('sync_data'),
    },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="div">
            Quick Actions
          </Typography>
          <Tooltip title="More Actions">
            <IconButton
              size="small"
              onClick={handleMenuClick}
              disabled={loading}
              sx={{ opacity: loading ? 0.5 : 1 }}
            >
              <MoreVert />
            </IconButton>
          </Tooltip>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* Primary Actions */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Primary Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {primaryActions.map((action) => (
                <Tooltip key={action.id} title={action.tooltip || action.label}>
                  <Button
                    variant={action.variant}
                    color={action.color}
                    startIcon={action.icon}
                    onClick={action.onClick}
                    disabled={action.disabled || loading}
                    fullWidth
                    sx={{ justifyContent: 'flex-start' }}
                  >
                    {action.label}
                  </Button>
                </Tooltip>
              ))}
            </Box>
          </Box>

          <Divider />

          {/* Secondary Actions */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Data Management
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {secondaryActions.map((action) => (
                <Tooltip key={action.id} title={action.tooltip || action.label}>
                  <Button
                    variant={action.variant}
                    color={action.color}
                    startIcon={action.icon}
                    onClick={action.onClick}
                    disabled={action.disabled || loading}
                    fullWidth
                    sx={{ justifyContent: 'flex-start' }}
                  >
                    {action.label}
                  </Button>
                </Tooltip>
              ))}
            </Box>
          </Box>

          <Divider />

          {/* Status Indicators */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              System Status
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip
                icon={<Science />}
                label="Models Ready"
                color="success"
                size="small"
                variant="outlined"
              />
              <Chip
                icon={<Analytics />}
                label="Training Active"
                color="primary"
                size="small"
                variant="outlined"
              />
              <Chip
                icon={<Security />}
                label="Security OK"
                color="success"
                size="small"
                variant="outlined"
              />
            </Box>
          </Box>
        </Box>

        {/* More Actions Menu */}
        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          {menuActions.map((action) => (
            <MenuItem key={action.id} onClick={() => handleAction(action.id)}>
              <ListItemIcon>{action.icon}</ListItemIcon>
              <ListItemText>{action.label}</ListItemText>
            </MenuItem>
          ))}
        </Menu>
      </CardContent>
    </Card>
  );
};

export default QuickActionsPanel;

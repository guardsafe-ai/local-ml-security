import React from 'react';
import {
  Box,
  Typography,
  Chip,
  LinearProgress,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  Refresh,
  Wifi,
  WifiOff,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  responseTime?: number;
  lastCheck?: string;
}

interface StatusBarProps {
  services?: ServiceStatus[];
  connectionStatus?: 'connected' | 'disconnected' | 'connecting';
  onRefresh?: () => void;
}

const StatusBarContainer = styled(Box)(({ theme }) => ({
  position: 'fixed',
  bottom: 0,
  left: 0,
  right: 0,
  height: 32,
  backgroundColor: theme.palette.background.paper,
  borderTop: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 16px',
  zIndex: theme.zIndex.appBar - 1,
}));

const ServiceChip = styled(Chip)(({ theme }) => ({
  height: 20,
  fontSize: '0.75rem',
  marginRight: theme.spacing(1),
  '& .MuiChip-icon': {
    fontSize: '0.875rem',
  },
}));

const StatusBar: React.FC<StatusBarProps> = ({
  services = [],
  connectionStatus = 'connected',
  onRefresh,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle fontSize="small" />;
      case 'warning':
        return <Warning fontSize="small" />;
      case 'error':
        return <Error fontSize="small" />;
      default:
        return <Info fontSize="small" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi fontSize="small" color="success" />;
      case 'disconnected':
        return <WifiOff fontSize="small" color="error" />;
      case 'connecting':
        return <Wifi fontSize="small" color="warning" />;
      default:
        return <WifiOff fontSize="small" color="disabled" />;
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'disconnected':
        return 'Disconnected';
      case 'connecting':
        return 'Connecting...';
      default:
        return 'Unknown';
    }
  };

  const healthyServices = services.filter(s => s.status === 'healthy').length;
  const totalServices = services.length;
  const healthPercentage = totalServices > 0 ? (healthyServices / totalServices) * 100 : 0;

  return (
    <StatusBarContainer>
      {/* Left side - Services status */}
      <Box display="flex" alignItems="center">
        <Typography variant="caption" color="text.secondary" mr={1}>
          Services:
        </Typography>
        {services.slice(0, 5).map((service) => (
          <Tooltip
            key={service.name}
            title={`${service.name} - ${service.status}${service.responseTime ? ` (${service.responseTime}ms)` : ''}`}
            arrow
          >
            <ServiceChip
              icon={getStatusIcon(service.status)}
              label={service.name}
              color={getStatusColor(service.status) as any}
              size="small"
            />
          </Tooltip>
        ))}
        {services.length > 5 && (
          <Typography variant="caption" color="text.secondary">
            +{services.length - 5} more
          </Typography>
        )}
      </Box>

      {/* Center - Health progress */}
      <Box display="flex" alignItems="center" flex={1} mx={2}>
        <Typography variant="caption" color="text.secondary" mr={1}>
          Health:
        </Typography>
        <Box flex={1} mr={1}>
          <LinearProgress
            variant="determinate"
            value={healthPercentage}
            sx={{
              height: 4,
              borderRadius: 2,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                borderRadius: 2,
              },
            }}
          />
        </Box>
        <Typography variant="caption" color="text.secondary">
          {healthPercentage.toFixed(0)}%
        </Typography>
      </Box>

      {/* Right side - Connection status */}
      <Box display="flex" alignItems="center">
        {getConnectionIcon()}
        <Typography variant="caption" color="text.secondary" ml={0.5}>
          {getConnectionText()}
        </Typography>
        {onRefresh && (
          <Tooltip title="Refresh status">
            <IconButton size="small" onClick={onRefresh} sx={{ ml: 1 }}>
              <Refresh fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </StatusBarContainer>
  );
};

export default StatusBar;

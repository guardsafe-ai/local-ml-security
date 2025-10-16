import React from 'react';
import {
  Card,
  CardContent,
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
  Refresh,
  Speed,
  Memory,
  CloudSync,
} from '@mui/icons-material';

interface ServiceHealthCardProps {
  service: {
    name: string;
    status: string;
    response_time: number;
    last_check: string;
    details?: any;
  };
  onRefresh?: () => void;
  loading?: boolean;
}

const ServiceHealthCard: React.FC<ServiceHealthCardProps> = ({
  service,
  onRefresh,
  loading = false,
}) => {
  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return 'success';
      case 'warning':
      case 'degraded':
        return 'warning';
      case 'error':
      case 'unhealthy':
      case 'down':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'warning':
      case 'degraded':
        return <Warning color="warning" />;
      case 'error':
      case 'unhealthy':
      case 'down':
        return <Error color="error" />;
      default:
        return <Warning color="warning" />;
    }
  };

  const formatResponseTime = (time: number) => {
    if (time < 1000) {
      return `${time.toFixed(0)}ms`;
    }
    return `${(time / 1000).toFixed(2)}s`;
  };

  const getResponseTimeColor = (time: number) => {
    if (time < 500) return 'success';
    if (time < 1000) return 'warning';
    return 'error';
  };

  const getServiceIcon = (serviceName: string) => {
    switch (serviceName.toLowerCase()) {
      case 'model-api':
        return <Memory color="primary" />;
      case 'training':
        return <CloudSync color="primary" />;
      case 'analytics':
        return <Speed color="primary" />;
      case 'business-metrics':
        return <Memory color="primary" />;
      case 'data-privacy':
        return <Memory color="primary" />;
      default:
        return <Memory color="primary" />;
    }
  };

  return (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getServiceIcon(service.name)}
            <Typography variant="h6" component="div">
              {service.name}
            </Typography>
          </Box>
          {onRefresh && (
            <Tooltip title="Refresh Service">
              <IconButton
                size="small"
                onClick={onRefresh}
                disabled={loading}
                sx={{ opacity: loading ? 0.5 : 1 }}
              >
                <Refresh />
              </IconButton>
            </Tooltip>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          {getStatusIcon(service.status)}
          <Chip
            label={service.status}
            color={getStatusColor(service.status) as any}
            size="small"
            variant="outlined"
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Speed fontSize="small" color="action" />
            <Typography variant="body2" color="text.secondary">
              Response Time
            </Typography>
          </Box>
          <Typography
            variant="h6"
            color={`${getResponseTimeColor(service.response_time)}.main`}
          >
            {formatResponseTime(service.response_time)}
          </Typography>
        </Box>

        {service.details && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary">
              Last Check: {new Date(service.last_check).toLocaleString()}
            </Typography>
            {service.details.error && (
              <Typography variant="caption" color="error" display="block">
                Error: {service.details.error}
              </Typography>
            )}
          </Box>
        )}

        {loading && (
          <LinearProgress
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
            }}
          />
        )}
      </CardContent>
    </Card>
  );
};

export default ServiceHealthCard;

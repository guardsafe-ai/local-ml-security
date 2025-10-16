import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Box,
  Typography,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Security,
  Science,
  Analytics,
  Business,
  TrendingUp,
  TrendingDown,
  Refresh,
  CheckCircle,
  Warning,
  Error,
} from '@mui/icons-material';

interface MetricCardProps {
  title: string;
  value: number | string;
  subtitle?: string;
  trend?: {
    value: number;
    direction: 'up' | 'down' | 'neutral';
  };
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  progress?: number;
  status?: 'excellent' | 'good' | 'degraded' | 'unhealthy';
  onRefresh?: () => void;
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  icon,
  color,
  progress,
  status,
  onRefresh,
  loading = false,
}) => {
  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'up':
        return <TrendingUp color="success" fontSize="small" />;
      case 'down':
        return <TrendingDown color="error" fontSize="small" />;
      default:
        return null;
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'excellent':
        return <CheckCircle color="success" fontSize="small" />;
      case 'good':
        return <CheckCircle color="success" fontSize="small" />;
      case 'degraded':
        return <Warning color="warning" fontSize="small" />;
      case 'unhealthy':
        return <Error color="error" fontSize="small" />;
      default:
        return null;
    }
  };

  return (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {icon}
            <Typography variant="h6" component="div">
              {title}
            </Typography>
          </Box>
          {onRefresh && (
            <Tooltip title="Refresh Metric">
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

        <Typography variant="h4" color={`${color}.main`} sx={{ mb: 1 }}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </Typography>

        {subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {subtitle}
          </Typography>
        )}

        {progress !== undefined && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress
              variant="determinate"
              value={progress}
              color={color}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              {progress.toFixed(1)}%
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {trend && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {getTrendIcon(trend.direction)}
                <Typography
                  variant="body2"
                  color={trend.direction === 'up' ? 'success.main' : 'error.main'}
                >
                  {trend.value > 0 ? '+' : ''}{trend.value}%
                </Typography>
              </Box>
            )}
            {status && (
              <Chip
                icon={getStatusIcon(status)}
                label={status}
                size="small"
                color={status === 'excellent' || status === 'good' ? 'success' : 'warning'}
                variant="outlined"
              />
            )}
          </Box>
        </Box>

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

interface SystemMetricsGridProps {
  metrics: {
    total_models: number;
    active_jobs: number;
    total_attacks: number;
    detection_rate: number;
    system_health: number;
  };
  onRefresh?: () => void;
  loading?: boolean;
}

const SystemMetricsGrid: React.FC<SystemMetricsGridProps> = ({
  metrics,
  onRefresh,
  loading = false,
}) => {
  const getSystemHealthStatus = (health: number) => {
    if (health >= 90) return 'excellent';
    if (health >= 75) return 'good';
    if (health >= 50) return 'degraded';
    return 'unhealthy';
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Total Models"
          value={metrics.total_models}
          subtitle="Available Models"
          icon={<Science color="primary" />}
          color="primary"
          onRefresh={onRefresh}
          loading={loading}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Active Jobs"
          value={metrics.active_jobs}
          subtitle="Training Jobs"
          icon={<Analytics color="secondary" />}
          color="secondary"
          onRefresh={onRefresh}
          loading={loading}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="Security Tests"
          value={metrics.total_attacks}
          subtitle="Red Team Attacks"
          icon={<Security color="error" />}
          color="error"
          onRefresh={onRefresh}
          loading={loading}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <MetricCard
          title="System Health"
          value={`${metrics.system_health.toFixed(1)}%`}
          subtitle="Overall Status"
          icon={<Business color="success" />}
          color="success"
          progress={metrics.system_health}
          status={getSystemHealthStatus(metrics.system_health)}
          onRefresh={onRefresh}
          loading={loading}
        />
      </Grid>
    </Grid>
  );
};

export default SystemMetricsGrid;

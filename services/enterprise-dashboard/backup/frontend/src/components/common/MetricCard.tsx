import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  MoreVert,
  Info,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: {
    value: number;
    direction: 'up' | 'down' | 'flat';
    period?: string;
  };
  status?: 'success' | 'warning' | 'error' | 'info' | 'primary' | 'secondary';
  loading?: boolean;
  progress?: number;
  icon?: React.ReactNode;
  description?: string;
  actions?: React.ReactNode;
  onClick?: () => void;
  className?: string;
}

const StyledCard = styled(Card, {
  shouldForwardProp: (prop) => prop !== 'clickable',
})<{ clickable?: boolean }>(({ theme, clickable }) => ({
  cursor: clickable ? 'pointer' : 'default',
  transition: 'all 0.2s ease-in-out',
  '&:hover': clickable ? {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[8],
  } : {},
}));

const TrendIcon = ({ direction }: { direction: 'up' | 'down' | 'flat' }) => {
  switch (direction) {
    case 'up':
      return <TrendingUp color="success" fontSize="small" />;
    case 'down':
      return <TrendingDown color="error" fontSize="small" />;
    case 'flat':
      return <TrendingFlat color="info" fontSize="small" />;
    default:
      return null;
  }
};

const StatusChip = styled(Chip)(({ theme }) => ({
  height: 24,
  fontSize: '0.75rem',
  fontWeight: 500,
}));

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  status,
  loading = false,
  progress,
  icon,
  description,
  actions,
  onClick,
  className,
}) => {
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      case 'info':
        return 'info';
      case 'primary':
        return 'primary';
      case 'secondary':
        return 'secondary';
      default:
        return 'default';
    }
  };

  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      if (val >= 1000000) {
        return `${(val / 1000000).toFixed(1)}M`;
      } else if (val >= 1000) {
        return `${(val / 1000).toFixed(1)}K`;
      }
      return val.toLocaleString();
    }
    return val;
  };

  return (
    <StyledCard
      clickable={!!onClick}
      onClick={onClick}
      className={className}
      sx={{ height: '100%' }}
    >
      <CardContent sx={{ p: 2 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
          <Box display="flex" alignItems="center" gap={1}>
            {icon && (
              <Box color="primary.main">
                {icon}
              </Box>
            )}
            <Typography variant="body2" color="text.secondary" fontWeight={500}>
              {title}
            </Typography>
            {description && (
              <Tooltip title={description} arrow>
                <IconButton size="small" sx={{ p: 0.5 }}>
                  <Info fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            {status && (
              <StatusChip
                label={status.toUpperCase()}
                color={getStatusColor(status) as any}
                size="small"
              />
            )}
            {actions && (
              <IconButton size="small">
                <MoreVert fontSize="small" />
              </IconButton>
            )}
          </Box>
        </Box>

        <Box mb={1}>
          <Typography variant="h4" component="div" fontWeight={600}>
            {loading ? '...' : formatValue(value)}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>

        {progress !== undefined && (
          <Box mb={1}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 3,
                },
              }}
            />
            <Typography variant="caption" color="text.secondary" mt={0.5}>
              {progress.toFixed(1)}%
            </Typography>
          </Box>
        )}

        {trend && (
          <Box display="flex" alignItems="center" gap={0.5}>
            <TrendIcon direction={trend.direction} />
            <Typography
              variant="body2"
              color={
                trend.direction === 'up' ? 'success.main' :
                trend.direction === 'down' ? 'error.main' :
                'text.secondary'
              }
              fontWeight={500}
            >
              {Math.abs(trend.value).toFixed(1)}%
            </Typography>
            {trend.period && (
              <Typography variant="caption" color="text.secondary">
                {trend.period}
              </Typography>
            )}
          </Box>
        )}
      </CardContent>
    </StyledCard>
  );
};

export default MetricCard;

import React from 'react';
import { Box, Chip } from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Help as HelpIcon,
} from '@mui/icons-material';

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  label?: string;
  showIcon?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  showIcon = true,
  size = 'medium',
}) => {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'healthy':
        return {
          icon: <CheckCircleIcon />,
          color: '#4caf50',
          backgroundColor: 'rgba(76, 175, 80, 0.2)',
          borderColor: '#4caf50',
          label: 'Healthy',
        };
      case 'warning':
        return {
          icon: <WarningIcon />,
          color: '#ff9800',
          backgroundColor: 'rgba(255, 152, 0, 0.2)',
          borderColor: '#ff9800',
          label: 'Warning',
        };
      case 'error':
        return {
          icon: <ErrorIcon />,
          color: '#f44336',
          backgroundColor: 'rgba(244, 67, 54, 0.2)',
          borderColor: '#f44336',
          label: 'Error',
        };
      default:
        return {
          icon: <HelpIcon />,
          color: '#9e9e9e',
          backgroundColor: 'rgba(158, 158, 158, 0.2)',
          borderColor: '#9e9e9e',
          label: 'Unknown',
        };
    }
  };

  const config = getStatusConfig(status);
  const displayLabel = label || config.label;

  const getSizeConfig = (size: string) => {
    switch (size) {
      case 'small':
        return {
          chipSize: 'small' as const,
          iconSize: 16,
          typography: 'caption' as const,
        };
      case 'large':
        return {
          chipSize: 'medium' as const,
          iconSize: 24,
          typography: 'body1' as const,
        };
      default:
        return {
          chipSize: 'small' as const,
          iconSize: 20,
          typography: 'body2' as const,
        };
    }
  };

  const sizeConfig = getSizeConfig(size);

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {showIcon && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: sizeConfig.iconSize + 8,
            height: sizeConfig.iconSize + 8,
            borderRadius: '50%',
            backgroundColor: config.backgroundColor,
            color: config.color,
          }}
        >
          {React.cloneElement(config.icon, { sx: { fontSize: sizeConfig.iconSize } })}
        </Box>
      )}
      
      <Chip
        label={displayLabel}
        size={sizeConfig.chipSize}
        sx={{
          backgroundColor: config.backgroundColor,
          color: config.color,
          border: `1px solid ${config.borderColor}`,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}
      />
    </Box>
  );
};

export default StatusIndicator;

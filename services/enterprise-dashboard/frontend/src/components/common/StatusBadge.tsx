import React from 'react';
import { Chip, ChipProps } from '@mui/material';
import { styled } from '@mui/material/styles';

interface StatusBadgeProps extends Omit<ChipProps, 'color'> {
  status: 'success' | 'warning' | 'error' | 'info' | 'primary' | 'secondary' | 'default';
  variant?: 'filled' | 'outlined';
  size?: 'small' | 'medium';
  showIcon?: boolean;
}

const StatusChip = styled(Chip)<{ status: string }>(({ theme, status }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return theme.palette.success.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'info':
        return theme.palette.info.main;
      case 'primary':
        return theme.palette.primary.main;
      case 'secondary':
        return theme.palette.secondary.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const color = getStatusColor(status);

  return {
    backgroundColor: `${color}20`,
    color: color,
    borderColor: color,
    fontWeight: 500,
    '& .MuiChip-icon': {
      color: color,
    },
  };
});

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  variant = 'filled',
  size = 'small',
  showIcon = false,
  label,
  ...props
}) => {
  const getStatusIcon = (status: string) => {
    if (!showIcon) return undefined;
    
    switch (status) {
      case 'success':
        return '✓';
      case 'warning':
        return '⚠';
      case 'error':
        return '✗';
      case 'info':
        return 'i';
      case 'primary':
        return '●';
      case 'secondary':
        return '●';
      default:
        return '●';
    }
  };

  const getStatusLabel = (status: string, originalLabel?: React.ReactNode) => {
    if (originalLabel) return originalLabel;
    
    switch (status) {
      case 'success':
        return 'Success';
      case 'warning':
        return 'Warning';
      case 'error':
        return 'Error';
      case 'info':
        return 'Info';
      case 'primary':
        return 'Active';
      case 'secondary':
        return 'Inactive';
      default:
        return 'Unknown';
    }
  };

  return (
    <StatusChip
      status={status}
      variant={variant}
      size={size}
      icon={getStatusIcon(status)}
      label={getStatusLabel(status, label)}
      {...props}
    />
  );
};

export default StatusBadge;

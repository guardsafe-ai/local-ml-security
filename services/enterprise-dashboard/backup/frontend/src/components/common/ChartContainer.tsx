import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Box,
  Typography,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  MoreVert,
  Download,
  Fullscreen,
  Refresh,
  Settings,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface ChartContainerProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  height?: number | string;
  loading?: boolean;
  error?: string;
  actions?: React.ReactNode;
  onRefresh?: () => void;
  onDownload?: () => void;
  onFullscreen?: () => void;
  onSettings?: () => void;
  className?: string;
}

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartContent = styled(Box)<{ height?: number | string }>(({ height }) => ({
  height: height || 300,
  width: '100%',
  position: 'relative',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const LoadingOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1,
}));

const ErrorOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: theme.palette.error.dark + '20',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1,
  border: `1px solid ${theme.palette.error.main}`,
  borderRadius: theme.shape.borderRadius,
}));

export const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  subtitle,
  children,
  height = 300,
  loading = false,
  error,
  actions,
  onRefresh,
  onDownload,
  onFullscreen,
  onSettings,
  className,
}) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleAction = (action: () => void) => {
    action();
    handleMenuClose();
  };

  const defaultActions = (
    <>
      {onRefresh && (
        <Tooltip title="Refresh">
          <IconButton size="small" onClick={onRefresh}>
            <Refresh />
          </IconButton>
        </Tooltip>
      )}
      {onDownload && (
        <Tooltip title="Download">
          <IconButton size="small" onClick={onDownload}>
            <Download />
          </IconButton>
        </Tooltip>
      )}
      {onFullscreen && (
        <Tooltip title="Fullscreen">
          <IconButton size="small" onClick={onFullscreen}>
            <Fullscreen />
          </IconButton>
        </Tooltip>
      )}
      {(onSettings || onDownload || onFullscreen) && (
        <Tooltip title="More options">
          <IconButton size="small" onClick={handleMenuClick}>
            <MoreVert />
          </IconButton>
        </Tooltip>
      )}
    </>
  );

  return (
    <StyledCard className={className}>
      <CardHeader
        title={
          <Typography variant="h6" component="div" fontWeight={600}>
            {title}
          </Typography>
        }
        subheader={
          subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )
        }
        action={actions || defaultActions}
        sx={{ pb: 1 }}
      />
      <CardContent sx={{ pt: 0, flex: 1, display: 'flex', flexDirection: 'column' }}>
        <ChartContent height={height}>
          {loading && (
            <LoadingOverlay>
              <Typography variant="body2" color="text.secondary">
                Loading...
              </Typography>
            </LoadingOverlay>
          )}
          {error && (
            <ErrorOverlay>
              <Box textAlign="center">
                <Typography variant="body2" color="error.main" mb={1}>
                  Error loading chart
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {error}
                </Typography>
              </Box>
            </ErrorOverlay>
          )}
          {!loading && !error && children}
        </ChartContent>
      </CardContent>

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
        {onRefresh && (
          <MenuItem onClick={() => handleAction(onRefresh)}>
            <ListItemIcon>
              <Refresh fontSize="small" />
            </ListItemIcon>
            <ListItemText>Refresh</ListItemText>
          </MenuItem>
        )}
        {onDownload && (
          <MenuItem onClick={() => handleAction(onDownload)}>
            <ListItemIcon>
              <Download fontSize="small" />
            </ListItemIcon>
            <ListItemText>Download</ListItemText>
          </MenuItem>
        )}
        {onFullscreen && (
          <MenuItem onClick={() => handleAction(onFullscreen)}>
            <ListItemIcon>
              <Fullscreen fontSize="small" />
            </ListItemIcon>
            <ListItemText>Fullscreen</ListItemText>
          </MenuItem>
        )}
        {onSettings && (
          <MenuItem onClick={() => handleAction(onSettings)}>
            <ListItemIcon>
              <Settings fontSize="small" />
            </ListItemIcon>
            <ListItemText>Settings</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </StyledCard>
  );
};

export default ChartContainer;

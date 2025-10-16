import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  Avatar,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  Refresh,
  Science,
  Security,
  Analytics,
  Business,
  Timeline,
} from '@mui/icons-material';

interface ActivityItem {
  id: string;
  type: 'training' | 'red_team' | 'model' | 'system' | 'analytics';
  message: string;
  timestamp: string;
  status: 'success' | 'error' | 'warning' | 'info';
  details?: any;
}

interface ActivityTimelineProps {
  activities: ActivityItem[];
  onRefresh?: () => void;
  loading?: boolean;
  maxHeight?: number;
}

const ActivityTimeline: React.FC<ActivityTimelineProps> = ({
  activities,
  onRefresh,
  loading = false,
  maxHeight = 400,
}) => {
  const getActivityIcon = (type: string, status: string) => {
    const iconProps = { fontSize: 'small' as const };
    
    switch (type) {
      case 'training':
        return <Science color="primary" {...iconProps} />;
      case 'red_team':
        return <Security color="error" {...iconProps} />;
      case 'model':
        return <Analytics color="secondary" {...iconProps} />;
      case 'system':
        return <Timeline color="info" {...iconProps} />;
      case 'analytics':
        return <Analytics color="success" {...iconProps} />;
      default:
        return <Info {...iconProps} />;
    }
  };

  const getStatusIcon = (status: string) => {
    const iconProps = { fontSize: 'small' as const };
    
    switch (status) {
      case 'success':
        return <CheckCircle color="success" {...iconProps} />;
      case 'error':
        return <Error color="error" {...iconProps} />;
      case 'warning':
        return <Warning color="warning" {...iconProps} />;
      case 'info':
        return <Info color="info" {...iconProps} />;
      default:
        return <Info color="info" {...iconProps} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const getActivityTypeColor = (type: string) => {
    switch (type) {
      case 'training':
        return 'primary';
      case 'red_team':
        return 'error';
      case 'model':
        return 'secondary';
      case 'system':
        return 'info';
      case 'analytics':
        return 'success';
      default:
        return 'default';
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline color="primary" />
            <Typography variant="h6" component="div">
              Recent Activity
            </Typography>
          </Box>
          {onRefresh && (
            <Tooltip title="Refresh Activity">
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

        <Box sx={{ maxHeight, overflow: 'auto' }}>
          {activities.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary">
                No recent activity
              </Typography>
            </Box>
          ) : (
            <List dense>
              {activities.map((activity, index) => (
                <React.Fragment key={activity.id}>
                  <ListItem
                    sx={{
                      py: 1,
                      px: 0,
                      '&:hover': {
                        backgroundColor: 'action.hover',
                        borderRadius: 1,
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40 }}>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          bgcolor: `${getActivityTypeColor(activity.type)}.light`,
                        }}
                      >
                        {getActivityIcon(activity.type, activity.status)}
                      </Avatar>
                    </ListItemIcon>
                    
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography variant="body2" component="span">
                            {activity.message}
                          </Typography>
                          <Chip
                            icon={getStatusIcon(activity.status)}
                            label={activity.status}
                            size="small"
                            color={getStatusColor(activity.status) as any}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            {formatTimestamp(activity.timestamp)}
                          </Typography>
                          <Chip
                            label={activity.type}
                            size="small"
                            color={getActivityTypeColor(activity.type) as any}
                            variant="outlined"
                          />
                        </Box>
                      }
                    />
                  </ListItem>
                  
                  {index < activities.length - 1 && <Divider variant="inset" component="li" />}
                </React.Fragment>
              ))}
            </List>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ActivityTimeline;

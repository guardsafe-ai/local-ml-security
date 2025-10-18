import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Card,
  CardContent,
  Chip,
  Grid,
  Fade,
  Zoom,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Collapse,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  PlayArrow,
  CheckCircle,
  Error,
  Download,
  Memory,
  Speed,
  Storage,
  ExpandMore,
  ExpandLess,
  Refresh,
  BugReport,
  Timeline
} from '@mui/icons-material';
import { apiService } from '../services/apiService';

interface MLPipelineProgressProps {
  modelName: string;
  progress: {
    status: string;
    progress: number;
    message: string;
    timestamp: string;
    stage: string;
    details: {
      pipeline_stage?: string;
      cache_status?: string;
      model_source?: string;
      download_speed?: string;
      memory_usage?: string;
      model_size?: string;
    };
  };
  onRefresh?: () => void;
}

const MLPipelineProgress: React.FC<MLPipelineProgressProps> = ({
  modelName,
  progress,
  onRefresh
}) => {
  const [expanded, setExpanded] = React.useState(false);
  const [logs, setLogs] = React.useState<string[]>([]);
  const [modelCacheLogs, setModelCacheLogs] = React.useState<any[]>([]);
  const [activeTab, setActiveTab] = React.useState(0);
  const [loadingCacheLogs, setLoadingCacheLogs] = React.useState(false);

  // Add new log entries from model-api progress
  React.useEffect(() => {
    const newLog = `[${new Date(progress.timestamp).toLocaleTimeString()}] ${progress.stage}: ${progress.message}`;
    setLogs(prev => [...prev.slice(-9), newLog]); // Keep last 10 logs
  }, [progress]);

  // Fetch model-cache logs
  const fetchModelCacheLogs = async () => {
    try {
      setLoadingCacheLogs(true);
      const response = await apiService.getModelCacheModelLogs(modelName, 50);
      setModelCacheLogs(response.logs || []);
    } catch (error) {
      console.error('Failed to fetch model-cache logs:', error);
    } finally {
      setLoadingCacheLogs(false);
    }
  };

  // Fetch logs when component mounts or when expanded
  useEffect(() => {
    if (expanded) {
      fetchModelCacheLogs();
      // Set up interval to fetch logs every 2 seconds when expanded
      const interval = setInterval(fetchModelCacheLogs, 2000);
      return () => clearInterval(interval);
    }
  }, [expanded, modelName]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'loading': return 'info';
      case 'downloading': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'error': return <Error />;
      case 'loading': return <Memory />;
      case 'downloading': return <Download />;
      default: return <PlayArrow />;
    }
  };

  const getCacheStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'success';
      case 'loaded': return 'info';
      case 'checking': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  return (
    <Card sx={{ mb: 2, overflow: 'visible' }}>
      <CardContent>
        {/* Header with Model Name and Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Zoom in={true}>
              {getStatusIcon(progress.status)}
            </Zoom>
            <Typography variant="h6" component="div">
              {modelName}
            </Typography>
            <Chip
              label={progress.stage}
              color={getStatusColor(progress.status) as any}
              size="small"
              icon={getStatusIcon(progress.status)}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {onRefresh && (
              <IconButton size="small" onClick={onRefresh}>
                <Refresh />
              </IconButton>
            )}
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Pipeline Progress
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {progress.progress}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={progress.progress}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: 'rgba(0,0,0,0.1)',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
                background: progress.status === 'error' 
                  ? 'linear-gradient(45deg, #f44336, #ff5722)'
                  : progress.status === 'completed'
                  ? 'linear-gradient(45deg, #4caf50, #8bc34a)'
                  : 'linear-gradient(45deg, #2196f3, #03a9f4)'
              }
            }}
          />
        </Box>

        {/* Current Message */}
        <Alert 
          severity={progress.status === 'error' ? 'error' : progress.status === 'completed' ? 'success' : 'info'}
          sx={{ mb: 2 }}
        >
          <Typography variant="body2">
            {progress.message}
          </Typography>
        </Alert>

        {/* Collapsible Details */}
        <Collapse in={expanded}>
          <Divider sx={{ mb: 2 }} />
          
          {/* Performance Metrics */}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6}>
              <Card variant="outlined" sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Storage color="primary" />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Cache Status
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      <Chip
                        label={progress.details?.cache_status || 'Unknown'}
                        color={getCacheStatusColor(progress.details?.cache_status || '') as any}
                        size="small"
                      />
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined" sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Speed color="primary" />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Download Speed
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {progress.details?.download_speed || 'N/A'}
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined" sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Memory color="primary" />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Memory Usage
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {progress.details?.memory_usage || 'N/A'}
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined" sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Storage color="primary" />
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Model Size
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {progress.details?.model_size || 'N/A'}
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
          </Grid>

          {/* Pipeline Logs with Tabs */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Pipeline Logs
            </Typography>
            
            <Tabs 
              value={activeTab} 
              onChange={(_, newValue) => setActiveTab(newValue)}
              sx={{ mb: 2 }}
            >
              <Tab 
                icon={<Timeline />} 
                label="Model API" 
                iconPosition="start"
                sx={{ minHeight: 48 }}
              />
              <Tab 
                icon={<BugReport />} 
                label="Model Cache" 
                iconPosition="start"
                sx={{ minHeight: 48 }}
              />
            </Tabs>

            <Card variant="outlined" sx={{ maxHeight: 200, overflow: 'auto' }}>
              {activeTab === 0 ? (
                // Model API Logs
                <List dense>
                  {logs.map((log, index) => (
                    <Fade key={index} in={true} timeout={300}>
                      <ListItem>
                        <ListItemText
                          primary={log}
                          primaryTypographyProps={{ 
                            variant: 'caption',
                            fontFamily: 'monospace',
                            color: index === logs.length - 1 ? 'primary.main' : 'text.secondary'
                          }}
                        />
                      </ListItem>
                    </Fade>
                  ))}
                </List>
              ) : (
                // Model Cache Logs
                <List dense>
                  {loadingCacheLogs ? (
                    <ListItem>
                      <ListItemText
                        primary="Loading model-cache logs..."
                        primaryTypographyProps={{ 
                          variant: 'caption',
                          fontFamily: 'monospace',
                          color: 'text.secondary'
                        }}
                      />
                    </ListItem>
                  ) : modelCacheLogs.length === 0 ? (
                    <ListItem>
                      <ListItemText
                        primary="No model-cache logs available"
                        primaryTypographyProps={{ 
                          variant: 'caption',
                          fontFamily: 'monospace',
                          color: 'text.secondary'
                        }}
                      />
                    </ListItem>
                  ) : (
                    modelCacheLogs.map((log, index) => (
                      <Fade key={index} in={true} timeout={300}>
                        <ListItem>
                          <ListItemIcon>
                            {log.level === 'ERROR' ? <Error color="error" /> :
                             log.level === 'WARNING' ? <Error color="warning" /> :
                             <CheckCircle color="success" />}
                          </ListItemIcon>
                          <ListItemText
                            primary={`[${new Date(log.timestamp * 1000).toLocaleTimeString()}] [${log.level}] ${log.message}`}
                            secondary={log.details && Object.keys(log.details).length > 0 ? 
                              `Details: ${JSON.stringify(log.details)}` : undefined}
                            primaryTypographyProps={{ 
                              variant: 'caption',
                              fontFamily: 'monospace',
                              color: log.level === 'ERROR' ? 'error.main' : 
                                     log.level === 'WARNING' ? 'warning.main' : 'text.secondary'
                            }}
                            secondaryTypographyProps={{ 
                              variant: 'caption',
                              fontFamily: 'monospace',
                              color: 'text.secondary'
                            }}
                          />
                        </ListItem>
                      </Fade>
                    ))
                  )}
                </List>
              )}
            </Card>
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default MLPipelineProgress;

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  Snackbar,
  CircularProgress
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  ContentCopy,
  Download,
  Close,
  Refresh
} from '@mui/icons-material';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  source: string;
}

interface JobLogsModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobId: string;
  jobName: string;
  logs: LogEntry[];
  isLoading?: boolean;
  onRefresh?: () => void;
}

const JobLogsModal: React.FC<JobLogsModalProps> = ({
  isOpen,
  onClose,
  jobId,
  jobName,
  logs,
  isLoading = false,
  onRefresh
}) => {
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const getLogIcon = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return <Error color="error" />;
      case 'WARNING':
        return <Warning color="warning" />;
      case 'INFO':
        return <Info color="info" />;
      case 'SUCCESS':
        return <CheckCircle color="success" />;
      default:
        return <Info color="info" />;
    }
  };

  const getLogChipColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'error';
      case 'WARNING':
        return 'warning';
      case 'INFO':
        return 'info';
      case 'SUCCESS':
        return 'success';
      default:
        return 'default';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  const copyLogsToClipboard = async () => {
    try {
      const logText = logs.map(log => 
        `[${formatTimestamp(log.timestamp)}] ${log.level}: ${log.message} (${log.source})`
      ).join('\n');
      
      await navigator.clipboard.writeText(logText);
      setSnackbarMessage('Logs copied to clipboard!');
      setSnackbarOpen(true);
    } catch (error) {
      console.error('Failed to copy logs:', error);
      setSnackbarMessage('Failed to copy logs');
      setSnackbarOpen(true);
    }
  };

  const downloadLogs = () => {
    try {
      const logText = logs.map(log => 
        `[${formatTimestamp(log.timestamp)}] ${log.level}: ${log.message} (${log.source})`
      ).join('\n');
      
      const blob = new Blob([logText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `job-${jobId}-logs.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setSnackbarMessage('Logs downloaded!');
      setSnackbarOpen(true);
    } catch (error) {
      console.error('Failed to download logs:', error);
      setSnackbarMessage('Failed to download logs');
      setSnackbarOpen(true);
    }
  };

  return (
    <>
      <Dialog
        open={isOpen}
        onClose={onClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { height: '80vh' }
        }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6">
                Training Job Logs
              </Typography>
              <Chip 
                label={jobName} 
                color="primary" 
                size="small" 
              />
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Tooltip title="Refresh Logs">
                <IconButton onClick={onRefresh} disabled={isLoading}>
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Close">
                <IconButton onClick={onClose}>
                  <Close />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </DialogTitle>

        <DialogContent dividers>
          {isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
              <CircularProgress />
              <Typography variant="body2" sx={{ ml: 2 }}>
                Loading logs...
              </Typography>
            </Box>
          ) : logs.length === 0 ? (
            <Alert severity="info">
              No logs available for this job.
            </Alert>
          ) : (
            <Paper sx={{ maxHeight: '50vh', overflow: 'auto' }}>
              <List dense>
                {logs.map((log, index) => (
                  <React.Fragment key={index}>
                    <ListItem alignItems="flex-start">
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        {getLogIcon(log.level)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">
                              {formatTimestamp(log.timestamp)}
                            </Typography>
                            <Chip
                              label={log.level}
                              color={getLogChipColor(log.level) as any}
                              size="small"
                              variant="outlined"
                            />
                            <Typography variant="caption" color="text.secondary">
                              {log.source}
                            </Typography>
                          </Box>
                        }
                        secondary={
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {log.message}
                          </Typography>
                        }
                      />
                    </ListItem>
                    {index < logs.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </Paper>
          )}
        </DialogContent>

        <DialogActions>
          <Button
            startIcon={<ContentCopy />}
            onClick={copyLogsToClipboard}
            disabled={logs.length === 0}
          >
            Copy Logs
          </Button>
          <Button
            startIcon={<Download />}
            onClick={downloadLogs}
            disabled={logs.length === 0}
          >
            Download
          </Button>
          <Button onClick={onClose}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
      />
    </>
  );
};

export default JobLogsModal;

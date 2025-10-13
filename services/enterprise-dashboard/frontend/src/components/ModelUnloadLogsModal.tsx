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
  Snackbar
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  ContentCopy,
  Download,
  Close
} from '@mui/icons-material';

interface LogEntry {
  level: 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  message: string;
  timestamp: number;
}

interface ModelUnloadLogsModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelName: string;
  logs: LogEntry[];
  isSuccess: boolean;
}

const getLogIcon = (level: string) => {
  switch (level) {
    case 'INFO':
      return <Info color="primary" />;
    case 'WARNING':
      return <Warning color="warning" />;
    case 'ERROR':
      return <Error color="error" />;
    case 'SUCCESS':
      return <CheckCircle color="success" />;
    default:
      return <Info color="action" />;
  }
};

const getLogChipColor = (level: string) => {
  switch (level) {
    case 'INFO':
      return 'primary';
    case 'WARNING':
      return 'warning';
    case 'ERROR':
      return 'error';
    case 'SUCCESS':
      return 'success';
    default:
      return 'default';
  }
};

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp * 1000).toLocaleTimeString();
};

export const ModelUnloadLogsModal: React.FC<ModelUnloadLogsModalProps> = ({
  isOpen,
  onClose,
  modelName,
  logs,
  isSuccess,
}) => {
  const [copied, setCopied] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const copyLogsToClipboard = async () => {
    const logText = logs
      .map(log => `[${formatTimestamp(log.timestamp)}] ${log.level}: ${log.message}`)
      .join('\n');
    
    try {
      await navigator.clipboard.writeText(logText);
      setCopied(true);
      setSnackbarMessage('Logs copied to clipboard');
      setSnackbarOpen(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      setSnackbarMessage('Failed to copy logs');
      setSnackbarOpen(true);
    }
  };

  const downloadLogs = () => {
    const logText = logs
      .map(log => `[${formatTimestamp(log.timestamp)}] ${log.level}: ${log.message}`)
      .join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `model-unload-logs-${modelName}-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setSnackbarMessage('Logs downloaded');
    setSnackbarOpen(true);
  };

  return (
    <>
      <Dialog 
        open={isOpen} 
        onClose={onClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { maxHeight: '80vh' }
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isSuccess ? (
            <CheckCircle color="success" />
          ) : (
            <Error color="error" />
          )}
          <Typography variant="h6">
            Model Unload Logs: {modelName}
          </Typography>
        </DialogTitle>
        
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {isSuccess 
              ? 'Model has been successfully unloaded from memory. Below are the detailed logs of the unload process.'
              : 'Model unload failed. Below are the error logs and details.'
            }
          </Typography>

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={copyLogsToClipboard}
                disabled={logs.length === 0}
                startIcon={<ContentCopy />}
              >
                {copied ? 'Copied!' : 'Copy Logs'}
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={downloadLogs}
                disabled={logs.length === 0}
                startIcon={<Download />}
              >
                Download
              </Button>
            </Box>
            <Chip 
              label={isSuccess ? 'Success' : 'Failed'} 
              color={isSuccess ? 'success' : 'error'}
              size="small"
            />
          </Box>

          {/* Logs Display */}
          <Paper sx={{ maxHeight: 400, overflow: 'auto', p: 1 }}>
            {logs.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                No logs available
              </Box>
            ) : (
              <List dense>
                {logs.map((log, index) => (
                  <ListItem key={index} sx={{ alignItems: 'flex-start', py: 1 }}>
                    <ListItemIcon sx={{ minWidth: 40, mt: 0.5 }}>
                      {getLogIcon(log.level)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Chip 
                            label={log.level} 
                            size="small" 
                            color={getLogChipColor(log.level) as any}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {formatTimestamp(log.timestamp)}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                          {log.message}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>

          {/* Memory Cleanup Summary */}
          {isSuccess && (
            <Alert 
              severity="success" 
              sx={{ mt: 2 }}
              icon={<CheckCircle />}
            >
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Memory Cleanup Summary
              </Typography>
              <Box component="ul" sx={{ m: 0, pl: 2 }}>
                <li>Model object removed from memory</li>
                <li>Tokenizer object cleared</li>
                <li>Model removed from active models dictionary</li>
                <li>Cache service notified and updated</li>
                <li>Statistics updated (load count decremented)</li>
              </Box>
            </Alert>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={onClose} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
      />
    </>
  );
};

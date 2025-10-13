import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Tooltip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Badge,
} from '@mui/material';
import {
  Science,
  PlayArrow,
  Stop,
  Pause,
  Refresh,
  Add,
  Settings,
  Visibility,
  Edit,
  Delete,
  Download,
  Upload,
  Timeline,
  Assessment,
  Queue,
  CheckCircle,
  Error,
  Warning,
  Info,
  ExpandMore,
  Schedule,
  Memory,
  Speed,
} from '@mui/icons-material';
import MetricCard from '../../components/MetricCard/MetricCard';
import { LineChart, BarChart } from '../../components/Charts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`training-tabpanel-${index}`}
      aria-labelledby={`training-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Mock data
const mockTrainingJobs = [
  {
    id: '1',
    name: 'BERT Security Training',
    status: 'running',
    progress: 65,
    modelName: 'BERT-base',
    dataset: 'security-dataset-v2',
    startTime: '2024-01-20 10:30:00',
    estimatedEnd: '2024-01-20 14:30:00',
    duration: '2h 15m',
    accuracy: 0.94,
    loss: 0.12,
    epoch: 15,
    totalEpochs: 25,
    learningRate: 0.0001,
    batchSize: 32,
  },
  {
    id: '2',
    name: 'DistilBERT Fine-tuning',
    status: 'completed',
    progress: 100,
    modelName: 'DistilBERT',
    dataset: 'security-dataset-v1',
    startTime: '2024-01-19 09:00:00',
    endTime: '2024-01-19 12:30:00',
    duration: '3h 30m',
    accuracy: 0.96,
    loss: 0.08,
    epoch: 20,
    totalEpochs: 20,
    learningRate: 0.0001,
    batchSize: 16,
  },
  {
    id: '3',
    name: 'RoBERTa Large Training',
    status: 'failed',
    progress: 45,
    modelName: 'RoBERTa-large',
    dataset: 'security-dataset-v3',
    startTime: '2024-01-18 14:00:00',
    endTime: '2024-01-18 16:45:00',
    duration: '2h 45m',
    accuracy: 0.89,
    loss: 0.25,
    epoch: 9,
    totalEpochs: 30,
    learningRate: 0.00005,
    batchSize: 8,
    error: 'CUDA out of memory',
  },
  {
    id: '4',
    name: 'BERT Security v2',
    status: 'pending',
    progress: 0,
    modelName: 'BERT-base',
    dataset: 'security-dataset-v2',
    startTime: '2024-01-21 08:00:00',
    estimatedEnd: '2024-01-21 16:00:00',
    duration: '0h 0m',
    accuracy: 0,
    loss: 0,
    epoch: 0,
    totalEpochs: 25,
    learningRate: 0.0001,
    batchSize: 32,
  },
];

const mockConfigurations = [
  {
    id: '1',
    name: 'BERT Security Config',
    modelName: 'BERT-base',
    dataset: 'security-dataset-v2',
    epochs: 25,
    learningRate: 0.0001,
    batchSize: 32,
    maxSequenceLength: 512,
    warmupSteps: 1000,
    weightDecay: 0.01,
    createdAt: '2024-01-15',
    lastUsed: '2024-01-20',
    description: 'Standard configuration for BERT security training',
  },
  {
    id: '2',
    name: 'DistilBERT Fast Config',
    modelName: 'DistilBERT',
    dataset: 'security-dataset-v1',
    epochs: 20,
    learningRate: 0.0001,
    batchSize: 16,
    maxSequenceLength: 256,
    warmupSteps: 500,
    weightDecay: 0.01,
    createdAt: '2024-01-10',
    lastUsed: '2024-01-19',
    description: 'Fast training configuration for DistilBERT',
  },
];

const mockQueue = [
  { id: '1', name: 'BERT Security v2', priority: 'high', estimatedStart: '2024-01-21 08:00:00', estimatedDuration: '8h' },
  { id: '2', name: 'RoBERTa Retry', priority: 'medium', estimatedStart: '2024-01-21 16:00:00', estimatedDuration: '6h' },
  { id: '3', name: 'DistilBERT v3', priority: 'low', estimatedStart: '2024-01-22 08:00:00', estimatedDuration: '4h' },
];

const Training: React.FC = () => {
  const [value, setValue] = useState(0);
  const [newJobDialogOpen, setNewJobDialogOpen] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [selectedConfig, setSelectedConfig] = useState<string>('');
  const [jobName, setJobName] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleStartJob = (jobId: string) => {
    console.log('Starting job:', jobId);
  };

  const handleStopJob = (jobId: string) => {
    console.log('Stopping job:', jobId);
  };

  const handlePauseJob = (jobId: string) => {
    console.log('Pausing job:', jobId);
  };

  const handleDeleteJob = (jobId: string) => {
    console.log('Deleting job:', jobId);
  };

  const handleCreateConfig = () => {
    setConfigDialogOpen(true);
  };

  const handleNewJob = () => {
    setNewJobDialogOpen(true);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <PlayArrow color="success" />;
      case 'completed':
        return <CheckCircle color="success" />;
      case 'failed':
        return <Error color="error" />;
      case 'pending':
        return <Schedule color="warning" />;
      default:
        return <Info color="info" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'info';
      default:
        return 'default';
    }
  };

  const steps = [
    'Select Configuration',
    'Choose Model & Dataset',
    'Review & Start',
  ];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Training Center
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage model training jobs, configurations, and MLflow experiments
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<Settings />} onClick={handleCreateConfig}>
            Manage Configs
          </Button>
          <Button variant="contained" startIcon={<Add />} onClick={handleNewJob}>
            New Training Job
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Jobs"
            value={mockTrainingJobs.filter(j => j.status === 'running').length}
            icon={<PlayArrow />}
            color="#4caf50"
            trend={2}
            subtitle="currently running"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Completed Jobs"
            value={mockTrainingJobs.filter(j => j.status === 'completed').length}
            icon={<CheckCircle />}
            color="#1976d2"
            trend={8}
            subtitle="this week"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Queue Length"
            value={mockQueue.length}
            icon={<Queue />}
            color="#ff9800"
            trend={-1}
            subtitle="pending jobs"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Training Time"
            value="4.2h"
            icon={<Timeline />}
            color="#9c27b0"
            trend={-0.5}
            subtitle="per job"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="training center tabs">
            <Tab icon={<PlayArrow />} label="Training Jobs" />
            <Tab icon={<Settings />} label="Configurations" />
            <Tab icon={<Queue />} label="Queue Management" />
            <Tab icon={<Timeline />} label="Training History" />
            <Tab icon={<Assessment />} label="Performance" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <TextField
              placeholder="Search jobs..."
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Visibility />
                  </InputAdornment>
                ),
              }}
              sx={{ minWidth: 300 }}
            />
            <Box display="flex" gap={1}>
              <IconButton>
                <Refresh />
              </IconButton>
              <IconButton>
                <Download />
              </IconButton>
            </Box>
          </Box>

          <Grid container spacing={3}>
            {mockTrainingJobs.map((job) => (
              <Grid item xs={12} md={6} lg={4} key={job.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {job.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {job.modelName} • {job.dataset}
                        </Typography>
                      </Box>
                      <Box display="flex" alignItems="center" gap={1}>
                        {getStatusIcon(job.status)}
                        <Chip
                          label={job.status.toUpperCase()}
                          color={getStatusColor(job.status) as any}
                          size="small"
                        />
                      </Box>
                    </Box>

                    {job.status === 'running' && (
                      <Box mb={2}>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2" color="text.secondary">
                            Progress
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {job.progress}%
                          </Typography>
                        </Box>
                        <LinearProgress variant="determinate" value={job.progress} />
                        <Box display="flex" justifyContent="space-between" mt={1}>
                          <Typography variant="caption" color="text.secondary">
                            Epoch {job.epoch}/{job.totalEpochs}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {job.duration}
                          </Typography>
                        </Box>
                      </Box>
                    )}

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {(job.accuracy * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Accuracy
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {job.loss.toFixed(3)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Loss
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {job.batchSize}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Batch Size
                        </Typography>
                      </Box>
                    </Box>

                    {job.error && (
                      <Alert severity="error" sx={{ mb: 2 }}>
                        {job.error}
                      </Alert>
                    )}

                    <Box display="flex" gap={1} mt={2}>
                      {job.status === 'running' && (
                        <>
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<Pause />}
                            onClick={() => handlePauseJob(job.id)}
                            fullWidth
                          >
                            Pause
                          </Button>
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<Stop />}
                            onClick={() => handleStopJob(job.id)}
                            fullWidth
                          >
                            Stop
                          </Button>
                        </>
                      )}
                      {job.status === 'pending' && (
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<PlayArrow />}
                          onClick={() => handleStartJob(job.id)}
                          fullWidth
                        >
                          Start
                        </Button>
                      )}
                      {job.status === 'completed' && (
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Visibility />}
                          fullWidth
                        >
                          View Results
                        </Button>
                      )}
                      {job.status === 'failed' && (
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<PlayArrow />}
                          onClick={() => handleStartJob(job.id)}
                          fullWidth
                        >
                          Retry
                        </Button>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <Typography variant="h6" gutterBottom>
            Training Configurations
          </Typography>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="body2" color="text.secondary">
              Manage your training configurations for quick job creation
            </Typography>
            <Button variant="contained" startIcon={<Add />} onClick={handleCreateConfig}>
              New Configuration
            </Button>
          </Box>

          <Grid container spacing={3}>
            {mockConfigurations.map((config) => (
              <Grid item xs={12} md={6} key={config.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {config.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {config.modelName} • {config.dataset}
                        </Typography>
                      </Box>
                      <Box display="flex" gap={1}>
                        <IconButton size="small">
                          <Edit />
                        </IconButton>
                        <IconButton size="small">
                          <Delete />
                        </IconButton>
                      </Box>
                    </Box>

                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {config.description}
                    </Typography>

                    <Grid container spacing={2} mb={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Epochs: {config.epochs}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Learning Rate: {config.learningRate}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Batch Size: {config.batchSize}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Max Length: {config.maxSequenceLength}
                        </Typography>
                      </Grid>
                    </Grid>

                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="caption" color="text.secondary">
                        Last used: {config.lastUsed}
                      </Typography>
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<PlayArrow />}
                        onClick={() => {
                          setSelectedConfig(config.id);
                          setNewJobDialogOpen(true);
                        }}
                      >
                        Use Config
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            Queue Management
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Manage the training job queue and priorities
          </Typography>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Job Name</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>Estimated Start</TableCell>
                  <TableCell>Estimated Duration</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockQueue.map((job) => (
                  <TableRow key={job.id}>
                    <TableCell>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {job.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={job.priority.toUpperCase()}
                        color={getPriorityColor(job.priority) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {job.estimatedStart}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {job.estimatedDuration}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <IconButton size="small">
                          <Edit />
                        </IconButton>
                        <IconButton size="small">
                          <Delete />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
        
        <TabPanel value={value} index={3}>
          <Typography variant="h6" gutterBottom>
            Training History
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Training Progress Over Time
                  </Typography>
                  <LineChart
                    data={[
                      { time: '00:00', accuracy: 0.85, loss: 0.45 },
                      { time: '02:00', accuracy: 0.89, loss: 0.38 },
                      { time: '04:00', accuracy: 0.92, loss: 0.28 },
                      { time: '06:00', accuracy: 0.94, loss: 0.18 },
                      { time: '08:00', accuracy: 0.96, loss: 0.12 },
                    ]}
                    dataKey="accuracy"
                    xAxisKey="time"
                    colors={['#1976d2']}
                    width={600}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Jobs
                  </Typography>
                  <List>
                    {mockTrainingJobs.slice(0, 3).map((job) => (
                      <ListItem key={job.id} divider>
                        <ListItemIcon>
                          {getStatusIcon(job.status)}
                        </ListItemIcon>
                        <ListItemText
                          primary={job.name}
                          secondary={`${job.duration} • ${(job.accuracy * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Performance Analytics
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Training Time by Model
                  </Typography>
                  <BarChart
                    data={[
                      { model: 'BERT', time: 8.5 },
                      { model: 'DistilBERT', time: 4.2 },
                      { model: 'RoBERTa', time: 12.3 },
                    ]}
                    dataKey="time"
                    xAxisKey="model"
                    colors={['#4caf50']}
                    width={400}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Success Rate by Configuration
                  </Typography>
                  <BarChart
                    data={[
                      { config: 'BERT Security', success: 95 },
                      { config: 'DistilBERT Fast', success: 88 },
                      { config: 'RoBERTa Large', success: 75 },
                    ]}
                    dataKey="success"
                    xAxisKey="config"
                    colors={['#ff9800']}
                    width={400}
                    height={300}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* New Job Dialog */}
      <Dialog open={newJobDialogOpen} onClose={() => setNewJobDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Training Job</DialogTitle>
        <DialogContent>
          <Stepper activeStep={activeStep} orientation="vertical">
            <Step>
              <StepLabel>Select Configuration</StepLabel>
              <StepContent>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Configuration</InputLabel>
                  <Select value={selectedConfig} onChange={(e) => setSelectedConfig(e.target.value)}>
                    {mockConfigurations.map((config) => (
                      <MenuItem key={config.id} value={config.id}>
                        {config.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Box mt={2}>
                  <Button
                    variant="contained"
                    onClick={() => setActiveStep(1)}
                    disabled={!selectedConfig}
                  >
                    Next
                  </Button>
                </Box>
              </StepContent>
            </Step>
            <Step>
              <StepLabel>Choose Model & Dataset</StepLabel>
              <StepContent>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Model</InputLabel>
                  <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                    <MenuItem value="bert">BERT-base</MenuItem>
                    <MenuItem value="distilbert">DistilBERT</MenuItem>
                    <MenuItem value="roberta">RoBERTa-large</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Dataset</InputLabel>
                  <Select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>
                    <MenuItem value="security-v1">Security Dataset v1</MenuItem>
                    <MenuItem value="security-v2">Security Dataset v2</MenuItem>
                    <MenuItem value="security-v3">Security Dataset v3</MenuItem>
                  </Select>
                </FormControl>
                <Box mt={2}>
                  <Button
                    variant="contained"
                    onClick={() => setActiveStep(2)}
                    disabled={!selectedModel || !selectedDataset}
                  >
                    Next
                  </Button>
                </Box>
              </StepContent>
            </Step>
            <Step>
              <StepLabel>Review & Start</StepLabel>
              <StepContent>
                <TextField
                  fullWidth
                  label="Job Name"
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  margin="normal"
                />
                <Alert severity="info" sx={{ mt: 2 }}>
                  Review your configuration and click Start to begin training.
                </Alert>
                <Box mt={2}>
                  <Button
                    variant="contained"
                    onClick={() => {
                      console.log('Starting training job');
                      setNewJobDialogOpen(false);
                      setActiveStep(0);
                    }}
                    disabled={!jobName}
                  >
                    Start Training
                  </Button>
                </Box>
              </StepContent>
            </Step>
          </Stepper>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewJobDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Training;
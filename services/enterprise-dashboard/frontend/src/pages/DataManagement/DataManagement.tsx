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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Badge,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  DropzoneArea,
} from '@mui/material';
import {
  Storage,
  Upload,
  Transform,
  CloudUpload,
  Assessment,
  Security,
  Search,
  FilterList,
  Download,
  Refresh,
  Add,
  Settings,
  Visibility,
  Edit,
  Delete,
  CheckCircle,
  Error,
  Warning,
  Info,
  ExpandMore,
  Folder,
  FileCopy,
  DataObject,
  PrivacyTip,
  Shield,
  Analytics,
  Speed,
  Memory,
} from '@mui/icons-material';
import MetricCard from '../../components/MetricCard/MetricCard';
import { LineChart, BarChart, PieChart } from '../../components/Charts';

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
      id={`data-tabpanel-${index}`}
      aria-labelledby={`data-tab-${index}`}
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
const mockDatasets = [
  {
    id: '1',
    name: 'Security Dataset v2',
    description: 'Comprehensive security training dataset',
    size: '2.4GB',
    recordCount: 150000,
    format: 'JSONL',
    privacyLevel: 'Internal',
    qualityScore: 94,
    createdAt: '2024-01-15',
    modifiedAt: '2024-01-20',
    tags: ['security', 'training', 'text'],
    path: '/datasets/security-v2',
  },
  {
    id: '2',
    name: 'Privacy Dataset v1',
    description: 'Privacy-focused dataset for compliance testing',
    size: '1.8GB',
    recordCount: 95000,
    format: 'CSV',
    privacyLevel: 'Confidential',
    qualityScore: 87,
    createdAt: '2024-01-10',
    modifiedAt: '2024-01-18',
    tags: ['privacy', 'compliance', 'pii'],
    path: '/datasets/privacy-v1',
  },
  {
    id: '3',
    name: 'Augmented Dataset v3',
    description: 'Augmented dataset with synthetic data',
    size: '3.2GB',
    recordCount: 200000,
    format: 'Parquet',
    privacyLevel: 'Public',
    qualityScore: 91,
    createdAt: '2024-01-12',
    modifiedAt: '2024-01-19',
    tags: ['augmented', 'synthetic', 'training'],
    path: '/datasets/augmented-v3',
  },
];

const mockMinIOBuckets = [
  {
    name: 'ml-datasets',
    size: '15.2GB',
    objects: 1250,
    lastModified: '2024-01-20 14:30:00',
    region: 'us-east-1',
    versioning: true,
  },
  {
    name: 'model-artifacts',
    size: '8.7GB',
    objects: 340,
    lastModified: '2024-01-19 09:15:00',
    region: 'us-east-1',
    versioning: false,
  },
  {
    name: 'experiments',
    size: '22.1GB',
    objects: 2100,
    lastModified: '2024-01-20 16:45:00',
    region: 'us-east-1',
    versioning: true,
  },
];

const mockQualityMetrics = [
  { metric: 'Completeness', score: 94, status: 'good' },
  { metric: 'Accuracy', score: 89, status: 'warning' },
  { metric: 'Consistency', score: 96, status: 'good' },
  { metric: 'Validity', score: 92, status: 'good' },
  { metric: 'Uniqueness', score: 88, status: 'warning' },
  { metric: 'Timeliness', score: 95, status: 'good' },
];

const mockPrivacyViolations = [
  {
    id: '1',
    type: 'PII Detection',
    severity: 'high',
    description: 'Email addresses found in dataset',
    count: 45,
    dataset: 'Security Dataset v2',
    status: 'open',
    detectedAt: '2024-01-20 10:30:00',
  },
  {
    id: '2',
    type: 'Data Classification',
    severity: 'medium',
    description: 'Unclassified sensitive data detected',
    count: 12,
    dataset: 'Privacy Dataset v1',
    status: 'acknowledged',
    detectedAt: '2024-01-19 15:20:00',
  },
];

const DataManagement: React.FC = () => {
  const [value, setValue] = useState(0);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [augmentationDialogOpen, setAugmentationDialogOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const handleUpload = () => {
    setIsUploading(true);
    setUploadProgress(0);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          setUploadDialogOpen(false);
          setSelectedFiles([]);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  const handleFileDrop = (files: File[]) => {
    setSelectedFiles(files);
  };

  const getPrivacyColor = (level: string) => {
    switch (level) {
      case 'Public':
        return 'success';
      case 'Internal':
        return 'info';
      case 'Confidential':
        return 'warning';
      case 'Restricted':
        return 'error';
      default:
        return 'default';
    }
  };

  const getQualityStatus = (score: number) => {
    if (score >= 90) return { status: 'good', color: 'success' };
    if (score >= 80) return { status: 'warning', color: 'warning' };
    return { status: 'poor', color: 'error' };
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
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

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} gutterBottom>
            Data Management
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive data lifecycle management with privacy and quality controls
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Button variant="outlined" startIcon={<Upload />} onClick={() => setUploadDialogOpen(true)}>
            Upload Data
          </Button>
          <Button variant="contained" startIcon={<Add />}>
            New Dataset
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Datasets"
            value={mockDatasets.length}
            icon={<Storage />}
            color="#1976d2"
            trend={3}
            subtitle="this month"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Size"
            value="7.4GB"
            icon={<Memory />}
            color="#4caf50"
            trend={12}
            subtitle="stored data"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Quality Score"
            value="91%"
            icon={<Assessment />}
            color="#ff9800"
            trend={2}
            subtitle="average"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Privacy Violations"
            value={mockPrivacyViolations.length}
            icon={<Security />}
            color="#f44336"
            trend={-1}
            subtitle="active alerts"
          />
        </Grid>
      </Grid>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={value} onChange={handleChange} aria-label="data management tabs">
            <Tab icon={<Storage />} label="Datasets" />
            <Tab icon={<Upload />} label="Upload" />
            <Tab icon={<Transform />} label="Augmentation" />
            <Tab icon={<CloudUpload />} label="MinIO Storage" />
            <Tab icon={<Assessment />} label="Quality" />
            <Tab icon={<Security />} label="Privacy" />
          </Tabs>
        </Box>
        
        <TabPanel value={value} index={0}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <TextField
              placeholder="Search datasets..."
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              sx={{ minWidth: 300 }}
            />
            <Box display="flex" gap={1}>
              <IconButton>
                <FilterList />
              </IconButton>
              <IconButton>
                <Download />
              </IconButton>
              <IconButton>
                <Refresh />
              </IconButton>
            </Box>
          </Box>

          <Grid container spacing={3}>
            {mockDatasets.map((dataset) => {
              const qualityStatus = getQualityStatus(dataset.qualityScore);
              return (
                <Grid item xs={12} md={6} lg={4} key={dataset.id}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                        <Box>
                          <Typography variant="h6" fontWeight={600}>
                            {dataset.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {dataset.format} • {dataset.recordCount.toLocaleString()} records
                          </Typography>
                        </Box>
                        <Box display="flex" gap={1}>
                          <Chip
                            label={dataset.privacyLevel}
                            color={getPrivacyColor(dataset.privacyLevel) as any}
                            size="small"
                          />
                          <Chip
                            label={`${dataset.qualityScore}%`}
                            color={qualityStatus.color as any}
                            size="small"
                          />
                        </Box>
                      </Box>

                      <Typography variant="body2" color="text.secondary" mb={2}>
                        {dataset.description}
                      </Typography>

                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Box textAlign="center">
                          <Typography variant="h6" color="primary">
                            {dataset.size}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Size
                          </Typography>
                        </Box>
                        <Box textAlign="center">
                          <Typography variant="h6" color="primary">
                            {dataset.recordCount.toLocaleString()}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Records
                          </Typography>
                        </Box>
                        <Box textAlign="center">
                          <Typography variant="h6" color="primary">
                            {dataset.qualityScore}%
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Quality
                          </Typography>
                        </Box>
                      </Box>

                      <Box display="flex" gap={1} mt={2}>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Visibility />}
                          fullWidth
                        >
                          View
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Edit />}
                          fullWidth
                        >
                          Edit
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Delete />}
                          fullWidth
                        >
                          Delete
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <Typography variant="h6" gutterBottom>
            Upload New Data
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Upload datasets in various formats for processing and training
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Drag & Drop Upload
                  </Typography>
                  <Box
                    sx={{
                      border: '2px dashed #ccc',
                      borderRadius: 2,
                      p: 4,
                      textAlign: 'center',
                      backgroundColor: 'background.elevated',
                    }}
                  >
                    <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      Drop files here or click to browse
                    </Typography>
                    <Typography variant="body2" color="text.secondary" mb={2}>
                      Supports JSONL, CSV, Parquet, and TXT files up to 10GB
                    </Typography>
                    <Button variant="contained" component="label">
                      Choose Files
                      <input
                        type="file"
                        hidden
                        multiple
                        onChange={(e) => {
                          if (e.target.files) {
                            setSelectedFiles(Array.from(e.target.files));
                          }
                        }}
                      />
                    </Button>
                  </Box>

                  {selectedFiles.length > 0 && (
                    <Box mt={3}>
                      <Typography variant="h6" gutterBottom>
                        Selected Files
                      </Typography>
                      <List>
                        {selectedFiles.map((file, index) => (
                          <ListItem key={index} divider>
                            <ListItemIcon>
                              <FileCopy />
                            </ListItemIcon>
                            <ListItemText
                              primary={file.name}
                              secondary={`${(file.size / 1024 / 1024).toFixed(2)} MB`}
                            />
                            <IconButton onClick={() => {
                              setSelectedFiles(prev => prev.filter((_, i) => i !== index));
                            }}>
                              <Delete />
                            </IconButton>
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {isUploading && (
                    <Box mt={3}>
                      <Typography variant="body2" color="text.secondary" mb={1}>
                        Uploading... {uploadProgress}%
                      </Typography>
                      <LinearProgress variant="determinate" value={uploadProgress} />
                    </Box>
                  )}

                  <Box display="flex" gap={2} mt={3}>
                    <Button
                      variant="contained"
                      onClick={handleUpload}
                      disabled={selectedFiles.length === 0 || isUploading}
                      fullWidth
                    >
                      Upload Files
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => setSelectedFiles([])}
                      disabled={isUploading}
                    >
                      Clear
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Upload Guidelines
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Supported Formats"
                        secondary="JSONL, CSV, Parquet, TXT"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Max File Size"
                        secondary="10GB per file"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Data Validation"
                        secondary="Automatic format checking"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Privacy Scanning"
                        secondary="PII detection enabled"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <Typography variant="h6" gutterBottom>
            Data Augmentation
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Apply augmentation techniques to enhance your datasets
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Augmentation Techniques
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Transform />
                      </ListItemIcon>
                      <ListItemText
                        primary="Text Augmentation"
                        secondary="Synonym replacement, back translation"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Transform />
                      </ListItemIcon>
                      <ListItemText
                        primary="Synthetic Data Generation"
                        secondary="GPT-based data synthesis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Transform />
                      </ListItemIcon>
                      <ListItemText
                        primary="Data Balancing"
                        secondary="SMOTE, undersampling"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Transform />
                      </ListItemIcon>
                      <ListItemText
                        primary="Noise Injection"
                        secondary="Controlled noise addition"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Augmentations
                  </Typography>
                  <List>
                    <ListItem divider>
                      <ListItemText
                        primary="Security Dataset v2"
                        secondary="Text augmentation • 2x increase"
                      />
                    </ListItem>
                    <ListItem divider>
                      <ListItemText
                        primary="Privacy Dataset v1"
                        secondary="Synthetic generation • 1.5x increase"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Augmented Dataset v3"
                        secondary="Data balancing • 3x increase"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={3}>
          <Typography variant="h6" gutterBottom>
            MinIO Storage Management
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Manage your object storage buckets and files
          </Typography>

          <Grid container spacing={3}>
            {mockMinIOBuckets.map((bucket) => (
              <Grid item xs={12} md={4} key={bucket.name}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                      <Box>
                        <Typography variant="h6" fontWeight={600}>
                          {bucket.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {bucket.region} • {bucket.objects} objects
                        </Typography>
                      </Box>
                      <Chip
                        label={bucket.versioning ? 'Versioned' : 'Standard'}
                        color={bucket.versioning ? 'success' : 'default'}
                        size="small"
                      />
                    </Box>

                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {bucket.size}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Total Size
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {bucket.objects}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Objects
                        </Typography>
                      </Box>
                      <Box textAlign="center">
                        <Typography variant="h6" color="primary">
                          {bucket.lastModified.split(' ')[0]}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Last Modified
                        </Typography>
                      </Box>
                    </Box>

                    <Box display="flex" gap={1} mt={2}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Visibility />}
                        fullWidth
                      >
                        Browse
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Settings />}
                        fullWidth
                      >
                        Settings
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={4}>
          <Typography variant="h6" gutterBottom>
            Data Quality Metrics
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor and improve data quality across all datasets
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quality Score Breakdown
                  </Typography>
                  <BarChart
                    data={mockQualityMetrics}
                    dataKey="score"
                    xAxisKey="metric"
                    colors={['#4caf50']}
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
                    Quality Metrics
                  </Typography>
                  <List>
                    {mockQualityMetrics.map((metric, index) => {
                      const status = getQualityStatus(metric.score);
                      return (
                        <ListItem key={index} divider>
                          <ListItemIcon>
                            {status.status === 'good' ? (
                              <CheckCircle color="success" />
                            ) : status.status === 'warning' ? (
                              <Warning color="warning" />
                            ) : (
                              <Error color="error" />
                            )}
                          </ListItemIcon>
                          <ListItemText
                            primary={metric.metric}
                            secondary={`${metric.score}%`}
                          />
                        </ListItem>
                      );
                    })}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={value} index={5}>
          <Typography variant="h6" gutterBottom>
            Privacy & Compliance
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Monitor privacy violations and ensure compliance
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Privacy Violations
                  </Typography>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Type</TableCell>
                          <TableCell>Severity</TableCell>
                          <TableCell>Count</TableCell>
                          <TableCell>Dataset</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockPrivacyViolations.map((violation) => (
                          <TableRow key={violation.id}>
                            <TableCell>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {violation.type}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={violation.severity.toUpperCase()}
                                color={getSeverityColor(violation.severity) as any}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {violation.count}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {violation.dataset}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={violation.status.toUpperCase()}
                                color={violation.status === 'open' ? 'error' : 'warning'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Box display="flex" gap={1}>
                                <IconButton size="small">
                                  <Visibility />
                                </IconButton>
                                <IconButton size="small">
                                  <Edit />
                                </IconButton>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Compliance Status
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="GDPR Compliance"
                        secondary="95% compliant"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="CCPA Compliance"
                        secondary="92% compliant"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Warning color="warning" />
                      </ListItemIcon>
                      <ListItemText
                        primary="HIPAA Compliance"
                        secondary="87% compliant"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Data Retention"
                        secondary="100% compliant"
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default DataManagement;
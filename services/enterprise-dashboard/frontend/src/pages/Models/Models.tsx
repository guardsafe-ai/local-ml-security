import React from 'react';
import { Box, Typography } from '@mui/material';

const Models: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        Models
      </Typography>
      <Typography variant="body1" color="text.secondary">
        Model management, registry, and performance monitoring
      </Typography>
    </Box>
  );
};

export default Models;
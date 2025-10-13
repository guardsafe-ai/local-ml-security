import React from 'react';
import { Box, Typography } from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight={600} gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary">
        System configuration, preferences, and security settings
      </Typography>
    </Box>
  );
};

export default Settings;
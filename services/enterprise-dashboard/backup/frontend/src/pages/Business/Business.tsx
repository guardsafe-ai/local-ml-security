import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Button, Avatar } from '@mui/material';
import { Speed, TrendingUp, Assessment, MonetizationOn } from '@mui/icons-material';

const Business: React.FC = () => {
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', mb: 1 }}>
            Business Metrics
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Cost analysis and ROI metrics for your ML security system
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<Speed />}>
          Generate Report
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <MonetizationOn />
                </Avatar>
                <Box>
                  <Typography variant="h6">Cost Analysis</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Track operational costs and spending
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Cost analysis will be available here
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'success.main' }}>
                  <TrendingUp />
                </Avatar>
                <Box>
                  <Typography variant="h6">ROI Analysis</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Calculate return on investment
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary">
                ROI analysis will be available here
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Avatar sx={{ bgcolor: 'warning.main' }}>
                  <Assessment />
                </Avatar>
                <Box>
                  <Typography variant="h6">Resource Utilization</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monitor resource efficiency
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Resource utilization will be available here
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Business;


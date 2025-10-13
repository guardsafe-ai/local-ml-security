import React from 'react';
import { Box, Skeleton, Card, CardContent, Grid } from '@mui/material';

interface LoadingSkeletonProps {
  variant?: 'card' | 'table' | 'chart' | 'list' | 'metrics';
  count?: number;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ 
  variant = 'card', 
  count = 1 
}) => {
  const renderCardSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="60%" height={32} />
        <Skeleton variant="text" width="40%" height={24} sx={{ mt: 1 }} />
        <Skeleton variant="rectangular" width="100%" height={60} sx={{ mt: 2 }} />
      </CardContent>
    </Card>
  );

  const renderTableSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
        {Array.from({ length: 5 }).map((_, index) => (
          <Box key={index} sx={{ display: 'flex', gap: 2, mb: 1 }}>
            <Skeleton variant="text" width="25%" height={24} />
            <Skeleton variant="text" width="20%" height={24} />
            <Skeleton variant="text" width="15%" height={24} />
            <Skeleton variant="text" width="20%" height={24} />
            <Skeleton variant="text" width="20%" height={24} />
          </Box>
        ))}
      </CardContent>
    </Card>
  );

  const renderChartSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="40%" height={32} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" width="100%" height={300} />
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, gap: 2 }}>
          <Skeleton variant="circular" width={20} height={20} />
          <Skeleton variant="text" width="60px" height={20} />
          <Skeleton variant="circular" width={20} height={20} />
          <Skeleton variant="text" width="60px" height={20} />
        </Box>
      </CardContent>
    </Card>
  );

  const renderListSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
        {Array.from({ length: 6 }).map((_, index) => (
          <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Skeleton variant="circular" width={40} height={40} sx={{ mr: 2 }} />
            <Box sx={{ flex: 1 }}>
              <Skeleton variant="text" width="60%" height={20} />
              <Skeleton variant="text" width="40%" height={16} />
            </Box>
            <Skeleton variant="rectangular" width={80} height={32} />
          </Box>
        ))}
      </CardContent>
    </Card>
  );

  const renderMetricsSkeleton = () => (
    <Grid container spacing={2}>
      {Array.from({ length: 4 }).map((_, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Skeleton variant="circular" width={24} height={24} sx={{ mr: 1 }} />
                <Skeleton variant="text" width="60%" height={32} />
              </Box>
              <Skeleton variant="text" width="40%" height={20} />
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Skeleton variant="text" width="20px" height={16} />
                <Skeleton variant="text" width="60px" height={16} sx={{ ml: 0.5 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderSkeleton = () => {
    switch (variant) {
      case 'table':
        return renderTableSkeleton();
      case 'chart':
        return renderChartSkeleton();
      case 'list':
        return renderListSkeleton();
      case 'metrics':
        return renderMetricsSkeleton();
      default:
        return renderCardSkeleton();
    }
  };

  return (
    <Box>
      {Array.from({ length: count }).map((_, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          {renderSkeleton()}
        </Box>
      ))}
    </Box>
  );
};

export default LoadingSkeleton;

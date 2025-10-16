import React from 'react';
import { Box, Skeleton, SkeletonProps } from '@mui/material';

interface LoadingSkeletonProps extends SkeletonProps {
  lines?: number;
  height?: number | string;
  width?: number | string;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  lines = 1,
  height = 20,
  width = '100%',
  ...props
}) => {
  if (lines === 1) {
    return <Skeleton height={height} width={width} {...props} />;
  }

  return (
    <Box>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton
          key={index}
          height={height}
          width={width}
          sx={{ mb: 1 }}
          {...props}
        />
      ))}
    </Box>
  );
};

export default LoadingSkeleton;
export { LoadingSkeleton };

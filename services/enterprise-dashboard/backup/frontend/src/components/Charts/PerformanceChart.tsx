import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart
} from 'recharts';
import { Box, Typography, Card, CardContent, Chip } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';

interface PerformanceChartProps {
  data: any[];
  title: string;
  type?: 'line' | 'area';
  height?: number;
  showLegend?: boolean;
  dataKey?: string;
  color?: string;
  isLoading?: boolean;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data = [],
  title,
  type = 'line',
  height = 300,
  showLegend = true,
  dataKey = 'value',
  color = '#00bcd4',
  isLoading = false
}) => {
  if (isLoading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Box sx={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography color="text.secondary">Loading chart...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Box sx={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography color="text.secondary">No data available</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  const calculateTrend = () => {
    if (data.length < 2) return { trend: 'neutral', percentage: 0 };
    
    const first = data[0][dataKey] || 0;
    const last = data[data.length - 1][dataKey] || 0;
    const percentage = first !== 0 ? ((last - first) / first) * 100 : 0;
    
    return {
      trend: percentage > 0 ? 'up' : percentage < 0 ? 'down' : 'neutral',
      percentage: Math.abs(percentage)
    };
  };

  const trend = calculateTrend();

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
            p: 1.5,
            boxShadow: 2
          }}
        >
          <Typography variant="body2" color="text.secondary">
            {label}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              sx={{ color: entry.color }}
            >
              {`${entry.dataKey}: ${entry.value}`}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  const ChartComponent = type === 'area' ? AreaChart : LineChart;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            {title}
          </Typography>
          <Chip
            icon={trend.trend === 'up' ? <TrendingUp /> : trend.trend === 'down' ? <TrendingDown /> : undefined}
            label={`${trend.percentage.toFixed(1)}%`}
            color={trend.trend === 'up' ? 'success' : trend.trend === 'down' ? 'error' : 'default'}
            size="small"
          />
        </Box>
        
        <ResponsiveContainer width="100%" height={height}>
          <ChartComponent data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="name" 
              stroke="#9CA3AF"
              fontSize={12}
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            
            {type === 'area' ? (
              <Area
                type="monotone"
                dataKey={dataKey}
                stroke={color}
                fill={color}
                fillOpacity={0.1}
                strokeWidth={2}
              />
            ) : (
              <Line
                type="monotone"
                dataKey={dataKey}
                stroke={color}
                strokeWidth={2}
                dot={{ fill: color, strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: color, strokeWidth: 2 }}
              />
            )}
          </ChartComponent>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart;

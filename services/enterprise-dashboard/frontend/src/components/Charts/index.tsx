import React from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface ChartProps {
  data: any[];
  width?: number;
  height?: number;
  type?: 'line' | 'area' | 'bar' | 'pie';
  dataKey?: string;
  xAxisKey?: string;
  colors?: string[];
}

export const LineChartComponent: React.FC<ChartProps> = ({
  data,
  width = 400,
  height = 300,
  dataKey = 'value',
  xAxisKey = 'name',
  colors = ['#8884d8']
}) => {
  return (
    <ResponsiveContainer width={width} height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey={dataKey} stroke={colors[0]} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export const AreaChartComponent: React.FC<ChartProps> = ({
  data,
  width = 400,
  height = 300,
  dataKey = 'value',
  xAxisKey = 'name',
  colors = ['#8884d8']
}) => {
  return (
    <ResponsiveContainer width={width} height={height}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey={dataKey} stroke={colors[0]} fill={colors[0]} />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export const BarChartComponent: React.FC<ChartProps> = ({
  data,
  width = 400,
  height = 300,
  dataKey = 'value',
  xAxisKey = 'name',
  colors = ['#8884d8']
}) => {
  return (
    <ResponsiveContainer width={width} height={height}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey={dataKey} fill={colors[0]} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export const PieChartComponent: React.FC<ChartProps> = ({
  data,
  width = 400,
  height = 300,
  dataKey = 'value',
  colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00']
}) => {
  return (
    <ResponsiveContainer width={width} height={height}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          outerRadius={80}
          fill="#8884d8"
          dataKey={dataKey}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
};

export const PerformanceChart: React.FC<ChartProps> = ({
  data,
  width = 400,
  height = 300,
  dataKey = 'value',
  xAxisKey = 'name',
  colors = ['#8884d8']
}) => {
  return (
    <ResponsiveContainer width={width} height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey={dataKey} stroke={colors[0]} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
};

// Export individual components
export { LineChartComponent as LineChart };
export { AreaChartComponent as AreaChart };
export { BarChartComponent as BarChart };
export { PieChartComponent as PieChart };

const Charts = {
  LineChart: LineChartComponent,
  AreaChart: AreaChartComponent,
  BarChart: BarChartComponent,
  PieChart: PieChartComponent,
  PerformanceChart
};

export default Charts;

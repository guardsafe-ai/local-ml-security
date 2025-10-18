import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Checkbox,
  IconButton,
  Tooltip,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Typography,
  CircularProgress,
} from '@mui/material';
import {
  Search,
  MoreVert,
  Download,
  FilterList,
  Refresh,
  ViewColumn,
  Sort,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

export interface Column<T = any> {
  id: string;
  label: string;
  minWidth?: number;
  align?: 'left' | 'right' | 'center';
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: T) => React.ReactNode;
  getValue?: (row: T) => any;
}

interface DataGridProps<T = any> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  error?: string;
  searchable?: boolean;
  selectable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  pagination?: boolean;
  pageSize?: number;
  pageSizeOptions?: number[];
  onRowClick?: (row: T) => void;
  onSelectionChange?: (selectedRows: T[]) => void;
  onSort?: (column: string, direction: 'asc' | 'desc') => void;
  onFilter?: (filters: Record<string, any>) => void;
  onRefresh?: () => void;
  onExport?: () => void;
  actions?: React.ReactNode;
  className?: string;
  height?: number | string;
  emptyMessage?: string;
}

const StyledTableContainer = styled(TableContainer)<{ height?: number | string }>(({ height }) => ({
  height: height || 'auto',
  maxHeight: height || 'none',
}));

const StyledTableHead = styled(TableHead)(({ theme }) => ({
  backgroundColor: theme.palette.background.elevated,
  position: 'sticky',
  top: 0,
  zIndex: 1,
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(even)': {
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
  },
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    cursor: 'pointer',
  },
}));

const SearchField = styled(TextField)(({ theme }) => ({
  minWidth: 300,
  '& .MuiOutlinedInput-root': {
    backgroundColor: theme.palette.background.elevated,
  },
}));

export function DataGrid<T = any>({
  data,
  columns,
  loading = false,
  error,
  searchable = true,
  selectable = false,
  sortable = true,
  filterable = false,
  pagination = true,
  pageSize = 10,
  pageSizeOptions = [5, 10, 25, 50],
  onRowClick,
  onSelectionChange,
  onSort,
  onFilter,
  onRefresh,
  onExport,
  actions,
  className,
  height,
  emptyMessage = 'No data available',
}: DataGridProps<T>) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(pageSize);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortColumn, setSortColumn] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [selectedRows, setSelectedRows] = useState<T[]>([]);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  // Filter and sort data
  const processedData = useMemo(() => {
    let filtered = data;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter((row) =>
        columns.some((column) => {
          const value = column.getValue ? column.getValue(row) : (row as any)[column.id];
          return String(value).toLowerCase().includes(searchTerm.toLowerCase());
        })
      );
    }

    // Apply sorting
    if (sortColumn) {
      filtered.sort((a, b) => {
        const aValue = columns.find(col => col.id === sortColumn)?.getValue?.(a) ?? (a as any)[sortColumn];
        const bValue = columns.find(col => col.id === sortColumn)?.getValue?.(b) ?? (b as any)[sortColumn];
        
        if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
    }

    return filtered;
  }, [data, searchTerm, sortColumn, sortDirection, columns]);

  // Paginate data
  const paginatedData = useMemo(() => {
    if (!pagination) return processedData;
    const start = page * rowsPerPage;
    return processedData.slice(start, start + rowsPerPage);
  }, [processedData, page, rowsPerPage, pagination]);

  const handleSort = (columnId: string) => {
    if (!sortable) return;
    
    const newDirection = sortColumn === columnId && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortColumn(columnId);
    setSortDirection(newDirection);
    onSort?.(columnId, newDirection);
  };

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      setSelectedRows(paginatedData);
      onSelectionChange?.(paginatedData);
    } else {
      setSelectedRows([]);
      onSelectionChange?.([]);
    }
  };

  const handleSelectRow = (row: T, event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const newSelection = [...selectedRows, row];
      setSelectedRows(newSelection);
      onSelectionChange?.(newSelection);
    } else {
      const newSelection = selectedRows.filter((selected) => selected !== row);
      setSelectedRows(newSelection);
      onSelectionChange?.(newSelection);
    }
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const isRowSelected = (row: T) => selectedRows.includes(row);
  const isAllSelected = paginatedData.length > 0 && selectedRows.length === paginatedData.length;
  const isIndeterminate = selectedRows.length > 0 && selectedRows.length < paginatedData.length;

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height || 400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height || 400}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Paper className={className} sx={{ width: '100%', overflow: 'hidden' }}>
      {/* Toolbar */}
      <Box p={2} display="flex" justifyContent="space-between" alignItems="center">
        <Box display="flex" alignItems="center" gap={2}>
          {searchable && (
            <SearchField
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              size="small"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
          )}
          {filterable && (
            <Tooltip title="Filter">
              <IconButton size="small">
                <FilterList />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        
        <Box display="flex" alignItems="center" gap={1}>
          {actions}
          {onRefresh && (
            <Tooltip title="Refresh">
              <IconButton size="small" onClick={onRefresh}>
                <Refresh />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="More options">
            <IconButton size="small" onClick={handleMenuClick}>
              <MoreVert />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Table */}
      <StyledTableContainer height={height}>
        <Table stickyHeader>
          <StyledTableHead>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={isIndeterminate}
                    checked={isAllSelected}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align || 'left'}
                  style={{ minWidth: column.minWidth }}
                >
                  {sortable && column.sortable !== false ? (
                    <TableSortLabel
                      active={sortColumn === column.id}
                      direction={sortColumn === column.id ? sortDirection : 'asc'}
                      onClick={() => handleSort(column.id)}
                      IconComponent={Sort}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </StyledTableHead>
          <TableBody>
            {paginatedData.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length + (selectable ? 1 : 0)} align="center">
                  <Typography color="text.secondary" py={4}>
                    {emptyMessage}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              paginatedData.map((row, index) => (
                <StyledTableRow
                  key={index}
                  hover
                  onClick={() => onRowClick?.(row)}
                  selected={isRowSelected(row)}
                >
                  {selectable && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={isRowSelected(row)}
                        onChange={(e) => handleSelectRow(row, e)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                  )}
                  {columns.map((column) => (
                    <TableCell key={column.id} align={column.align || 'left'}>
                      {column.render
                        ? column.render(column.getValue ? column.getValue(row) : (row as any)[column.id], row)
                        : column.getValue
                        ? column.getValue(row)
                        : (row as any)[column.id]}
                    </TableCell>
                  ))}
                </StyledTableRow>
              ))
            )}
          </TableBody>
        </Table>
      </StyledTableContainer>

      {/* Pagination */}
      {pagination && (
        <TablePagination
          rowsPerPageOptions={pageSizeOptions}
          component="div"
          count={processedData.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      )}

      {/* Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        {onExport && (
          <MenuItem onClick={() => { onExport(); handleMenuClose(); }}>
            <ListItemIcon>
              <Download fontSize="small" />
            </ListItemIcon>
            <ListItemText>Export</ListItemText>
          </MenuItem>
        )}
        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <ViewColumn fontSize="small" />
          </ListItemIcon>
          <ListItemText>Columns</ListItemText>
        </MenuItem>
      </Menu>
    </Paper>
  );
}

export default DataGrid;

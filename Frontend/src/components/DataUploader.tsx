import React, { useState, useCallback } from 'react';
import { Upload, FileText, BarChart3, AlertCircle, CheckCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface DataColumn {
  name: string;
  type: 'numeric' | 'categorical' | 'text' | 'datetime';
  nullCount: number;
  uniqueCount: number;
  sample: any[];
}

interface DataPreview {
  columns: DataColumn[];
  rowCount: number;
  sample: Record<string, any>[];
}

interface DataUploaderProps {
  onDataUploaded?: (data: DataPreview) => void;
  acceptedFormats?: string[];
  maxSizeMB?: number;
}

export default function DataUploader({ 
  onDataUploaded, 
  acceptedFormats = ['.csv', '.parquet', '.json'],
  maxSizeMB = 100 
}: DataUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [schemaMapping, setSchemaMapping] = useState<Record<string, string>>({});

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFile = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Simulate file processing with progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Simulate API call to process file
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock data preview generation
      const mockPreview: DataPreview = {
        rowCount: 1000,
        columns: [
          {
            name: 'feature_1',
            type: 'numeric',
            nullCount: 5,
            uniqueCount: 800,
            sample: [1.2, 3.4, 5.6, 7.8, 9.0]
          },
          {
            name: 'category',
            type: 'categorical',
            nullCount: 0,
            uniqueCount: 3,
            sample: ['A', 'B', 'C', 'A', 'B']
          },
          {
            name: 'timestamp',
            type: 'datetime',
            nullCount: 2,
            uniqueCount: 998,
            sample: ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
          }
        ],
        sample: [
          { feature_1: 1.2, category: 'A', timestamp: '2024-01-01' },
          { feature_1: 3.4, category: 'B', timestamp: '2024-01-02' },
          { feature_1: 5.6, category: 'C', timestamp: '2024-01-03' },
          { feature_1: 7.8, category: 'A', timestamp: '2024-01-04' },
          { feature_1: 9.0, category: 'B', timestamp: '2024-01-05' }
        ]
      };

      clearInterval(progressInterval);
      setUploadProgress(100);
      setDataPreview(mockPreview);
      
      // Initialize schema mapping with auto-detected types
      const initialMapping: Record<string, string> = {};
      mockPreview.columns.forEach(col => {
        initialMapping[col.name] = col.type;
      });
      setSchemaMapping(initialMapping);
      
      onDataUploaded?.(mockPreview);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process file');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedFormats.includes(fileExtension)) {
      setError(`Unsupported file format. Please use: ${acceptedFormats.join(', ')}`);
      return;
    }

    // Validate file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      setError(`File too large. Maximum size: ${maxSizeMB}MB`);
      return;
    }

    await processFile(file);
  }, [acceptedFormats, maxSizeMB]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    await processFile(files[0]);
  }, []);

  const updateSchemaMapping = (columnName: string, newType: string) => {
    setSchemaMapping(prev => ({
      ...prev,
      [columnName]: newType
    }));
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'numeric': return 'bg-blue-100 text-blue-800';
      case 'categorical': return 'bg-green-100 text-green-800';
      case 'text': return 'bg-purple-100 text-purple-800';
      case 'datetime': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (dataPreview) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              Data Successfully Loaded
            </CardTitle>
            <CardDescription>
              {dataPreview.rowCount.toLocaleString()} rows, {dataPreview.columns.length} columns
            </CardDescription>
          </CardHeader>
        </Card>

        <Tabs defaultValue="preview" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="preview">Data Preview</TabsTrigger>
            <TabsTrigger value="schema">Schema Mapping</TabsTrigger>
            <TabsTrigger value="eda">Quick EDA</TabsTrigger>
          </TabsList>

          <TabsContent value="preview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Sample Data</CardTitle>
                <CardDescription>First 5 rows of your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {dataPreview.columns.map(col => (
                          <TableHead key={col.name} className="min-w-[120px]">
                            <div className="flex flex-col gap-1">
                              <span>{col.name}</span>
                              <Badge className={`text-xs ${getTypeColor(col.type)}`}>
                                {col.type}
                              </Badge>
                            </div>
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataPreview.sample.map((row, idx) => (
                        <TableRow key={idx}>
                          {dataPreview.columns.map(col => (
                            <TableCell key={col.name}>
                              {row[col.name]?.toString() || 'null'}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="schema" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Column Type Mapping</CardTitle>
                <CardDescription>
                  Review and adjust the detected column types before training
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dataPreview.columns.map(col => (
                    <div key={col.name} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex-1">
                        <h4 className="font-medium">{col.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {col.uniqueCount} unique values, {col.nullCount} nulls
                        </p>
                        <div className="flex gap-2 mt-2">
                          {col.sample.slice(0, 3).map((val, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {val?.toString() || 'null'}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <Select
                        value={schemaMapping[col.name]}
                        onValueChange={(value) => updateSchemaMapping(col.name, value)}
                      >
                        <SelectTrigger className="w-40">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="numeric">Numeric</SelectItem>
                          <SelectItem value="categorical">Categorical</SelectItem>
                          <SelectItem value="text">Text</SelectItem>
                          <SelectItem value="datetime">DateTime</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="eda" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Dataset Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Total Rows:</span>
                      <span className="font-medium">{dataPreview.rowCount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Columns:</span>
                      <span className="font-medium">{dataPreview.columns.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Missing Values:</span>
                      <span className="font-medium">
                        {dataPreview.columns.reduce((sum, col) => sum + col.nullCount, 0)}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Column Types</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {['numeric', 'categorical', 'text', 'datetime'].map(type => {
                      const count = dataPreview.columns.filter(col => col.type === type).length;
                      return count > 0 ? (
                        <div key={type} className="flex justify-between text-sm">
                          <Badge className={getTypeColor(type)}>{type}</Badge>
                          <span className="font-medium">{count}</span>
                        </div>
                      ) : null;
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Data Quality</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Completeness:</span>
                      <span className="font-medium">
                        {(((dataPreview.rowCount * dataPreview.columns.length) - 
                           dataPreview.columns.reduce((sum, col) => sum + col.nullCount, 0)) / 
                          (dataPreview.rowCount * dataPreview.columns.length) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Avg Uniqueness:</span>
                      <span className="font-medium">
                        {(dataPreview.columns.reduce((sum, col) => sum + col.uniqueCount, 0) / 
                          dataPreview.columns.length / dataPreview.rowCount * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex gap-2">
          <Button onClick={() => setDataPreview(null)} variant="outline">
            Upload Different File
          </Button>
          <Button>
            Continue to Training
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Data Upload
        </CardTitle>
        <CardDescription>
          Upload your dataset to begin training. Supported formats: {acceptedFormats.join(', ')}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragging 
                ? 'border-primary bg-primary/5' 
                : 'border-muted-foreground/25 hover:border-primary/50'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 rounded-full bg-muted">
                <FileText className="h-8 w-8 text-muted-foreground" />
              </div>
              
              <div>
                <h3 className="text-lg font-semibold">Drop your file here</h3>
                <p className="text-muted-foreground">
                  or click to browse files
                </p>
              </div>

              <input
                type="file"
                accept={acceptedFormats.join(',')}
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
                disabled={isProcessing}
              />
              
              <Button 
                asChild 
                variant="outline"
                disabled={isProcessing}
              >
                <label htmlFor="file-upload" className="cursor-pointer">
                  Select File
                </label>
              </Button>

              <p className="text-xs text-muted-foreground">
                Maximum file size: {maxSizeMB}MB
              </p>
            </div>
          </div>

          {isProcessing && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Processing file...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
/**
 * Enhanced DataUpload page with dataset analysis integration.
 * Handles file upload, displays DataProfile, and shows ActionPlan buttons.
 */

import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Upload, 
  FileText, 
  BarChart3, 
  AlertCircle, 
  CheckCircle, 
  Clock,
  Download,
  Play,
  Settings,
  Database,
  Brain,
  TrendingUp,
  Zap
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import TechnicalTerm from '@/components/TechnicalTerm';
import { formatFileSize } from '@/lib/i18n-utils';
import ChatWidget from '@/components/ChatWidget';

// Types
interface ColumnProfile {
  name: string;
  dtype: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  cardinality_ratio: number;
  suggested_role: string;
  potential_pii: boolean;
  statistics: Record<string, any>;
  sample_values: any[];
}

interface DataProfile {
  dataset_id: string;
  filename: string;
  shape: [number, number];
  size_mb: number;
  columns: ColumnProfile[];
  suggested_task_type: string;
  candidate_targets: string[];
  data_quality: string;
  issues: string[];
  metadata: Record<string, any>;
  created_at: number;
}

interface ActionPlan {
  quick_train_recommended: boolean;
  chunk_embed_recommended: boolean;
  preprocessing_steps: string[];
  estimated_training_time: string;
  memory_requirements: string;
  suggested_models: string[];
  priority_actions: Array<{
    label: string;
    action: string;
    endpoint: string;
    description: string;
  }>;
}

const DataUpload: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dataProfile, setDataProfile] = useState<DataProfile | null>(null);
  const [actionPlan, setActionPlan] = useState<ActionPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('upload');

  const isRTL = i18n.language === 'ar';

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Validate file type
      const allowedTypes = ['.csv', '.parquet', '.json'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      
      if (!allowedTypes.includes(fileExtension)) {
        setError(t('upload.error.unsupportedFormat', { format: fileExtension }));
        return;
      }
      
      // Validate file size (max 100MB)
      const maxSize = 100 * 1024 * 1024;
      if (file.size > maxSize) {
        setError(t('upload.error.fileTooLarge', { 
          size: formatFileSize(file.size, i18n.language),
          maxSize: formatFileSize(maxSize, i18n.language)
        }));
        return;
      }
      
      setUploadedFile(file);
      setError(null);
      setActiveTab('upload');
      
      // Start upload and analysis
      uploadAndAnalyze(file);
    }
  }, [t, i18n.language]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/octet-stream': ['.parquet']
    },
    multiple: false,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  const uploadAndAnalyze = async (file: File) => {
    setIsUploading(true);
    setIsAnalyzing(false);
    setUploadProgress(0);
    setError(null);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Upload file
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const uploadResult = await uploadResponse.json();
      setIsUploading(false);
      setIsAnalyzing(true);

      // Analyze dataset
      const analysisResponse = await fetch(`/api/data/${uploadResult.dataset_id}/profile`);
      
      if (!analysisResponse.ok) {
        throw new Error(`Analysis failed: ${analysisResponse.statusText}`);
      }

      const analysisResult = await analysisResponse.json();
      setDataProfile(analysisResult.profile);
      setActionPlan(analysisResult.action_plan);
      setActiveTab('profile');

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  };

  const handleActionClick = async (action: { endpoint: string; action: string }) => {
    try {
      if (action.action === 'quick_train') {
        // Navigate to training page or trigger training
        window.location.href = `/train?dataset_id=${dataProfile?.dataset_id}`;
      } else if (action.action === 'create_embeddings') {
        // Start embedding creation
        const response = await fetch(action.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: dataProfile?.dataset_id })
        });
        
        if (response.ok) {
          setActiveTab('chat');
        }
      } else {
        // Default: navigate to endpoint
        window.location.href = action.endpoint;
      }
    } catch (err) {
      setError(`Action failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const getQualityBadgeVariant = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'excellent': return 'default';
      case 'good': return 'secondary';
      case 'fair': return 'outline';
      case 'poor': return 'destructive';
      default: return 'outline';
    }
  };

  const renderUploadZone = () => (
    <div className="max-w-4xl mx-auto">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            {t('upload.title')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-muted-foreground/50'}
            `}
          >
            <input {...getInputProps()} />
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            
            {isDragActive ? (
              <p className="text-lg">{t('upload.dropZone.dropping')}</p>
            ) : (
              <div>
                <p className="text-lg mb-2">{t('upload.dropZone.title')}</p>
                <p className="text-muted-foreground mb-4">{t('upload.dropZone.subtitle')}</p>
                <Button variant="outline">
                  {t('upload.dropZone.selectFile')}
                </Button>
              </div>
            )}
            
            <div className="mt-6 text-sm text-muted-foreground">
              <p>{t('upload.supportedFormats')}: <TechnicalTerm>CSV, JSON, Parquet</TechnicalTerm></p>
              <p>{t('upload.maxSize')}: <TechnicalTerm>100 MB</TechnicalTerm></p>
            </div>
          </div>

          {uploadedFile && (
            <div className="mt-6 p-4 bg-muted rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className="w-8 h-8 text-primary" />
                <div className="flex-1">
                  <p className="font-medium">{uploadedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(uploadedFile.size, i18n.language)}
                  </p>
                </div>
              </div>

              {(isUploading || isAnalyzing) && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm">
                      {isUploading ? t('upload.uploading') : t('upload.analyzing')}
                    </span>
                    <span className="text-sm">{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} className="w-full" />
                </div>
              )}
            </div>
          )}

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );

  const renderDataProfile = () => {
    if (!dataProfile) return null;

    return (
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Overview Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              {t('upload.profile.title')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold">{dataProfile.shape[0].toLocaleString()}</div>
                <div className="text-sm text-muted-foreground">{t('upload.profile.rows')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">{dataProfile.shape[1]}</div>
                <div className="text-sm text-muted-foreground">{t('upload.profile.columns')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {formatFileSize(dataProfile.size_mb * 1024 * 1024, i18n.language)}
                </div>
                <div className="text-sm text-muted-foreground">{t('upload.profile.size')}</div>
              </div>
              <div className="text-center">
                <Badge variant={getQualityBadgeVariant(dataProfile.data_quality)}>
                  {t(`upload.profile.quality.${dataProfile.data_quality}`)}
                </Badge>
                <div className="text-sm text-muted-foreground mt-1">{t('upload.profile.quality.label')}</div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">{t('upload.profile.taskType')}</h4>
                <Badge variant="outline" className="text-sm">
                  {t(`upload.profile.tasks.${dataProfile.suggested_task_type}`)}
                </Badge>
                
                {dataProfile.candidate_targets.length > 0 && (
                  <div className="mt-4">
                    <h5 className="text-sm font-medium mb-2">{t('upload.profile.candidateTargets')}</h5>
                    <div className="flex flex-wrap gap-1">
                      {dataProfile.candidate_targets.map(target => (
                        <Badge key={target} variant="secondary" className="text-xs">
                          <TechnicalTerm>{target}</TechnicalTerm>
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {dataProfile.issues.length > 0 && (
                <div>
                  <h4 className="font-medium mb-3 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-yellow-500" />
                    {t('upload.profile.issues')}
                  </h4>
                  <ul className="text-sm space-y-1">
                    {dataProfile.issues.map((issue, index) => (
                      <li key={index} className="text-muted-foreground">• {issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Action Plan Card */}
        {actionPlan && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                {t('upload.actionPlan.title')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6 mb-6">
                <div>
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    {t('upload.actionPlan.estimatedTime')}
                  </h4>
                  <p className="text-sm text-muted-foreground">{actionPlan.estimated_training_time}</p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    {t('upload.actionPlan.memoryReq')}
                  </h4>
                  <p className="text-sm text-muted-foreground">{actionPlan.memory_requirements}</p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Brain className="w-4 h-4" />
                    {t('upload.actionPlan.suggestedModels')}
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {actionPlan.suggested_models.map(model => (
                      <Badge key={model} variant="outline" className="text-xs">
                        <TechnicalTerm>{model}</TechnicalTerm>
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <Separator className="my-4" />

              <h4 className="font-medium mb-4">{t('upload.actionPlan.recommendedActions')}</h4>
              <div className="grid gap-3">
                {actionPlan.priority_actions.map((action, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <h5 className="font-medium">{action.label}</h5>
                      <p className="text-sm text-muted-foreground">{action.description}</p>
                    </div>
                    <Button 
                      onClick={() => handleActionClick(action)}
                      variant="outline"
                      size="sm"
                      className="ml-4"
                    >
                      {action.action === 'quick_train' && <Play className="w-4 h-4 mr-1 icon-mirror" />}
                      {action.action === 'create_embeddings' && <TrendingUp className="w-4 h-4 mr-1 icon-mirror" />}
                      {action.action === 'preprocess' && <Settings className="w-4 h-4 mr-1 icon-mirror" />}
                      {t('upload.actionPlan.start')}
                    </Button>
                  </div>
                ))}
              </div>

              {actionPlan.preprocessing_steps.length > 0 && (
                <div className="mt-6">
                  <h5 className="font-medium mb-2">{t('upload.actionPlan.preprocessing')}</h5>
                  <ul className="text-sm space-y-1">
                    {actionPlan.preprocessing_steps.map((step, index) => (
                      <li key={index} className="text-muted-foreground">• {step}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Columns Detail */}
        <Card>
          <CardHeader>
            <CardTitle>{t('upload.profile.columnsDetail')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">{t('upload.profile.column.name')}</th>
                    <th className="text-left p-2">{t('upload.profile.column.type')}</th>
                    <th className="text-left p-2">{t('upload.profile.column.missing')}</th>
                    <th className="text-left p-2">{t('upload.profile.column.unique')}</th>
                    <th className="text-left p-2">{t('upload.profile.column.role')}</th>
                  </tr>
                </thead>
                <tbody>
                  {dataProfile.columns.map((column, index) => (
                    <tr key={index} className="border-b">
                      <td className="p-2">
                        <div className="flex items-center gap-2">
                          <TechnicalTerm>{column.name}</TechnicalTerm>
                          {column.potential_pii && (
                            <Badge variant="destructive" className="text-xs">PII</Badge>
                          )}
                        </div>
                      </td>
                      <td className="p-2">
                        <TechnicalTerm className="text-xs">{column.dtype}</TechnicalTerm>
                      </td>
                      <td className="p-2">{column.null_percentage.toFixed(1)}%</td>
                      <td className="p-2">{column.unique_count.toLocaleString()}</td>
                      <td className="p-2">
                        <Badge variant="outline" className="text-xs">
                          {t(`upload.profile.roles.${column.suggested_role}`)}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">{t('upload.pageTitle')}</h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            {t('upload.pageSubtitle')}
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3 max-w-md mx-auto mb-8">
            <TabsTrigger value="upload">{t('upload.tabs.upload')}</TabsTrigger>
            <TabsTrigger value="profile" disabled={!dataProfile}>
              {t('upload.tabs.profile')}
            </TabsTrigger>
            <TabsTrigger value="chat" disabled={!dataProfile}>
              {t('upload.tabs.chat')}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            {renderUploadZone()}
          </TabsContent>

          <TabsContent value="profile">
            {renderDataProfile()}
          </TabsContent>

          <TabsContent value="chat">
            {dataProfile && (
              <div className="max-w-4xl mx-auto">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      {t('upload.chat.title')}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      {t('upload.chat.description')}
                    </p>
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertDescription>
                        {t('upload.chat.datasetReady', { filename: dataProfile.filename })}
                      </AlertDescription>
                    </Alert>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Chat Widget */}
        <ChatWidget 
          datasetId={dataProfile?.dataset_id}
          onActionTrigger={(action) => {
            // Handle action triggers from chat
            if (action.action_endpoint.includes('train')) {
              setActiveTab('profile');
            }
          }}
        />
      </div>
    </div>
  );
};

export default DataUpload;

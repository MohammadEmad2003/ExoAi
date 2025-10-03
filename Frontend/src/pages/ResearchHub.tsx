import React, { useState } from 'react';
import { 
  Brain, 
  Database, 
  Settings, 
  BarChart3, 
  Code, 
  Terminal,
  Layers,
  Zap,
  Shield,
  Download,
  Play,
  Pause,
  Square,
  RefreshCw,
  FileText,
  GitBranch,
  Clock
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import DataUploader from '@/components/DataUploader';
import HyperparamPanel from '@/components/HyperparamPanel';
import ModelCard from '@/components/ModelCard';

interface TrainingJob {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  progress: number;
  startTime: string;
  duration: string;
  accuracy?: number;
  loss?: number;
}

interface ModelVersion {
  id: string;
  name: string;
  version: string;
  accuracy: number;
  created: string;
  size: string;
  framework: string;
}

const TRAINING_JOBS: TrainingJob[] = [
  {
    id: '1',
    name: 'TabKANet-Exoplanet-v1',
    status: 'running',
    progress: 67,
    startTime: '14:23:15',
    duration: '8m 42s',
    accuracy: 0.891,
    loss: 0.324
  },
  {
    id: '2',
    name: 'FT-Transformer-Stellar',
    status: 'completed',
    progress: 100,
    startTime: '13:45:22',
    duration: '12m 18s',
    accuracy: 0.876,
    loss: 0.287
  },
  {
    id: '3',
    name: 'TabKANet-Galaxy-v2',
    status: 'queued',
    progress: 0,
    startTime: '-',
    duration: '-'
  }
];

const MODEL_VERSIONS: ModelVersion[] = [
  {
    id: '1',
    name: 'TabKANet-Exoplanet',
    version: '2.1.0',
    accuracy: 0.924,
    created: '2024-01-15',
    size: '12.4 MB',
    framework: 'PyTorch'
  },
  {
    id: '2',
    name: 'FT-Transformer-Stellar',
    version: '1.3.2',
    accuracy: 0.876,
    created: '2024-01-14',
    size: '8.7 MB',
    framework: 'PyTorch'
  },
  {
    id: '3',
    name: 'XGBoost-Galaxy',
    version: '1.0.1',
    accuracy: 0.845,
    created: '2024-01-13',
    size: '2.1 MB',
    framework: 'XGBoost'
  }
];

// Mock data for the current model
const MOCK_MODEL_INFO = {
  name: 'TabKANet-Research',
  version: '2.1.0',
  architecture: 'TabKANet (KAN + Transformer)',
  created_at: '2024-01-15 14:30:00',
  training_time: '12m 34s',
  dataset_size: 2847,
  parameters: 125000,
  framework: 'PyTorch'
};

const MOCK_METRICS = {
  accuracy: 0.924,
  precision: 0.918,
  recall: 0.912,
  f1_score: 0.915,
  auc_roc: 0.967,
  loss: 0.234,
  val_loss: 0.267
};

const MOCK_ROBUSTNESS = {
  adversarial_accuracy: 0.856,
  calibration_error: 0.123,
  uncertainty_score: 0.089,
  attack_success_rate: 0.144
};

const MOCK_PERFORMANCE = {
  inference_time_ms: 2.3,
  throughput_samples_sec: 4347,
  memory_usage_mb: 156,
  model_size_mb: 12.4
};

export default function ResearchHub() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-100 text-blue-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'failed': return 'bg-red-100 text-red-800';
      case 'queued': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="h-3 w-3" />;
      case 'completed': return <Square className="h-3 w-3" />;
      case 'failed': return <Square className="h-3 w-3" />;
      case 'queued': return <Clock className="h-3 w-3" />;
      default: return <Square className="h-3 w-3" />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-2">
                <Brain className="h-8 w-8 text-primary" />
                Research Hub
              </h1>
              <p className="text-muted-foreground">
                Advanced machine learning workspace for researchers and power users
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline">
                <Terminal className="h-4 w-4 mr-2" />
                Jupyter Lab
              </Button>
              <Button>
                <Code className="h-4 w-4 mr-2" />
                New Experiment
              </Button>
            </div>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="data">Data Pipeline</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="models">Model Registry</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="deployment">Deployment</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Active Jobs</p>
                      <p className="text-2xl font-bold">2</p>
                    </div>
                    <Play className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Models Trained</p>
                      <p className="text-2xl font-bold">12</p>
                    </div>
                    <Layers className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Best Accuracy</p>
                      <p className="text-2xl font-bold">92.4%</p>
                    </div>
                    <BarChart3 className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">GPU Hours</p>
                      <p className="text-2xl font-bold">47.2</p>
                    </div>
                    <Zap className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Active Training Jobs</CardTitle>
                  <CardDescription>Currently running and queued experiments</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {TRAINING_JOBS.filter(job => job.status !== 'completed').map(job => (
                      <div key={job.id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">{job.name}</span>
                            <Badge className={getStatusColor(job.status)}>
                              {getStatusIcon(job.status)}
                              {job.status}
                            </Badge>
                          </div>
                          {job.status === 'running' && (
                            <>
                              <Progress value={job.progress} className="h-1 mb-1" />
                              <div className="text-xs text-muted-foreground">
                                {job.progress}% • Acc: {job.accuracy?.toFixed(3)} • Loss: {job.loss?.toFixed(3)}
                              </div>
                            </>
                          )}
                        </div>
                        <div className="flex gap-1">
                          {job.status === 'running' && (
                            <Button size="sm" variant="outline">
                              <Pause className="h-3 w-3" />
                            </Button>
                          )}
                          <Button size="sm" variant="outline">
                            <Square className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Models</CardTitle>
                  <CardDescription>Latest trained models in your registry</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {MODEL_VERSIONS.slice(0, 3).map(model => (
                      <div key={model.id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <div className="font-medium">{model.name}</div>
                          <div className="text-sm text-muted-foreground">
                            v{model.version} • {model.accuracy.toFixed(1)}% accuracy • {model.size}
                          </div>
                        </div>
                        <div className="flex gap-1">
                          <Button size="sm" variant="outline">
                            <Download className="h-3 w-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <GitBranch className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* System Status */}
            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
                <CardDescription>Current resource utilization</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>CPU Usage</span>
                      <span>34%</span>
                    </div>
                    <Progress value={34} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Memory</span>
                      <span>67%</span>
                    </div>
                    <Progress value={67} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>GPU Memory</span>
                      <span>89%</span>
                    </div>
                    <Progress value={89} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <DataUploader 
                  acceptedFormats={['.csv', '.parquet', '.json', '.h5', '.feather']}
                  maxSizeMB={1000}
                />
              </div>
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Data Pipeline</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button variant="outline" className="w-full justify-start">
                      <Database className="h-4 w-4 mr-2" />
                      Connect Database
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Auto-refresh Data
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="h-4 w-4 mr-2" />
                      Custom Preprocessing
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Data Versioning</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Current Version:</span>
                        <Badge variant="outline">v1.2.3</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Last Updated:</span>
                        <span>2 hours ago</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>2.4 GB</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="training" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HyperparamPanel mode="advanced" />
              </div>
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Training Queue</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {TRAINING_JOBS.map(job => (
                        <div key={job.id} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(job.status)}
                            <span className="truncate">{job.name}</span>
                          </div>
                          <Badge className={getStatusColor(job.status)} variant="outline">
                            {job.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Experiment Tracking</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button variant="outline" className="w-full justify-start">
                      <FileText className="h-4 w-4 mr-2" />
                      MLflow UI
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <BarChart3 className="h-4 w-4 mr-2" />
                      TensorBoard
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="h-4 w-4 mr-2" />
                      Weights & Biases
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="models" className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">Model Registry</h2>
              <div className="flex gap-2">
                <Button variant="outline">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
                <Button>
                  <Code className="h-4 w-4 mr-2" />
                  Register Model
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {MODEL_VERSIONS.map(model => (
                <Card key={model.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardHeader>
                    <CardTitle className="text-base">{model.name}</CardTitle>
                    <CardDescription>
                      Version {model.version} • {model.framework}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Accuracy:</span>
                        <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>{model.size}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Created:</span>
                        <span>{model.created}</span>
                      </div>
                    </div>
                    <Separator className="my-3" />
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" className="flex-1">
                        <Download className="h-3 w-3 mr-1" />
                        Export
                      </Button>
                      <Button size="sm" variant="outline" className="flex-1">
                        <GitBranch className="h-3 w-3 mr-1" />
                        Compare
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="evaluation" className="space-y-6">
            <ModelCard
              modelInfo={MOCK_MODEL_INFO}
              metrics={MOCK_METRICS}
              robustness={MOCK_ROBUSTNESS}
              performance={MOCK_PERFORMANCE}
              onExport={(format) => {
                console.log('Exporting model in format:', format);
              }}
              onShare={() => {
                console.log('Sharing model');
              }}
            />
          </TabsContent>

          <TabsContent value="deployment" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Model Deployment</CardTitle>
                  <CardDescription>Deploy your models to production environments</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <Button className="w-full justify-start">
                      <Zap className="h-4 w-4 mr-2" />
                      Deploy to Cloud
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Terminal className="h-4 w-4 mr-2" />
                      Generate Docker Image
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="h-4 w-4 mr-2" />
                      Create API Endpoint
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Shield className="h-4 w-4 mr-2" />
                      Edge Deployment
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Active Deployments</CardTitle>
                  <CardDescription>Currently running model endpoints</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Alert>
                      <Shield className="h-4 w-4" />
                      <AlertDescription>
                        No active deployments. Deploy your first model to get started.
                      </AlertDescription>
                    </Alert>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Monitoring & Analytics</CardTitle>
                <CardDescription>Track model performance in production</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold">0</div>
                    <div className="text-sm text-muted-foreground">Requests/min</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold">-</div>
                    <div className="text-sm text-muted-foreground">Avg Latency</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold">-</div>
                    <div className="text-sm text-muted-foreground">Error Rate</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
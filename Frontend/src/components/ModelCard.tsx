import React, { useState } from 'react';
import { 
  Download, 
  Share2, 
  BarChart3, 
  Shield, 
  Clock, 
  Cpu, 
  Zap,
  CheckCircle,
  AlertTriangle,
  Info,
  ExternalLink
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  loss: number;
  val_loss: number;
}

interface RobustnessMetrics {
  adversarial_accuracy: number;
  calibration_error: number;
  uncertainty_score: number;
  attack_success_rate: number;
}

interface PerformanceMetrics {
  inference_time_ms: number;
  throughput_samples_sec: number;
  memory_usage_mb: number;
  model_size_mb: number;
}

interface ModelInfo {
  name: string;
  version: string;
  architecture: string;
  created_at: string;
  training_time: string;
  dataset_size: number;
  parameters: number;
  framework: string;
}

interface ExportFormat {
  name: string;
  extension: string;
  description: string;
  size_mb: number;
  supported: boolean;
}

interface ModelCardProps {
  modelInfo: ModelInfo;
  metrics: ModelMetrics;
  robustness: RobustnessMetrics;
  performance: PerformanceMetrics;
  onExport?: (format: string) => void;
  onShare?: () => void;
}

const EXPORT_FORMATS: ExportFormat[] = [
  {
    name: 'TorchScript',
    extension: '.pt',
    description: 'Optimized for PyTorch deployment',
    size_mb: 45.2,
    supported: true
  },
  {
    name: 'ONNX',
    extension: '.onnx',
    description: 'Cross-platform inference',
    size_mb: 42.8,
    supported: true
  },
  {
    name: 'TensorRT',
    extension: '.plan',
    description: 'NVIDIA GPU optimized',
    size_mb: 38.5,
    supported: false // GPU not available
  },
  {
    name: 'Core ML',
    extension: '.mlmodel',
    description: 'Apple device deployment',
    size_mb: 44.1,
    supported: false
  }
];

export default function ModelCard({
  modelInfo,
  metrics,
  robustness,
  performance,
  onExport,
  onShare
}: ModelCardProps) {
  const [selectedFormat, setSelectedFormat] = useState('torchscript');
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async (format: string) => {
    setIsExporting(true);
    try {
      await onExport?.(format);
      // Simulate export delay
      await new Promise(resolve => setTimeout(resolve, 2000));
    } finally {
      setIsExporting(false);
    }
  };

  const getScoreColor = (score: number, reverse = false) => {
    if (reverse) score = 1 - score;
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadge = (score: number, reverse = false) => {
    if (reverse) score = 1 - score;
    if (score >= 0.8) return 'bg-green-100 text-green-800';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="space-y-6">
      {/* Model Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                {modelInfo.name}
                <Badge variant="outline">v{modelInfo.version}</Badge>
              </CardTitle>
              <CardDescription>
                {modelInfo.architecture} • {modelInfo.parameters.toLocaleString()} parameters
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={onShare}>
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
              <Dialog>
                <DialogTrigger asChild>
                  <Button size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Export Model</DialogTitle>
                    <DialogDescription>
                      Choose the format for model deployment
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Export Format</label>
                      <Select value={selectedFormat} onValueChange={setSelectedFormat}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {EXPORT_FORMATS.map(format => (
                            <SelectItem 
                              key={format.name.toLowerCase()} 
                              value={format.name.toLowerCase()}
                              disabled={!format.supported}
                            >
                              <div className="flex items-center justify-between w-full">
                                <span>{format.name}</span>
                                {!format.supported && (
                                  <Badge variant="secondary" className="ml-2">
                                    Unavailable
                                  </Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {EXPORT_FORMATS.find(f => f.name.toLowerCase() === selectedFormat) && (
                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                          {EXPORT_FORMATS.find(f => f.name.toLowerCase() === selectedFormat)?.description}
                          <br />
                          Estimated size: {EXPORT_FORMATS.find(f => f.name.toLowerCase() === selectedFormat)?.size_mb}MB
                        </AlertDescription>
                      </Alert>
                    )}

                    <Button 
                      onClick={() => handleExport(selectedFormat)} 
                      disabled={isExporting || !EXPORT_FORMATS.find(f => f.name.toLowerCase() === selectedFormat)?.supported}
                      className="w-full"
                    >
                      {isExporting ? 'Exporting...' : 'Export Model'}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs defaultValue="metrics" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="metrics">Performance</TabsTrigger>
          <TabsTrigger value="robustness">Robustness</TabsTrigger>
          <TabsTrigger value="inference">Inference</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Accuracy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </div>
                <Progress value={metrics.accuracy * 100} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Precision</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold mb-2 ${getScoreColor(metrics.precision)}`}>
                  {(metrics.precision * 100).toFixed(1)}%
                </div>
                <Progress value={metrics.precision * 100} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Recall</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold mb-2 ${getScoreColor(metrics.recall)}`}>
                  {(metrics.recall * 100).toFixed(1)}%
                </div>
                <Progress value={metrics.recall * 100} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">F1 Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold mb-2 ${getScoreColor(metrics.f1_score)}`}>
                  {metrics.f1_score.toFixed(3)}
                </div>
                <Progress value={metrics.f1_score * 100} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">AUC-ROC</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold mb-2 ${getScoreColor(metrics.auc_roc)}`}>
                  {metrics.auc_roc.toFixed(3)}
                </div>
                <Progress value={metrics.auc_roc * 100} className="h-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-2">
                  {metrics.loss.toFixed(4)}
                </div>
                <div className="text-sm text-muted-foreground">
                  Val: {metrics.val_loss.toFixed(4)}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="robustness" className="space-y-4">
          <Alert>
            <Shield className="h-4 w-4" />
            <AlertDescription>
              Robustness metrics evaluate model reliability under adversarial conditions and uncertainty.
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center justify-between">
                  Adversarial Accuracy
                  <Badge className={getScoreBadge(robustness.adversarial_accuracy)}>
                    {robustness.adversarial_accuracy >= 0.8 ? 'Robust' : 
                     robustness.adversarial_accuracy >= 0.6 ? 'Moderate' : 'Vulnerable'}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-xl font-bold mb-2 ${getScoreColor(robustness.adversarial_accuracy)}`}>
                  {(robustness.adversarial_accuracy * 100).toFixed(1)}%
                </div>
                <Progress value={robustness.adversarial_accuracy * 100} className="h-2 mb-2" />
                <p className="text-sm text-muted-foreground">
                  Accuracy under FGSM and PGD attacks
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center justify-between">
                  Calibration Error
                  <Badge className={getScoreBadge(robustness.calibration_error, true)}>
                    {robustness.calibration_error <= 0.2 ? 'Well-calibrated' : 
                     robustness.calibration_error <= 0.4 ? 'Moderate' : 'Poor'}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-xl font-bold mb-2 ${getScoreColor(robustness.calibration_error, true)}`}>
                  {(robustness.calibration_error * 100).toFixed(1)}%
                </div>
                <Progress value={(1 - robustness.calibration_error) * 100} className="h-2 mb-2" />
                <p className="text-sm text-muted-foreground">
                  Expected Calibration Error (ECE)
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Uncertainty Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xl font-bold mb-2">
                  {robustness.uncertainty_score.toFixed(3)}
                </div>
                <Progress value={robustness.uncertainty_score * 100} className="h-2 mb-2" />
                <p className="text-sm text-muted-foreground">
                  Average prediction uncertainty
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Attack Success Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-xl font-bold mb-2 ${getScoreColor(robustness.attack_success_rate, true)}`}>
                  {(robustness.attack_success_rate * 100).toFixed(1)}%
                </div>
                <Progress value={robustness.attack_success_rate * 100} className="h-2 mb-2" />
                <p className="text-sm text-muted-foreground">
                  Percentage of successful adversarial attacks
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="inference" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Latency
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-1">
                  {performance.inference_time_ms.toFixed(1)}ms
                </div>
                <p className="text-sm text-muted-foreground">
                  Per sample (p50)
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Throughput
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-1">
                  {performance.throughput_samples_sec.toLocaleString()}
                </div>
                <p className="text-sm text-muted-foreground">
                  Samples/second
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Memory
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-1">
                  {performance.memory_usage_mb.toFixed(0)}MB
                </div>
                <p className="text-sm text-muted-foreground">
                  Peak usage
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Model Size</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mb-1">
                  {performance.model_size_mb.toFixed(1)}MB
                </div>
                <p className="text-sm text-muted-foreground">
                  Checkpoint size
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Export Formats</CardTitle>
              <CardDescription>
                Available deployment formats and their characteristics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {EXPORT_FORMATS.map(format => (
                  <div key={format.name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${format.supported ? 'bg-green-500' : 'bg-gray-300'}`} />
                      <div>
                        <div className="font-medium">{format.name}</div>
                        <div className="text-sm text-muted-foreground">{format.description}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{format.size_mb}MB</div>
                      <Badge variant={format.supported ? "default" : "secondary"} className="text-xs">
                        {format.supported ? "Available" : "Unavailable"}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="details" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Model Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Architecture</label>
                    <div className="font-medium">{modelInfo.architecture}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Framework</label>
                    <div className="font-medium">{modelInfo.framework}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Parameters</label>
                    <div className="font-medium">{modelInfo.parameters.toLocaleString()}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Dataset Size</label>
                    <div className="font-medium">{modelInfo.dataset_size.toLocaleString()} samples</div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Created</label>
                    <div className="font-medium">{modelInfo.created_at}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Training Time</label>
                    <div className="font-medium">{modelInfo.training_time}</div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Version</label>
                    <div className="font-medium">v{modelInfo.version}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Training Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm font-mono bg-muted p-4 rounded-lg">
                {`model_type: "tabkanet"
learning_rate: 0.001
batch_size: 64
epochs: 100
d_model: 64
K_inner: 16
trans_heads: 4
trans_depth: 3
optimizer: "adamw"
scheduler: "cosine"`}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
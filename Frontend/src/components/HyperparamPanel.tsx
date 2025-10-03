import React, { useState, useEffect } from 'react';
import { Settings, Code, RotateCcw, Save, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';

interface HyperparamConfig {
  model_type: string;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  dropout: number;
  d_model: number;
  K_inner: number;
  trans_heads: number;
  trans_depth: number;
  mlp_hidden: number;
  optimizer: string;
  scheduler: string;
  weight_decay: number;
  early_stopping: boolean;
  patience: number;
  validation_split: number;
}

interface Preset {
  name: string;
  description: string;
  config: Partial<HyperparamConfig>;
}

const DEFAULT_CONFIG: HyperparamConfig = {
  model_type: 'tabkanet',
  learning_rate: 0.001,
  batch_size: 64,
  epochs: 100,
  dropout: 0.1,
  d_model: 64,
  K_inner: 16,
  trans_heads: 4,
  trans_depth: 3,
  mlp_hidden: 128,
  optimizer: 'adamw',
  scheduler: 'cosine',
  weight_decay: 0.01,
  early_stopping: true,
  patience: 10,
  validation_split: 0.2
};

const PRESETS: Preset[] = [
  {
    name: 'Fast Training',
    description: 'Quick training for prototyping',
    config: {
      learning_rate: 0.01,
      batch_size: 128,
      epochs: 20,
      d_model: 32,
      trans_depth: 2,
      patience: 5
    }
  },
  {
    name: 'Balanced',
    description: 'Good balance of speed and performance',
    config: DEFAULT_CONFIG
  },
  {
    name: 'High Performance',
    description: 'Optimized for best accuracy',
    config: {
      learning_rate: 0.0005,
      batch_size: 32,
      epochs: 200,
      dropout: 0.05,
      d_model: 128,
      K_inner: 32,
      trans_heads: 8,
      trans_depth: 6,
      mlp_hidden: 256,
      patience: 20
    }
  },
  {
    name: 'Large Dataset',
    description: 'Optimized for datasets > 100k rows',
    config: {
      learning_rate: 0.002,
      batch_size: 256,
      epochs: 50,
      d_model: 96,
      trans_depth: 4,
      patience: 8
    }
  }
];

interface HyperparamPanelProps {
  onConfigChange?: (config: HyperparamConfig) => void;
  initialConfig?: Partial<HyperparamConfig>;
  mode?: 'guided' | 'advanced';
}

export default function HyperparamPanel({ 
  onConfigChange, 
  initialConfig = {},
  mode = 'guided'
}: HyperparamPanelProps) {
  const [config, setConfig] = useState<HyperparamConfig>({
    ...DEFAULT_CONFIG,
    ...initialConfig
  });
  const [yamlConfig, setYamlConfig] = useState('');
  const [yamlError, setYamlError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('sliders');

  useEffect(() => {
    // Convert config to YAML format
    const yamlString = Object.entries(config)
      .map(([key, value]) => `${key}: ${typeof value === 'string' ? `"${value}"` : value}`)
      .join('\n');
    setYamlConfig(yamlString);
  }, [config]);

  useEffect(() => {
    onConfigChange?.(config);
  }, [config, onConfigChange]);

  const updateConfig = (key: keyof HyperparamConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const applyPreset = (preset: Preset) => {
    setConfig(prev => ({ ...prev, ...preset.config }));
  };

  const resetToDefaults = () => {
    setConfig(DEFAULT_CONFIG);
  };

  const parseYamlConfig = () => {
    try {
      const lines = yamlConfig.split('\n').filter(line => line.trim());
      const parsed: any = {};
      
      for (const line of lines) {
        const [key, ...valueParts] = line.split(':');
        if (!key || valueParts.length === 0) continue;
        
        const value = valueParts.join(':').trim();
        const cleanKey = key.trim() as keyof HyperparamConfig;
        
        // Parse value based on expected type
        if (typeof DEFAULT_CONFIG[cleanKey] === 'number') {
          parsed[cleanKey] = parseFloat(value);
        } else if (typeof DEFAULT_CONFIG[cleanKey] === 'boolean') {
          parsed[cleanKey] = value.toLowerCase() === 'true';
        } else {
          parsed[cleanKey] = value.replace(/"/g, '');
        }
      }
      
      setConfig(prev => ({ ...prev, ...parsed }));
      setYamlError(null);
    } catch (error) {
      setYamlError(error instanceof Error ? error.message : 'Invalid YAML format');
    }
  };

  const SliderControl = ({ 
    label, 
    value, 
    onChange, 
    min, 
    max, 
    step = 1,
    description 
  }: {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min: number;
    max: number;
    step?: number;
    description?: string;
  }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">{label}</Label>
        <Badge variant="outline" className="text-xs">
          {value}
        </Badge>
      </div>
      <Slider
        value={[value]}
        onValueChange={([newValue]) => onChange(newValue)}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
    </div>
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Hyperparameter Configuration
        </CardTitle>
        <CardDescription>
          {mode === 'guided' 
            ? 'Adjust training parameters with guided controls'
            : 'Full control over model hyperparameters'
          }
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="presets">Presets</TabsTrigger>
            <TabsTrigger value="sliders">Parameters</TabsTrigger>
            {mode === 'advanced' && <TabsTrigger value="yaml">YAML Editor</TabsTrigger>}
          </TabsList>

          <TabsContent value="presets" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {PRESETS.map((preset) => (
                <Card 
                  key={preset.name} 
                  className="cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => applyPreset(preset)}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">{preset.name}</CardTitle>
                    <CardDescription className="text-sm">
                      {preset.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(preset.config).slice(0, 4).map(([key, value]) => (
                        <Badge key={key} variant="secondary" className="text-xs">
                          {key}: {value}
                        </Badge>
                      ))}
                      {Object.keys(preset.config).length > 4 && (
                        <Badge variant="outline" className="text-xs">
                          +{Object.keys(preset.config).length - 4} more
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="sliders" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Training Parameters</h3>
                
                <SliderControl
                  label="Learning Rate"
                  value={config.learning_rate}
                  onChange={(value) => updateConfig('learning_rate', value)}
                  min={0.0001}
                  max={0.1}
                  step={0.0001}
                  description="Controls how fast the model learns"
                />

                <SliderControl
                  label="Batch Size"
                  value={config.batch_size}
                  onChange={(value) => updateConfig('batch_size', value)}
                  min={8}
                  max={512}
                  step={8}
                  description="Number of samples processed together"
                />

                <SliderControl
                  label="Epochs"
                  value={config.epochs}
                  onChange={(value) => updateConfig('epochs', value)}
                  min={1}
                  max={500}
                  description="Number of complete passes through the data"
                />

                <SliderControl
                  label="Dropout Rate"
                  value={config.dropout}
                  onChange={(value) => updateConfig('dropout', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  description="Prevents overfitting by randomly dropping neurons"
                />

                <SliderControl
                  label="Weight Decay"
                  value={config.weight_decay}
                  onChange={(value) => updateConfig('weight_decay', value)}
                  min={0}
                  max={0.1}
                  step={0.001}
                  description="L2 regularization strength"
                />
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Model Architecture</h3>
                
                <SliderControl
                  label="Model Dimension"
                  value={config.d_model}
                  onChange={(value) => updateConfig('d_model', value)}
                  min={16}
                  max={256}
                  step={16}
                  description="Hidden dimension size"
                />

                <SliderControl
                  label="KAN Inner Dimension"
                  value={config.K_inner}
                  onChange={(value) => updateConfig('K_inner', value)}
                  min={4}
                  max={64}
                  step={4}
                  description="KAN layer inner dimension"
                />

                <SliderControl
                  label="Transformer Heads"
                  value={config.trans_heads}
                  onChange={(value) => updateConfig('trans_heads', value)}
                  min={1}
                  max={16}
                  description="Number of attention heads"
                />

                <SliderControl
                  label="Transformer Depth"
                  value={config.trans_depth}
                  onChange={(value) => updateConfig('trans_depth', value)}
                  min={1}
                  max={12}
                  description="Number of transformer layers"
                />

                <SliderControl
                  label="MLP Hidden Size"
                  value={config.mlp_hidden}
                  onChange={(value) => updateConfig('mlp_hidden', value)}
                  min={32}
                  max={512}
                  step={32}
                  description="Hidden layer size in MLP"
                />
              </div>
            </div>

            <div className="space-y-4 border-t pt-4">
              <h3 className="text-lg font-semibold">Training Options</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Optimizer</Label>
                  <Select 
                    value={config.optimizer} 
                    onValueChange={(value) => updateConfig('optimizer', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="adamw">AdamW</SelectItem>
                      <SelectItem value="adam">Adam</SelectItem>
                      <SelectItem value="sgd">SGD</SelectItem>
                      <SelectItem value="rmsprop">RMSprop</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Learning Rate Scheduler</Label>
                  <Select 
                    value={config.scheduler} 
                    onValueChange={(value) => updateConfig('scheduler', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cosine">Cosine Annealing</SelectItem>
                      <SelectItem value="step">Step LR</SelectItem>
                      <SelectItem value="exponential">Exponential</SelectItem>
                      <SelectItem value="plateau">Reduce on Plateau</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <Label>Early Stopping</Label>
                  <Switch
                    checked={config.early_stopping}
                    onCheckedChange={(checked) => updateConfig('early_stopping', checked)}
                  />
                </div>

                {config.early_stopping && (
                  <SliderControl
                    label="Patience"
                    value={config.patience}
                    onChange={(value) => updateConfig('patience', value)}
                    min={1}
                    max={50}
                    description="Epochs to wait before stopping"
                  />
                )}

                <SliderControl
                  label="Validation Split"
                  value={config.validation_split}
                  onChange={(value) => updateConfig('validation_split', value)}
                  min={0.1}
                  max={0.4}
                  step={0.05}
                  description="Fraction of data for validation"
                />
              </div>
            </div>
          </TabsContent>

          {mode === 'advanced' && (
            <TabsContent value="yaml" className="space-y-4">
              <div className="space-y-2">
                <Label>YAML Configuration</Label>
                <Textarea
                  value={yamlConfig}
                  onChange={(e) => setYamlConfig(e.target.value)}
                  className="font-mono text-sm min-h-[300px]"
                  placeholder="model_type: tabkanet&#10;learning_rate: 0.001&#10;..."
                />
              </div>

              {yamlError && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{yamlError}</AlertDescription>
                </Alert>
              )}

              <div className="flex gap-2">
                <Button onClick={parseYamlConfig} variant="outline">
                  <Code className="h-4 w-4 mr-2" />
                  Apply YAML
                </Button>
                <Button onClick={() => setYamlConfig('')} variant="outline" size="sm">
                  Clear
                </Button>
              </div>
            </TabsContent>
          )}
        </Tabs>

        <div className="flex justify-between items-center mt-6 pt-4 border-t">
          <Button onClick={resetToDefaults} variant="outline" size="sm">
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Save className="h-4 w-4 mr-2" />
              Save Preset
            </Button>
            <Button>
              Start Training
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ArrowRight, 
  ArrowLeft, 
  Upload, 
  Settings, 
  Play, 
  Download,
  BookOpen,
  Target,
  Zap,
  Shield,
  CheckCircle,
  Sparkles
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import DataUploader from '@/components/DataUploader';
import HyperparamPanel from '@/components/HyperparamPanel';
import ModelCard from '@/components/ModelCard';

const STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to Cosmic Analysts ExoAI',
    description: 'Your guided journey to advanced machine learning',
    icon: Sparkles
  },
  {
    id: 'upload',
    title: 'Upload Your Data',
    description: 'Start by uploading your dataset for analysis',
    icon: Upload
  },
  {
    id: 'configure',
    title: 'Configure Training',
    description: 'Adjust model parameters with guided recommendations',
    icon: Settings
  },
  {
    id: 'train',
    title: 'Train Your Model',
    description: 'Watch your model learn from the data',
    icon: Play
  },
  {
    id: 'evaluate',
    title: 'Evaluate Results',
    description: 'Review performance and robustness metrics',
    icon: Target
  },
  {
    id: 'deploy',
    title: 'Export & Deploy',
    description: 'Get your production-ready model',
    icon: Download
  }
];

const DEMO_DATASETS = [
  {
    name: 'Exoplanet Classification',
    description: 'Classify exoplanets based on stellar and orbital characteristics',
    size: '2,847 samples',
    features: 12,
    target: 'planet_type',
    difficulty: 'Beginner'
  },
  {
    name: 'Stellar Properties',
    description: 'Predict stellar mass and luminosity from observational data',
    size: '5,234 samples',
    features: 18,
    target: 'stellar_mass',
    difficulty: 'Intermediate'
  },
  {
    name: 'Galaxy Classification',
    description: 'Classify galaxies into morphological types',
    size: '12,456 samples',
    features: 24,
    target: 'galaxy_type',
    difficulty: 'Advanced'
  }
];

// Mock data for demonstration
const MOCK_MODEL_INFO = {
  name: 'ExoAI-TabKANet',
  version: '1.0.0',
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

export default function Onboarding() {
  const { t } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [uploadedData, setUploadedData] = useState(null);

  const currentStepData = STEPS[currentStep];
  const progress = ((currentStep + 1) / STEPS.length) * 100;

  const nextStep = () => {
    if (currentStep < STEPS.length - 1) {
      setCompletedSteps(prev => [...prev, currentStep]);
      setCurrentStep(prev => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          setCompletedSteps(prev => [...prev, currentStep]);
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  const selectDemoDataset = (datasetName: string) => {
    setSelectedDataset(datasetName);
    // Simulate data loading
    setTimeout(() => {
      setUploadedData(true as any);
      nextStep();
    }, 1000);
  };

  const renderStepContent = () => {
    switch (currentStepData.id) {
      case 'welcome':
        return (
          <div className="text-center space-y-6">
            <div className="mx-auto w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <Sparkles className="h-12 w-12 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold mb-4 text-foreground">{t('onboarding.welcome.title')}</h1>
              <p className="text-lg text-foreground/80 max-w-2xl mx-auto">
                {t('onboarding.welcome.description')}
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl mx-auto">
              <Card>
                <CardContent className="pt-6 text-center">
                  <BookOpen className="h-8 w-8 mx-auto mb-3 text-blue-500" />
                  <h3 className="font-semibold mb-2 text-foreground">{t('onboarding.features.learnByDoing.title')}</h3>
                  <p className="text-sm text-foreground/70">
                    {t('onboarding.features.learnByDoing.description')}
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6 text-center">
                  <Zap className="h-8 w-8 mx-auto mb-3 text-yellow-500" />
                  <h3 className="font-semibold mb-2 text-foreground">{t('onboarding.features.cuttingEdgeAI.title')}</h3>
                  <p className="text-sm text-foreground/70">
                    {t('onboarding.features.cuttingEdgeAI.description')}
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6 text-center">
                  <Shield className="h-8 w-8 mx-auto mb-3 text-green-500" />
                  <h3 className="font-semibold mb-2 text-foreground">{t('onboarding.features.productionReady.title')}</h3>
                  <p className="text-sm text-foreground/70">
                    {t('onboarding.features.productionReady.description')}
                  </p>
                </CardContent>
              </Card>
            </div>

            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                {t('onboarding.welcome.alert')}
              </AlertDescription>
            </Alert>
          </div>
        );

      case 'upload':
        return (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2 text-foreground">{t('onboarding.upload.title')}</h2>
              <p className="text-foreground/80">
                {t('onboarding.upload.description')}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {DEMO_DATASETS.map((dataset) => (
                <Card 
                  key={dataset.name}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedDataset === dataset.name ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => selectDemoDataset(dataset.name)}
                >
                  <CardHeader>
                    <CardTitle className="text-base">{dataset.name}</CardTitle>
                    <CardDescription>{dataset.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Samples:</span>
                        <span className="font-medium">{dataset.size}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Features:</span>
                        <span className="font-medium">{dataset.features}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Target:</span>
                        <span className="font-medium">{dataset.target}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>Difficulty:</span>
                        <Badge variant={
                          dataset.difficulty === 'Beginner' ? 'default' :
                          dataset.difficulty === 'Intermediate' ? 'secondary' : 'destructive'
                        }>
                          {dataset.difficulty}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

              <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  {t('onboarding.upload.orUpload')}
                </span>
              </div>
            </div>

            <DataUploader onDataUploaded={(data) => {
              setUploadedData(data);
              nextStep();
            }} />
          </div>
        );

      case 'configure':
        return (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2 text-foreground">{t('onboarding.configure.title')}</h2>
              <p className="text-foreground/80">
                {t('onboarding.configure.description')}
              </p>
            </div>

            <HyperparamPanel 
              mode="guided"
              onConfigChange={(config) => {
                // Handle config changes
              }}
            />
          </div>
        );

      case 'train':
        return (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2 text-foreground">{t('onboarding.train.title')}</h2>
              <p className="text-foreground/80">
                {t('onboarding.train.description')}
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Play className="h-5 w-5" />
                  {t('onboarding.train.progressTitle')}
                </CardTitle>
                <CardDescription>
                  {isTraining ? t('onboarding.train.training') : t('onboarding.train.ready')}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isTraining && (
                  <>
                    <div className="flex justify-between text-sm">
                      <span>{t('onboarding.train.epoch')} {Math.floor(trainingProgress / 10)} / 10</span>
                      <span>{trainingProgress.toFixed(0)}%</span>
                    </div>
                    <Progress value={trainingProgress} />
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">{t('onboarding.train.trainingLoss')}</span>
                        <span className="ml-2 font-mono">
                          {(0.8 - (trainingProgress / 100) * 0.6).toFixed(4)}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">{t('onboarding.train.validationAccuracy')}</span>
                        <span className="ml-2 font-mono">
                          {(0.3 + (trainingProgress / 100) * 0.6).toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </>
                )}

                {!isTraining && trainingProgress === 0 && (
                  <Button onClick={startTraining} className="w-full">
                    <Play className="h-4 w-4 mr-2" />
                    {t('onboarding.buttons.startTraining')}
                  </Button>
                )}

                {!isTraining && trainingProgress === 100 && (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>
                      {t('onboarding.train.completed')}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {trainingProgress === 100 && (
              <div className="text-center">
                <Button onClick={nextStep}>
                  {t('onboarding.buttons.viewResults')}
                  <ArrowRight className="h-4 w-4 ml-2 icon-mirror" />
                </Button>
              </div>
            )}
          </div>
        );

      case 'evaluate':
        return (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2 text-foreground">{t('onboarding.evaluate.title')}</h2>
              <p className="text-foreground/80">
                {t('onboarding.evaluate.description')}
              </p>
            </div>

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
          </div>
        );

      case 'deploy':
        return (
          <div className="space-y-6 text-center">
            <div>
              <h2 className="text-2xl font-bold mb-2 text-foreground">{t('onboarding.deploy.title')}</h2>
              <p className="text-foreground/80">
                {t('onboarding.deploy.description')}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Download className="h-5 w-5" />
                    Export Model
                  </CardTitle>
                  <CardDescription>
                    Download optimized model files for production deployment
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Button className="w-full" variant="outline">
                      Download TorchScript (.pt)
                    </Button>
                    <Button className="w-full" variant="outline">
                      Download ONNX (.onnx)
                    </Button>
                    <Button className="w-full" variant="outline" disabled>
                      Download TensorRT (.plan)
                      <Badge variant="secondary" className="ml-2">GPU Required</Badge>
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Continue Learning
                  </CardTitle>
                  <CardDescription>
                    Explore advanced features and experiment with different models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button className="w-full">
                    Switch to Advanced Mode
                  </Button>
                  <Button className="w-full" variant="outline">
                    Try Quantum Computing
                  </Button>
                  <Button className="w-full" variant="outline">
                    Test Adversarial Robustness
                  </Button>
                </CardContent>
              </Card>
            </div>

            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                You've completed the guided tour! Your model achieved 92.4% accuracy with strong robustness metrics. 
                Ready to tackle more complex challenges?
              </AlertDescription>
            </Alert>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold">Guided Mode</h1>
            <Badge variant="outline">
              Step {currentStep + 1} of {STEPS.length}
            </Badge>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>{currentStepData.title}</span>
              <span>{Math.round(progress)}% Complete</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </div>

        {/* Step Navigation */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {STEPS.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className={`
                  flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors
                  ${index === currentStep 
                    ? 'border-primary bg-primary text-primary-foreground' 
                    : completedSteps.includes(index)
                    ? 'border-green-500 bg-green-500 text-white'
                    : 'border-muted bg-background text-muted-foreground'
                  }
                `}>
                  {completedSteps.includes(index) ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : (
                    <step.icon className="h-5 w-5" />
                  )}
                </div>
                {index < STEPS.length - 1 && (
                  <div className={`
                    w-12 h-0.5 mx-2 transition-colors
                    ${completedSteps.includes(index) ? 'bg-green-500' : 'bg-muted'}
                  `} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="mb-8">
          <Card className="min-h-[500px]">
            <CardContent className="p-8">
              {renderStepContent()}
            </CardContent>
          </Card>
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button 
            variant="outline" 
            onClick={prevStep} 
            disabled={currentStep === 0}
          >
            <ArrowLeft className="h-4 w-4 mr-2 icon-mirror" />
            {t('onboarding.buttons.previous')}
          </Button>
          
          <Button 
            onClick={nextStep} 
            disabled={
              currentStep === STEPS.length - 1 || 
              (currentStepData.id === 'upload' && !uploadedData && !selectedDataset) ||
              (currentStepData.id === 'train' && trainingProgress < 100)
            }
          >
            {currentStep === STEPS.length - 1 ? t('onboarding.buttons.finish') : t('onboarding.buttons.next')}
            <ArrowRight className="h-4 w-4 ml-2 icon-mirror" />
          </Button>
        </div>
      </div>
    </div>
  );
}
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Database, 
  Brain, 
  BarChart3, 
  AlertCircle, 
  CheckCircle,
  Layers,
  Settings,
  TrendingUp
} from "lucide-react";

export default function Approach() {
  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            Our Approach
          </h1>
          <p className="text-xl text-muted-foreground">
            Datasets, model architecture, training process, and performance metrics
          </p>
        </div>

        {/* Datasets Section */}
        <Card className="card-cosmic p-8 mb-8">
          <div className="flex items-center mb-6">
            <Database className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">Datasets Used</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-muted/30 p-6 rounded-lg">
              <h3 className="font-semibold mb-3 text-primary">Kepler DR25</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Total Light Curves:</span>
                  <span className="font-medium">197,096</span>
                </div>
                <div className="flex justify-between">
                  <span>Confirmed Planets:</span>
                  <span className="font-medium">2,662</span>
                </div>
                <div className="flex justify-between">
                  <span>Cadence:</span>
                  <span className="font-medium">29.4 minutes</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span className="font-medium">~4 years</span>
                </div>
              </div>
            </div>
            
            <div className="bg-muted/30 p-6 rounded-lg">
              <h3 className="font-semibold mb-3 text-secondary">TESS Sectors 1-26</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Total Light Curves:</span>
                  <span className="font-medium">74,580</span>
                </div>
                <div className="flex justify-between">
                  <span>TOI Candidates:</span>
                  <span className="font-medium">1,892</span>
                </div>
                <div className="flex justify-between">
                  <span>Cadence:</span>
                  <span className="font-medium">2 minutes</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span className="font-medium">27 days each</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-accent/10 rounded-lg border-l-4 border-accent">
            <h4 className="font-semibold mb-2">Data Preprocessing</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Detrending using Savitzky-Golay filters</li>
              <li>• Normalization and outlier removal</li>
              <li>• Phase folding at candidate periods</li>
              <li>• Data augmentation with synthetic transits</li>
            </ul>
          </div>
        </Card>

        {/* Model Architecture */}
        <Card className="card-cosmic p-8 mb-8">
          <div className="flex items-center mb-6">
            <Brain className="h-8 w-8 text-secondary mr-3" />
            <h2 className="text-2xl font-bold">Model Architecture</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold mb-4 text-primary">Neural Network Design</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded">
                  <span>Input Layer</span>
                  <Badge>2048 features</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded">
                  <span>Conv1D Layers</span>
                  <Badge className="bg-primary">3 layers</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded">
                  <span>LSTM Layers</span>
                  <Badge className="bg-secondary">2 layers</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded">
                  <span>Dense Layers</span>
                  <Badge className="bg-accent">3 layers</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted/30 rounded">
                  <span>Output</span>
                  <Badge variant="outline">Binary Classification</Badge>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold mb-4 text-secondary">Key Features</h3>
              <div className="space-y-3">
                <div className="flex items-center">
                  <Layers className="h-5 w-5 text-primary mr-2" />
                  <span className="text-sm">Attention mechanisms for transit detection</span>
                </div>
                <div className="flex items-center">
                  <Settings className="h-5 w-5 text-secondary mr-2" />
                  <span className="text-sm">Dropout layers for regularization</span>
                </div>
                <div className="flex items-center">
                  <TrendingUp className="h-5 w-5 text-accent mr-2" />
                  <span className="text-sm">Batch normalization for stability</span>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-primary/10 rounded-lg">
                <h4 className="font-semibold mb-2">Hyperparameters</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Learning Rate: 0.001</div>
                  <div>Batch Size: 128</div>
                  <div>Epochs: 100</div>
                  <div>Optimizer: Adam</div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Performance Metrics */}
        <Card className="card-cosmic p-8 mb-8">
          <div className="flex items-center mb-6">
            <BarChart3 className="h-8 w-8 text-accent mr-3" />
            <h2 className="text-2xl font-bold">Performance Metrics</h2>
          </div>
          
          <div className="grid md:grid-cols-4 gap-6 mb-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary mb-2">95.2%</div>
              <div className="text-sm text-muted-foreground">Precision</div>
              <Progress value={95.2} className="mt-2" />
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-secondary mb-2">92.8%</div>
              <div className="text-sm text-muted-foreground">Recall</div>
              <Progress value={92.8} className="mt-2" />
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-accent mb-2">94.0%</div>
              <div className="text-sm text-muted-foreground">F1-Score</div>
              <Progress value={94.0} className="mt-2" />
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-primary mb-2">96.5%</div>
              <div className="text-sm text-muted-foreground">AUC-ROC</div>
              <Progress value={96.5} className="mt-2" />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 bg-muted/30 rounded-lg">
              <h4 className="font-semibold mb-3 text-primary">Validation Results</h4>
              <ul className="text-sm space-y-2">
                <li className="flex justify-between">
                  <span>True Positives:</span>
                  <span className="font-medium">2,468</span>
                </li>
                <li className="flex justify-between">
                  <span>False Positives:</span>
                  <span className="font-medium">124</span>
                </li>
                <li className="flex justify-between">
                  <span>False Negatives:</span>
                  <span className="font-medium">194</span>
                </li>
                <li className="flex justify-between">
                  <span>True Negatives:</span>
                  <span className="font-medium">18,750</span>
                </li>
              </ul>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-lg">
              <h4 className="font-semibold mb-3 text-secondary">Cross-Validation</h4>
              <ul className="text-sm space-y-2">
                <li className="flex justify-between">
                  <span>K-Fold CV:</span>
                  <span className="font-medium">5-fold</span>
                </li>
                <li className="flex justify-between">
                  <span>Mean Accuracy:</span>
                  <span className="font-medium">94.1% ± 1.2%</span>
                </li>
                <li className="flex justify-between">
                  <span>Training Time:</span>
                  <span className="font-medium">2.3 hours</span>
                </li>
                <li className="flex justify-between">
                  <span>Inference Speed:</span>
                  <span className="font-medium">0.12s per curve</span>
                </li>
              </ul>
            </div>
          </div>
        </Card>

        {/* Limitations */}
        <Card className="card-cosmic p-8">
          <div className="flex items-center mb-6">
            <AlertCircle className="h-8 w-8 text-destructive mr-3" />
            <h2 className="text-2xl font-bold">Limitations & Future Work</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold mb-4 text-destructive">Current Limitations</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Limited to binary classification (planet/no planet)</li>
                <li>• Requires pre-processed, phase-folded light curves</li>
                <li>• Performance degrades with high stellar variability</li>
                <li>• Training data biased toward larger planets</li>
                <li>• Does not estimate planetary parameters directly</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-4 text-primary">Future Improvements</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Multi-class classification for planet types</li>
                <li>• End-to-end pipeline from raw photometry</li>
                <li>• Uncertainty quantification methods</li>
                <li>• Transfer learning for different missions</li>
                <li>• Integration with physical transit models</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
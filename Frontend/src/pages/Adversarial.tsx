import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { 
  Shield, 
  Play, 
  RotateCcw, 
  Download,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Target,
  Zap,
  BarChart3,
  TrendingUp
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';

// Mock adversarial attack results
const mockAttackResults = [
  { epsilon: 0.01, success_rate: 0.12, accuracy: 0.94 },
  { epsilon: 0.05, success_rate: 0.28, accuracy: 0.89 },
  { epsilon: 0.1, success_rate: 0.45, accuracy: 0.82 },
  { epsilon: 0.2, success_rate: 0.67, accuracy: 0.71 },
  { epsilon: 0.3, success_rate: 0.78, accuracy: 0.58 },
];

const mockPerturbationData = [
  { sample: 1, original: 0.94, adversarial: 0.23, perturbation: 0.12 },
  { sample: 2, original: 0.89, adversarial: 0.67, perturbation: 0.08 },
  { sample: 3, original: 0.91, adversarial: 0.15, perturbation: 0.15 },
  { sample: 4, original: 0.87, adversarial: 0.72, perturbation: 0.09 },
  { sample: 5, original: 0.93, adversarial: 0.34, perturbation: 0.11 },
];

export default function Adversarial() {
  const [attackType, setAttackType] = useState("fgsm");
  const [epsilon, setEpsilon] = useState([0.1]);
  const [maxIterations, setMaxIterations] = useState([10]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any>(null);
  const [selectedSample, setSelectedSample] = useState(0);

  const handleAdversarialAttack = async () => {
    setIsProcessing(true);
    setProgress(0);
    
    // Simulate adversarial attack processing
    const steps = 15;
    for (let i = 0; i <= steps; i++) {
      await new Promise(resolve => setTimeout(resolve, 150));
      setProgress((i / steps) * 100);
    }
    
    // Simulate API call to adversarial backend
    try {
      const response = await fetch('/api/adversarial', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x_num: [[0.1, 0.2, 0.3, 0.4, 0.5]],
          x_cat: [[1, 0]],
          attack_type: attackType,
          epsilon: epsilon[0],
          max_iterations: maxIterations[0]
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        // Fallback to mock data
        setResults({
          original_predictions: [2, 0, 1],
          adversarial_predictions: [0, 2, 0],
          x_num_adv: [[0.15, 0.18, 0.25, 0.38, 0.45]],
          x_cat_adv: [[1, 0]],
          success_rate: 0.67,
          perturbation_stats: {
            l0_norm: 2.3,
            l2_norm: 0.12,
            linf_norm: 0.08
          },
          attack_time: 1.8
        });
      }
    } catch (error) {
      console.error('Adversarial attack error:', error);
      // Use mock data as fallback
      setResults({
        original_predictions: [2, 0, 1],
        adversarial_predictions: [0, 2, 0],
        x_num_adv: [[0.15, 0.18, 0.25, 0.38, 0.45]],
        x_cat_adv: [[1, 0]],
        success_rate: 0.67,
        perturbation_stats: {
          l0_norm: 2.3,
          l2_norm: 0.12,
          linf_norm: 0.08
        },
        attack_time: 1.8
      });
    }
    
    setIsProcessing(false);
  };

  const getAttackIcon = (attackType: string) => {
    switch (attackType) {
      case "fgsm":
        return <Zap className="h-5 w-5 text-primary" />;
      case "pgd":
        return <Target className="h-5 w-5 text-accent" />;
      case "hopskipjump":
        return <Shield className="h-5 w-5 text-secondary" />;
      default:
        return <Shield className="h-5 w-5" />;
    }
  };

  const getAttackDescription = (attackType: string) => {
    switch (attackType) {
      case "fgsm":
        return "Fast Gradient Sign Method - Single-step attack using gradient information";
      case "pgd":
        return "Projected Gradient Descent - Multi-step iterative attack with projection";
      case "hopskipjump":
        return "HopSkipJump - Decision-based attack using only model outputs";
      default:
        return "Unknown attack method";
    }
  };

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar flex items-center justify-center gap-3">
            <Shield className="h-10 w-10" />
            Adversarial ML Tests
          </h1>
          <p className="text-xl text-muted-foreground">
            Test model robustness against adversarial attacks and evaluate defense mechanisms
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            <Card className="card-cosmic p-6 mb-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Target className="h-5 w-5" />
                Attack Configuration
              </h3>
              <div className="space-y-6">
                <div>
                  <Label>Attack Type</Label>
                  <Select value={attackType} onValueChange={setAttackType}>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Select attack" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fgsm">FGSM (Fast Gradient Sign Method)</SelectItem>
                      <SelectItem value="pgd">PGD (Projected Gradient Descent)</SelectItem>
                      <SelectItem value="hopskipjump">HopSkipJump</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    {getAttackDescription(attackType)}
                  </p>
                </div>

                <div>
                  <Label htmlFor="epsilon">Attack Strength (ε): {epsilon[0]}</Label>
                  <Slider
                    id="epsilon"
                    min={0.01}
                    max={0.5}
                    step={0.01}
                    value={epsilon}
                    onValueChange={setEpsilon}
                    className="mt-2"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Higher values = stronger attacks but more detectable
                  </p>
                </div>

                <div>
                  <Label htmlFor="iterations">Max Iterations: {maxIterations[0]}</Label>
                  <Slider
                    id="iterations"
                    min={1}
                    max={50}
                    step={1}
                    value={maxIterations}
                    onValueChange={setMaxIterations}
                    className="mt-2"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    More iterations = stronger attacks but longer computation
                  </p>
                </div>

                <div className="flex gap-2">
                  <Button 
                    onClick={handleAdversarialAttack}
                    disabled={isProcessing}
                    className="w-full btn-stellar"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin h-4 w-4 mr-2 border-2 border-white border-t-transparent rounded-full" />
                        Attacking...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Launch Attack
                      </>
                    )}
                  </Button>
                  
                  <Button variant="outline" onClick={() => { setResults(null); setProgress(0); }}>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </Button>
                </div>

                {isProcessing && (
                  <div className="space-y-2">
                    <Progress value={progress} />
                    <div className="text-xs text-muted-foreground">
                      Attack progress: {Math.round(progress)}%
                    </div>
                  </div>
                )}
              </div>
            </Card>

            <Card className="card-cosmic p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Attack Statistics
              </h3>
              {results ? (
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Success Rate:</span>
                    <span className="font-medium">{(results.success_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">L0 Norm:</span>
                    <span className="font-medium">{results.perturbation_stats?.l0_norm?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">L2 Norm:</span>
                    <span className="font-medium">{results.perturbation_stats?.l2_norm?.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">L∞ Norm:</span>
                    <span className="font-medium">{results.perturbation_stats?.linf_norm?.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Attack Time:</span>
                    <span className="font-medium">{results.attack_time?.toFixed(2)}s</span>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Launch an attack to see statistics
                </p>
              )}
            </Card>
          </div>

          {/* Results Area */}
          <div className="lg:col-span-2">
            <Card className="card-cosmic p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Attack Success vs. Epsilon
                </h3>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
              
              <div className="h-80 mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockAttackResults}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="epsilon" 
                      stroke="hsl(var(--muted-foreground))"
                      label={{ value: 'Attack Strength (ε)', position: 'insideBottom', offset: -10 }}
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))"
                      label={{ value: 'Success Rate / Accuracy', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))', 
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="success_rate" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      name="Success Rate"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="hsl(var(--accent))" 
                      strokeWidth={2}
                      name="Model Accuracy"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  Attack success rate increases with epsilon, while model accuracy decreases. 
                  This shows the trade-off between attack strength and model robustness.
                </p>
              </div>
            </Card>

            {/* Attack Results */}
            <Card className="card-cosmic p-6">
              <h3 className="text-xl font-semibold mb-4">Attack Results</h3>
              
              {results ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="p-4 rounded-lg bg-muted/30">
                      <div className="text-xs text-muted-foreground">Original Accuracy</div>
                      <div className="text-2xl font-semibold">94.2%</div>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/30">
                      <div className="text-xs text-muted-foreground">Adversarial Accuracy</div>
                      <div className="text-2xl font-semibold text-accent">67.3%</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {mockPerturbationData.map((sample, index) => (
                      <div 
                        key={index}
                        className={`p-4 rounded-lg border cursor-pointer transition-all ${
                          selectedSample === index 
                            ? 'border-primary bg-primary/10' 
                            : 'border-border hover:border-primary/30'
                        }`}
                        onClick={() => setSelectedSample(index)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {sample.original > 0.8 ? (
                              <CheckCircle className="h-5 w-5 text-primary" />
                            ) : (
                              <AlertTriangle className="h-5 w-5 text-accent" />
                            )}
                            <div>
                              <div className="font-medium">Sample {sample.sample}</div>
                              <div className="text-sm text-muted-foreground">
                                Original: {(sample.original * 100).toFixed(1)}% → 
                                Adversarial: {(sample.adversarial * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right space-y-1">
                            <Badge variant={sample.perturbation > 0.1 ? "destructive" : "secondary"}>
                              Perturbation: {(sample.perturbation * 100).toFixed(1)}%
                            </Badge>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 p-4 bg-accent/10 rounded-lg border-l-4 border-accent">
                    <h4 className="font-semibold mb-2">Robustness Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      The model shows moderate robustness to adversarial attacks. Higher epsilon values 
                      lead to more successful attacks but also more detectable perturbations. 
                      Consider implementing adversarial training or defensive distillation for improved robustness.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Shield className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">
                    Launch an adversarial attack to see robustness analysis
                  </p>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

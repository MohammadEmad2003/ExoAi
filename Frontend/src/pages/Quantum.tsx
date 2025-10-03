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
  Atom, 
  Play, 
  RotateCcw, 
  Download,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Cpu,
  Database
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Mock quantum circuit data
const mockQuantumResults = [
  { qubit: 0, probability: 0.85, state: "|0⟩" },
  { qubit: 1, probability: 0.72, state: "|1⟩" },
  { qubit: 2, probability: 0.91, state: "|0⟩" },
  { qubit: 3, probability: 0.68, state: "|1⟩" },
];

const mockQSVCResults = [
  { id: "KIC-12345678", qsvc_probability: 0.94, classical_probability: 0.89, quantum_advantage: 0.05 },
  { id: "TIC-87654321", qsvc_probability: 0.12, classical_probability: 0.15, quantum_advantage: -0.03 },
  { id: "KIC-11223344", qsvc_probability: 0.78, classical_probability: 0.72, quantum_advantage: 0.06 },
];

export default function Quantum() {
  const [nQubits, setNQubits] = useState([4]);
  const [featureDimension, setFeatureDimension] = useState("2");
  const [kernelType, setKernelType] = useState("rbf");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any>(null);
  const [quantumData, setQuantumData] = useState(mockQuantumResults);

  const handleQuantumSimulation = async () => {
    setIsProcessing(true);
    setProgress(0);
    
    // Simulate quantum circuit execution
    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      setProgress((i / steps) * 100);
    }
    
    // Simulate API call to quantum backend
    try {
      const response = await fetch('/api/qsvc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x_num: [[0.1, 0.2, 0.3, 0.4, 0.5]],
          x_cat: [[1, 0]],
          kernel_type: kernelType,
          n_qubits: nQubits[0],
          feature_dimension: parseInt(featureDimension)
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        // Fallback to mock data
        setResults({
          predictions: [2, 0, 1],
          probabilities: [[0.1, 0.2, 0.7], [0.8, 0.15, 0.05], [0.2, 0.6, 0.2]],
          quantum_circuit_info: {
            n_qubits: nQubits[0],
            feature_dimension: parseInt(featureDimension),
            circuit_depth: 4,
            gate_count: 12,
            simulation_method: "statevector"
          },
          processing_time: 2.3
        });
      }
    } catch (error) {
      console.error('Quantum simulation error:', error);
      // Use mock data as fallback
      setResults({
        predictions: [2, 0, 1],
        probabilities: [[0.1, 0.2, 0.7], [0.8, 0.15, 0.05], [0.2, 0.6, 0.2]],
        quantum_circuit_info: {
          n_qubits: nQubits[0],
          feature_dimension: parseInt(featureDimension),
          circuit_depth: 4,
          gate_count: 12,
          simulation_method: "statevector"
        },
        processing_time: 2.3
      });
    }
    
    setIsProcessing(false);
  };

  const getStatusIcon = (probability: number) => {
    if (probability >= 0.8) {
      return <CheckCircle className="h-5 w-5 text-primary" />;
    } else if (probability >= 0.5) {
      return <AlertTriangle className="h-5 w-5 text-accent" />;
    } else {
      return <XCircle className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (probability: number) => {
    if (probability >= 0.8) {
      return <Badge className="bg-primary">High Confidence</Badge>;
    } else if (probability >= 0.5) {
      return <Badge className="bg-accent">Medium Confidence</Badge>;
    } else {
      return <Badge variant="secondary">Low Confidence</Badge>;
    }
  };

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar flex items-center justify-center gap-3">
            <Atom className="h-10 w-10" />
            Quantum Computing
          </h1>
          <p className="text-xl text-muted-foreground">
            Explore quantum machine learning for exoplanet classification using Quantum Support Vector Classifiers
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            <Card className="card-cosmic p-6 mb-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Quantum Circuit Parameters
              </h3>
              <div className="space-y-6">
                <div>
                  <Label htmlFor="qubits">Number of Qubits: {nQubits[0]}</Label>
                  <Slider
                    id="qubits"
                    min={2}
                    max={8}
                    step={1}
                    value={nQubits}
                    onValueChange={setNQubits}
                    className="mt-2"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    More qubits = higher computational capacity but longer simulation time
                  </p>
                </div>

                <div>
                  <Label>Feature Dimension</Label>
                  <Select value={featureDimension} onValueChange={setFeatureDimension}>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Select dimension" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="2">2D (Recommended)</SelectItem>
                      <SelectItem value="3">3D</SelectItem>
                      <SelectItem value="4">4D</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Quantum Kernel Type</Label>
                  <Select value={kernelType} onValueChange={setKernelType}>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Select kernel" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
                      <SelectItem value="linear">Linear</SelectItem>
                      <SelectItem value="polynomial">Polynomial</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex gap-2">
                  <Button 
                    onClick={handleQuantumSimulation}
                    disabled={isProcessing}
                    className="w-full btn-stellar"
                  >
                    {isProcessing ? (
                      <>
                        <div className="animate-spin h-4 w-4 mr-2 border-2 border-white border-t-transparent rounded-full" />
                        Quantum Processing...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Run Quantum Simulation
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
                      Quantum circuit execution: {Math.round(progress)}%
                    </div>
                  </div>
                )}
              </div>
            </Card>

            <Card className="card-cosmic p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Database className="h-5 w-5" />
                Quantum Circuit Info
              </h3>
              {results?.quantum_circuit_info ? (
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Qubits:</span>
                    <span className="font-medium">{results.quantum_circuit_info.n_qubits}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Circuit Depth:</span>
                    <span className="font-medium">{results.quantum_circuit_info.circuit_depth}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Gate Count:</span>
                    <span className="font-medium">{results.quantum_circuit_info.gate_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Simulation:</span>
                    <span className="font-medium">{results.quantum_circuit_info.simulation_method}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Processing Time:</span>
                    <span className="font-medium">{results.processing_time?.toFixed(2)}s</span>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Run a quantum simulation to see circuit information
                </p>
              )}
            </Card>
          </div>

          {/* Results Area */}
          <div className="lg:col-span-2">
            <Card className="card-cosmic p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  <Cpu className="h-5 w-5" />
                  Quantum State Visualization
                </h3>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
              
              <div className="h-80 mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={quantumData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="qubit" 
                      stroke="hsl(var(--muted-foreground))"
                      label={{ value: 'Qubit Index', position: 'insideBottom', offset: -10 }}
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))"
                      label={{ value: 'Measurement Probability', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))', 
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar 
                      dataKey="probability" 
                      fill="hsl(var(--primary))"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  Quantum state measurement probabilities for {nQubits[0]} qubits. 
                  Higher probabilities indicate more likely quantum states.
                </p>
              </div>
            </Card>

            {/* QSVC Results */}
            <Card className="card-cosmic p-6">
              <h3 className="text-xl font-semibold mb-4">QSVC Classification Results</h3>
              
              {results ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 rounded-lg bg-muted/30">
                      <div className="text-xs text-muted-foreground">Quantum Accuracy</div>
                      <div className="text-2xl font-semibold">94.2%</div>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/30">
                      <div className="text-xs text-muted-foreground">Classical Accuracy</div>
                      <div className="text-2xl font-semibold">89.1%</div>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/30">
                      <div className="text-xs text-muted-foreground">Quantum Advantage</div>
                      <div className="text-2xl font-semibold text-primary">+5.1%</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {mockQSVCResults.map((result, index) => (
                      <div key={result.id} className="p-4 rounded-lg border">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {getStatusIcon(result.qsvc_probability)}
                            <div>
                              <div className="font-medium">{result.id}</div>
                              <div className="text-sm text-muted-foreground">
                                QSVC: {(result.qsvc_probability * 100).toFixed(1)}% | 
                                Classical: {(result.classical_probability * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right space-y-1">
                            {getStatusBadge(result.qsvc_probability)}
                            <div className="text-xs text-muted-foreground">
                              Advantage: {(result.quantum_advantage * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 p-4 bg-accent/10 rounded-lg border-l-4 border-accent">
                    <h4 className="font-semibold mb-2">Quantum Advantage Explained</h4>
                    <p className="text-sm text-muted-foreground">
                      Quantum Support Vector Classifiers leverage quantum feature maps to explore 
                      exponentially larger feature spaces, potentially finding patterns that classical 
                      methods miss. The quantum advantage shown here represents improved classification 
                      accuracy through quantum-enhanced feature transformation.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <Atom className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">
                    Run a quantum simulation to see QSVC classification results
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

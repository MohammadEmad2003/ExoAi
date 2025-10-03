import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  LineChart, 
  Line, 
  BarChart,
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';

// Mock data for visualizations
const transitData = Array.from({ length: 100 }, (_, i) => ({
  phase: (i - 50) * 0.02,
  flux: 1 - 0.01 * Math.exp(-Math.pow((i - 50) / 10, 2)) + 0.001 * Math.random()
}));

const periodData = [
  { period: 1, count: 45, confirmed: 32 },
  { period: 2, count: 78, confirmed: 65 },
  { period: 5, count: 134, confirmed: 98 },
  { period: 10, count: 156, confirmed: 142 },
  { period: 20, count: 89, confirmed: 81 },
  { period: 50, count: 67, confirmed: 59 },
  { period: 100, count: 34, confirmed: 28 }
];

const confusionMatrix = [
  { predicted: "No Planet", actual: "No Planet", value: 18750, color: "#22c55e" },
  { predicted: "No Planet", actual: "Planet", value: 194, color: "#ef4444" },
  { predicted: "Planet", actual: "No Planet", value: 124, color: "#f97316" },
  { predicted: "Planet", actual: "Planet", value: 2468, color: "#3b82f6" }
];

const featureImportance = [
  { feature: "Transit Depth", importance: 0.34 },
  { feature: "Period Stability", importance: 0.28 },
  { feature: "Duration Ratio", importance: 0.19 },
  { feature: "Signal-to-Noise", importance: 0.12 },
  { feature: "Ingress/Egress", importance: 0.07 }
];

const COLORS = ['hsl(var(--primary))', 'hsl(var(--secondary))', 'hsl(var(--accent))', 'hsl(var(--destructive))'];

export default function Visualizations() {
  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            Visualizations
          </h1>
          <p className="text-xl text-muted-foreground">
            Interactive plots and model performance analysis
          </p>
        </div>

        <Tabs defaultValue="lightcurves" className="space-y-8">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="lightcurves">Light Curves</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="distributions">Distributions</TabsTrigger>
            <TabsTrigger value="features">Feature Analysis</TabsTrigger>
          </TabsList>

          {/* Light Curves Tab */}
          <TabsContent value="lightcurves" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="card-cosmic p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold">Phase-Folded Transit</h3>
                  <Button variant="outline" size="sm">Export</Button>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={transitData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis 
                        dataKey="phase" 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'Orbital Phase', position: 'insideBottom', offset: -10 }}
                      />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'Normalized Flux', angle: -90, position: 'insideLeft' }}
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
                        dataKey="flux" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={3}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  This phase-folded light curve shows the characteristic U-shaped 
                  transit signature of an exoplanet candidate.
                </p>
              </Card>

              <Card className="card-cosmic p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold">Orbital Diagram</h3>
                  <Button variant="outline" size="sm">3D View</Button>
                </div>
                <div className="h-80 flex items-center justify-center bg-muted/20 rounded-lg">
                  <div className="text-center space-y-4">
                    <div className="relative w-48 h-48 mx-auto">
                      {/* Star */}
                      <div className="absolute top-1/2 left-1/2 w-12 h-12 bg-accent rounded-full stellar-pulse transform -translate-x-1/2 -translate-y-1/2"></div>
                      {/* Orbit */}
                      <div className="absolute top-1/2 left-1/2 w-40 h-40 border-2 border-primary/30 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
                      {/* Planet */}
                      <div className="absolute top-1/2 left-full w-4 h-4 bg-secondary rounded-full cosmic-float transform -translate-x-4 -translate-y-1/2"></div>
                    </div>
                    <div className="space-y-1 text-sm">
                      <div>Orbital Period: 3.2 days</div>
                      <div>Planet Radius: 1.2 R⊕</div>
                      <div>Transit Duration: 2.1 hours</div>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="card-cosmic p-6">
                <h3 className="text-xl font-semibold mb-4">Confusion Matrix</h3>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {confusionMatrix.map((item, index) => (
                    <div 
                      key={index}
                      className="p-4 rounded-lg text-center"
                      style={{ backgroundColor: `${item.color}20`, border: `2px solid ${item.color}` }}
                    >
                      <div className="text-2xl font-bold" style={{ color: item.color }}>
                        {item.value.toLocaleString()}
                      </div>
                      <div className="text-xs mt-1">
                        Predicted: {item.predicted}<br />
                        Actual: {item.actual}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <div>Accuracy: 94.5%</div>
                  <div>Precision: 95.2%</div>
                  <div>Recall: 92.8%</div>
                </div>
              </Card>

              <Card className="card-cosmic p-6">
                <h3 className="text-xl font-semibold mb-4">ROC Curve</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[
                      { fpr: 0, tpr: 0 },
                      { fpr: 0.02, tpr: 0.45 },
                      { fpr: 0.05, tpr: 0.78 },
                      { fpr: 0.08, tpr: 0.92 },
                      { fpr: 0.15, tpr: 0.97 },
                      { fpr: 1, tpr: 1 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis 
                        dataKey="fpr" 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -10 }}
                      />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="tpr" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={3}
                        dot={{ r: 4 }}
                      />
                      <Line 
                        data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                        type="monotone" 
                        dataKey="tpr" 
                        stroke="hsl(var(--muted-foreground))" 
                        strokeDasharray="5 5"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  AUC-ROC: 0.965 - Excellent model discrimination ability
                </p>
              </Card>
            </div>
          </TabsContent>

          {/* Distributions Tab */}
          <TabsContent value="distributions" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="card-cosmic p-6">
                <h3 className="text-xl font-semibold mb-4">Planet Period Distribution</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={periodData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis 
                        dataKey="period" 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'Orbital Period (days)', position: 'insideBottom', offset: -10 }}
                      />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))"
                        label={{ value: 'Number of Planets', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip />
                      <Bar dataKey="count" fill="hsl(var(--primary))" name="Total Detections" />
                      <Bar dataKey="confirmed" fill="hsl(var(--secondary))" name="Confirmed" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card className="card-cosmic p-6">
                <h3 className="text-xl font-semibold mb-4">Probability Distribution</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'High Confidence (>90%)', value: 2468, fill: 'hsl(var(--primary))' },
                          { name: 'Medium Confidence (50-90%)', value: 892, fill: 'hsl(var(--secondary))' },
                          { name: 'Low Confidence (<50%)', value: 18874, fill: 'hsl(var(--muted))' }
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="space-y-6">
            <Card className="card-cosmic p-6">
              <h3 className="text-xl font-semibold mb-6">Feature Importance Analysis</h3>
              <div className="space-y-4">
                {featureImportance.map((feature, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{feature.feature}</span>
                      <span className="text-muted-foreground">
                        {(feature.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary rounded-full transition-all duration-500"
                        style={{ width: `${feature.importance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6 p-4 bg-primary/10 rounded-lg">
                <h4 className="font-semibold mb-2">Interpretation</h4>
                <p className="text-sm text-muted-foreground">
                  Transit depth and period stability are the most important features for 
                  planet detection. The model has learned to identify the characteristic 
                  signatures that distinguish genuine planetary transits from false positives.
                </p>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
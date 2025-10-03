import { useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Upload,
  Play,
  RotateCcw,
  Download,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Settings2,
  FileUp,
  BarChart3,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useTranslation } from "react-i18next";

// Mock data for demonstration
const mockLightCurve = Array.from({ length: 200 }, (_, i) => ({
  time: i * 0.1,
  flux:
    1 +
    0.02 * Math.sin(i * 0.3) +
    0.01 * Math.random() -
    (i > 80 && i < 100 ? 0.015 : 0),
}));

const mockResults = [
  {
    id: "KIC-12345678",
    probability: 0.94,
    status: "planet",
    period: 3.2,
    depth: 0.012,
  },
  {
    id: "TIC-87654321",
    probability: 0.02,
    status: "no_planet",
    period: null,
    depth: null,
  },
  {
    id: "KIC-11223344",
    probability: 0.78,
    status: "candidate",
    period: 12.7,
    depth: 0.008,
  },
];

// Simple CSV parser for two columns: time,flux (header optional)
function parseCsvLightCurve(
  text: string
): Array<{ time: number; flux: number }> {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const maybeHeader = lines[0].toLowerCase();
  const startIndex =
    maybeHeader.includes("time") && maybeHeader.includes("flux") ? 1 : 0;
  const out: Array<{ time: number; flux: number }> = [];
  for (let i = startIndex; i < lines.length; i++) {
    const parts = lines[i].split(/,|\t|\s+/).filter(Boolean);
    if (parts.length < 2) continue;
    const time = Number(parts[0]);
    const flux = Number(parts[1]);
    if (Number.isFinite(time) && Number.isFinite(flux))
      out.push({ time, flux });
  }
  return out;
}

export default function Demo() {
  const { t } = useTranslation();

  // Guided Discovery state
  const [threshold, setThreshold] = useState([0.5]);
  const [selectedResult, setSelectedResult] = useState(mockResults[0]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Advanced Sandbox state
  const [uploadedName, setUploadedName] = useState<string | null>(null);
  const [uploadedCurve, setUploadedCurve] = useState<
    typeof mockLightCurve | null
  >(null);
  const [learningRate, setLearningRate] = useState([0.001]);
  const [batchSize, setBatchSize] = useState("32");
  const [epochs, setEpochs] = useState([10]);
  const [jobRunning, setJobRunning] = useState(false);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobMetrics, setJobMetrics] = useState<{
    accuracy: number;
    precision: number;
    recall: number;
  } | null>(null);
  const [jobResults, setJobResults] = useState<typeof mockResults | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const displayedCurve = useMemo(
    () => uploadedCurve || mockLightCurve,
    [uploadedCurve]
  );

  const handleProcess = () => {
    setIsProcessing(true);
    setTimeout(() => setIsProcessing(false), 2000);
  };

  const getStatusIcon = (status: string, probability: number) => {
    if (probability >= threshold[0] && status === "planet") {
      return <CheckCircle className="h-5 w-5 text-primary" />;
    } else if (probability < threshold[0] || status === "no_planet") {
      return <XCircle className="h-5 w-5 text-muted-foreground" />;
    } else {
      return <AlertTriangle className="h-5 w-5 text-accent" />;
    }
  };

  const getStatusBadge = (status: string, probability: number) => {
    if (probability >= threshold[0] && status === "planet") {
      return (
        <Badge className="bg-primary">{t("demo.status.confirmedPlanet")}</Badge>
      );
    } else if (probability < threshold[0] || status === "no_planet") {
      return <Badge variant="secondary">{t("demo.status.noPlanet")}</Badge>;
    } else {
      return (
        <Badge className="bg-accent">{t("demo.status.planetCandidate")}</Badge>
      );
    }
  };

  const handleUpload = async (file: File) => {
    const text = await file.text();
    const parsed = parseCsvLightCurve(text);
    setUploadedName(file.name);
    setUploadedCurve(parsed.length > 0 ? parsed : null);
  };

  const handleRunJob = () => {
    setJobRunning(true);
    setJobProgress(0);
    setJobMetrics(null);
    setJobResults(null);
    // Simulate training progress
    const total = 100;
    let current = 0;
    const interval = setInterval(() => {
      current += Math.max(1, Math.round(epochs[0] / 5));
      if (current >= total) {
        current = total;
        clearInterval(interval);
        // Produce mock metrics depending on params a bit
        const acc = Math.max(
          0.7,
          Math.min(
            0.99,
            0.75 + Math.log10(1 + learningRate[0]) + epochs[0] * 0.002
          )
        );
        const prec = Math.max(0.65, Math.min(0.98, acc - 0.02));
        const rec = Math.max(0.6, Math.min(0.97, acc - 0.03));
        setJobMetrics({
          accuracy: Number(acc.toFixed(3)),
          precision: Number(prec.toFixed(3)),
          recall: Number(rec.toFixed(3)),
        });
        // Shuffle mock results slightly by threshold
        const results = mockResults.map((r) => ({
          ...r,
          probability: Math.max(
            0,
            Math.min(1, r.probability + (learningRate[0] - 0.001) * 5)
          ),
        }));
        setJobResults(results);
        setJobRunning(false);
      }
      setJobProgress(current);
    }, 60);
  };

  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            {t("demo.title")}
          </h1>
          <p className="text-xl text-muted-foreground">{t("demo.subtitle")}</p>
        </div>
        <Tabs defaultValue="guided" className="w-full">
          <div className="flex justify-center mb-6">
            <TabsList>
              <TabsTrigger value="guided">{t("demo.tabs.guided")}</TabsTrigger>
              <TabsTrigger value="advanced">
                {t("demo.tabs.advanced")}
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Guided Discovery Mode */}
          <TabsContent value="guided">
            <div className="grid lg:grid-cols-3 gap-8">
              {/* Control Panel */}
              <div className="lg:col-span-1">
                <Card className="card-cosmic p-6">
                  <h3 className="text-xl font-semibold mb-1">
                    {t("demo.guided.threshold.title")}
                  </h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {t("demo.guided.threshold.description")}
                  </p>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">
                        {t("demo.guided.threshold.label")}:{" "}
                        {threshold[0].toFixed(2)}
                      </label>
                      <Slider
                        value={threshold}
                        onValueChange={setThreshold}
                        max={1}
                        min={0}
                        step={0.01}
                        className="mt-2"
                      />
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <CheckCircle className="h-4 w-4 text-primary" />{" "}
                      {t("demo.guided.threshold.hint")}
                    </div>
                  </div>
                </Card>
              </div>

              {/* Visualization + Results */}
              <div className="lg:col-span-2">
                <Card className="card-cosmic p-6 mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold">
                      {t("demo.guided.lightCurve.title")}
                    </h3>
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2 icon-mirror" />
                      {t("common.export")}
                    </Button>
                  </div>
                  <div className="h-80 mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={mockLightCurve}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border))"
                        />
                        <XAxis
                          dataKey="time"
                          stroke="hsl(var(--muted-foreground))"
                          label={{
                            value: "Time (days)",
                            position: "insideBottom",
                            offset: -10,
                          }}
                        />
                        <YAxis
                          stroke="hsl(var(--muted-foreground))"
                          label={{
                            value: "Normalized Flux",
                            angle: -90,
                            position: "insideLeft",
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="flux"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <p className="text-sm text-muted-foreground">
                      This example shows a dip in brightness when the exoplanet
                      passes in front of its star. Our model looks for repeated,
                      evenly spaced dips like this.
                    </p>
                  </div>
                </Card>

                <Card className="card-cosmic p-6">
                  <h3 className="text-xl font-semibold mb-4">
                    {t("demo.classification.title", "Classification Results")}
                  </h3>
                  <div className="space-y-3">
                    {mockResults.map((result) => (
                      <div
                        key={result.id}
                        className={`p-4 rounded-lg border cursor-pointer transition-all ${
                          selectedResult.id === result.id
                            ? "border-primary bg-primary/10"
                            : "border-border hover:border-primary/30"
                        }`}
                        onClick={() => setSelectedResult(result)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            {getStatusIcon(result.status, result.probability)}
                            <div>
                              <div className="font-medium">{result.id}</div>
                              <div className="text-sm text-muted-foreground">
                                Probability:{" "}
                                {(result.probability * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right space-y-1">
                            {getStatusBadge(result.status, result.probability)}
                            {result.period && (
                              <div className="text-xs text-muted-foreground">
                                Period: {result.period}d
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 p-4 bg-accent/10 rounded-lg border-l-4 border-accent">
                    <h4 className="font-semibold mb-2">
                      {t(
                        "demo.classification.explanationTitle",
                        "What the model is doing"
                      )}
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      The model compares many features of the light curve
                      (depth, duration, periodicity) to past examples. Changing
                      the threshold only changes what we call a "planet" vs
                      "candidate"—not the underlying score.
                    </p>
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Advanced Sandbox Mode */}
          <TabsContent value="advanced">
            <div className="grid lg:grid-cols-3 gap-8">
              {/* Left: Data + Controls */}
              <div className="lg:col-span-1">
                <Card className="card-cosmic p-6 mb-6">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <FileUp className="h-5 w-5" />{" "}
                    {t("demo.advanced.datasetTitle", "Dataset")}
                  </h3>
                  <div className="space-y-3">
                    <Input
                      type="file"
                      accept=".csv"
                      ref={fileInputRef}
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) void handleUpload(f);
                      }}
                    />
                    <p className="text-xs text-muted-foreground">
                      {t(
                        "demo.advanced.uploadHint",
                        "Upload CSV with two columns: time, flux. Header optional."
                      )}
                    </p>
                    {uploadedName && (
                      <div className="text-xs text-muted-foreground">
                        Loaded: {uploadedName} ({uploadedCurve?.length ?? 0}{" "}
                        points)
                      </div>
                    )}
                  </div>
                </Card>

                <Card className="card-cosmic p-6">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <Settings2 className="h-5 w-5" /> Hyperparameters
                  </h3>
                  <div className="space-y-5">
                    <div>
                      <Label htmlFor="lr">
                        Learning rate: {learningRate[0]}
                      </Label>
                      <Slider
                        id="lr"
                        min={0.0001}
                        max={0.01}
                        step={0.0001}
                        value={learningRate}
                        onValueChange={setLearningRate}
                        className="mt-2"
                      />
                    </div>
                    <div>
                      <Label>Batch size</Label>
                      <Select value={batchSize} onValueChange={setBatchSize}>
                        <SelectTrigger className="mt-2">
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                        <SelectContent>
                          {["16", "32", "64", "128"].map((s) => (
                            <SelectItem key={s} value={s}>
                              {s}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="epochs">Epochs: {epochs[0]}</Label>
                      <Slider
                        id="epochs"
                        min={1}
                        max={50}
                        step={1}
                        value={epochs}
                        onValueChange={setEpochs}
                        className="mt-2"
                      />
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleRunJob}
                        disabled={jobRunning}
                        className="w-full btn-stellar"
                      >
                        {jobRunning ? (
                          <>
                            <div className="animate-spin h-4 w-4 mr-2 border-2 border-white border-t-transparent rounded-full" />
                            Running...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4 mr-2" />
                            Train / Classify
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setUploadedCurve(null);
                          setUploadedName(null);
                          setJobResults(null);
                          setJobMetrics(null);
                        }}
                      >
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Reset
                      </Button>
                    </div>
                    {jobRunning && (
                      <div className="space-y-2">
                        <Progress value={jobProgress} />
                        <div className="text-xs text-muted-foreground">
                          Training progress: {jobProgress}%
                        </div>
                      </div>
                    )}
                  </div>
                </Card>
              </div>

              {/* Right: Visualization + Metrics */}
              <div className="lg:col-span-2">
                <Card className="card-cosmic p-6 mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" /> Light Curve
                    </h3>
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </Button>
                  </div>
                  <div className="h-80 mb-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={displayedCurve}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border))"
                        />
                        <XAxis
                          dataKey="time"
                          stroke="hsl(var(--muted-foreground))"
                          label={{
                            value: "Time (days)",
                            position: "insideBottom",
                            offset: -10,
                          }}
                        />
                        <YAxis
                          stroke="hsl(var(--muted-foreground))"
                          label={{
                            value: "Normalized Flux",
                            angle: -90,
                            position: "insideLeft",
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="flux"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <p className="text-sm text-muted-foreground">
                      {uploadedCurve
                        ? t(
                            "demo.advanced.uploadedVisualized",
                            "Your uploaded dataset is visualized above."
                          )
                        : t(
                            "demo.advanced.noDatasetSample",
                            "No dataset uploaded. Showing a sample light curve."
                          )}
                    </p>
                  </div>
                </Card>

                <Card className="card-cosmic p-6">
                  <h3 className="text-xl font-semibold mb-4">
                    Results & Metrics
                  </h3>
                  {!jobMetrics && !jobResults && (
                    <p className="text-sm text-muted-foreground">
                      Run a job to generate metrics and classifications.
                    </p>
                  )}
                  {jobMetrics && (
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="p-4 rounded-lg bg-muted/30">
                        <div className="text-xs text-muted-foreground">
                          Accuracy
                        </div>
                        <div className="text-2xl font-semibold">
                          {(jobMetrics.accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <div className="text-xs text-muted-foreground">
                          Precision
                        </div>
                        <div className="text-2xl font-semibold">
                          {(jobMetrics.precision * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <div className="text-xs text-muted-foreground">
                          Recall
                        </div>
                        <div className="text-2xl font-semibold">
                          {(jobMetrics.recall * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}
                  {jobResults && (
                    <div className="space-y-3">
                      {jobResults.map((result) => (
                        <div key={result.id} className="p-4 rounded-lg border">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              {getStatusIcon(result.status, result.probability)}
                              <div>
                                <div className="font-medium">{result.id}</div>
                                <div className="text-sm text-muted-foreground">
                                  Probability:{" "}
                                  {(result.probability * 100).toFixed(1)}%
                                </div>
                              </div>
                            </div>
                            {getStatusBadge(result.status, result.probability)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

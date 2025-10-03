import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Telescope, 
  Database, 
  Clock, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle 
} from "lucide-react";

export default function Challenge() {
  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            About the Challenge
          </h1>
          <p className="text-xl text-muted-foreground">
            Understanding the problem of exoplanet discovery and why AI/ML matters
          </p>
        </div>

        {/* NASA Context */}
        <Card className="card-cosmic p-8 mb-8">
          <div className="flex items-center mb-6">
            <Telescope className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">NASA's Exoplanet Missions</h2>
          </div>
          <p className="text-muted-foreground mb-6 leading-relaxed">
            NASA's Kepler and TESS (Transiting Exoplanet Survey Satellite) missions have 
            revolutionized our understanding of planetary systems. These space telescopes 
            continuously monitor thousands of stars, measuring tiny dips in brightness 
            that indicate a planet passing in front of its host star.
          </p>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2 text-primary">Kepler Mission</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• 200,000+ target stars monitored</li>
                <li>• 4,000+ confirmed exoplanets</li>
                <li>• 30-minute observation cadence</li>
                <li>• 9+ years of continuous data</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2 text-secondary">TESS Mission</h3>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• 200,000+ bright stars surveyed</li>
                <li>• All-sky survey coverage</li>
                <li>• 2-minute observation cadence</li>
                <li>• Ongoing data collection</li>
              </ul>
            </div>
          </div>
        </Card>

        {/* The Problem */}
        <Card className="card-cosmic p-8 mb-8">
          <div className="flex items-center mb-6">
            <AlertTriangle className="h-8 w-8 text-accent mr-3" />
            <h2 className="text-2xl font-bold">The Detection Challenge</h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 mb-6">
            <div className="text-center">
              <Database className="h-12 w-12 text-primary mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Massive Scale</h3>
              <p className="text-sm text-muted-foreground">
                Millions of light curves requiring analysis
              </p>
            </div>
            <div className="text-center">
              <Clock className="h-12 w-12 text-secondary mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Time Intensive</h3>
              <p className="text-sm text-muted-foreground">
                Manual inspection takes experts hours per star
              </p>
            </div>
            <div className="text-center">
              <TrendingUp className="h-12 w-12 text-accent mx-auto mb-3" />
              <h3 className="font-semibold mb-2">Subtle Signals</h3>
              <p className="text-sm text-muted-foreground">
                Transit depths often less than 1% brightness change
              </p>
            </div>
          </div>

          <div className="bg-muted/30 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">Key Challenges:</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Distinguishing planetary transits from stellar variability</li>
              <li>• Handling instrumental noise and systematics</li>
              <li>• Identifying periodic signals in complex time series</li>
              <li>• Minimizing false positives from eclipsing binaries</li>
            </ul>
          </div>
        </Card>

        {/* Why AI/ML */}
        <Card className="card-cosmic p-8">
          <div className="flex items-center mb-6">
            <CheckCircle className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">Why AI/ML is Essential</h2>
          </div>
          
          <p className="text-muted-foreground mb-6 leading-relaxed">
            Machine learning offers unprecedented capabilities for processing vast 
            astronomical datasets and identifying patterns that might escape human analysis.
          </p>

          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold mb-4 text-primary">Traditional Methods</h3>
              <div className="space-y-3">
                <div className="flex items-center">
                  <Badge variant="destructive" className="mr-2">Slow</Badge>
                  <span className="text-sm">Manual visual inspection</span>
                </div>
                <div className="flex items-center">
                  <Badge variant="destructive" className="mr-2">Limited</Badge>
                  <span className="text-sm">Simple threshold algorithms</span>
                </div>
                <div className="flex items-center">
                  <Badge variant="destructive" className="mr-2">Biased</Badge>
                  <span className="text-sm">Human pattern recognition limits</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold mb-4 text-secondary">AI/ML Advantages</h3>
              <div className="space-y-3">
                <div className="flex items-center">
                  <Badge className="mr-2 bg-primary">Fast</Badge>
                  <span className="text-sm">Process thousands of light curves per minute</span>
                </div>
                <div className="flex items-center">
                  <Badge className="mr-2 bg-secondary">Scalable</Badge>
                  <span className="text-sm">Handles complex, multi-dimensional features</span>
                </div>
                <div className="flex items-center">
                  <Badge className="mr-2 bg-accent">Objective</Badge>
                  <span className="text-sm">Consistent, reproducible classifications</span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
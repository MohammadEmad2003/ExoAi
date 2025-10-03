import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Github, 
  ExternalLink, 
  Download, 
  BookOpen,
  Database,
  FileText,
  Code,
  Video,
  Users
} from "lucide-react";

const repositories = [
  {
    name: "exoplanet-ml-pipeline",
    description: "Complete machine learning pipeline for exoplanet detection using Kepler and TESS data",
    language: "Python",
    stars: 1247,
    url: "https://github.com/exoplanet-ai/ml-pipeline",
    topics: ["machine-learning", "astronomy", "exoplanets", "tensorflow"]
  },
  {
    name: "light-curve-preprocessing",
    description: "Data preprocessing tools for astronomical time series analysis",
    language: "Python",
    stars: 623,
    url: "https://github.com/exoplanet-ai/preprocessing",
    topics: ["data-science", "astronomy", "time-series"]
  },
  {
    name: "exoplanet-web-demo",
    description: "Interactive web application for exploring model predictions",
    language: "TypeScript",
    stars: 389,
    url: "https://github.com/exoplanet-ai/web-demo",
    topics: ["react", "visualization", "demo"]
  }
];

const datasets = [
  {
    name: "NASA Exoplanet Archive",
    description: "Comprehensive database of confirmed exoplanets and candidates",
    size: "~50 GB",
    format: ["FITS", "CSV", "JSON"],
    url: "https://exoplanetarchive.ipac.caltech.edu/",
    type: "Official"
  },
  {
    name: "Kepler DR25",
    description: "Final Kepler mission data release with light curves and planet candidates",
    size: "~2 TB",
    format: ["FITS"],
    url: "https://archive.stsci.edu/kepler/",
    type: "Mission Data"
  },
  {
    name: "TESS Data Products",
    description: "Transiting Exoplanet Survey Satellite observations and derived products",
    size: "~10 TB",
    format: ["FITS", "HDF5"],
    url: "https://archive.stsci.edu/tess/",
    type: "Mission Data"
  },
  {
    name: "Processed Training Set",
    description: "Pre-processed and labeled dataset used for model training",
    size: "1.2 GB",
    format: ["HDF5", "NPZ"],
    url: "/downloads/training-data.zip",
    type: "Derived"
  }
];

const documentation = [
  {
    title: "Technical Paper",
    description: "Deep Learning for Exoplanet Discovery: A Comprehensive Approach",
    type: "Research Paper",
    icon: FileText,
    url: "/papers/exoplanet-ml-2024.pdf"
  },
  {
    title: "Model Documentation",
    description: "Detailed documentation of neural network architecture and training process",
    type: "Technical Docs",
    icon: BookOpen,
    url: "/docs/model-documentation.html"
  },
  {
    title: "API Reference",
    description: "Complete reference for the exoplanet detection API endpoints",
    type: "API Docs",
    icon: Code,
    url: "/docs/api-reference.html"
  },
  {
    title: "Tutorial Notebooks",
    description: "Jupyter notebooks demonstrating usage and reproducing results",
    type: "Tutorials",
    icon: BookOpen,
    url: "https://github.com/exoplanet-ai/tutorials"
  }
];

const videos = [
  {
    title: "Project Overview & Demo",
    description: "15-minute presentation covering methodology and results",
    duration: "15:32",
    url: "https://youtube.com/watch?v=demo-overview"
  },
  {
    title: "Technical Deep Dive",
    description: "Detailed explanation of the neural network architecture",
    duration: "28:47",
    url: "https://youtube.com/watch?v=technical-dive"
  },
  {
    title: "Data Preprocessing Workshop",
    description: "Step-by-step guide to preparing astronomical time series data",
    duration: "42:15",
    url: "https://youtube.com/watch?v=preprocessing-workshop"
  }
];

export default function Resources() {
  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            Resources
          </h1>
          <p className="text-xl text-muted-foreground">
            Links to GitHub repositories, datasets, and documentation
          </p>
        </div>

        {/* GitHub Repositories */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <Github className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">GitHub Repositories</h2>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-6">
            {repositories.map((repo, index) => (
              <Card key={index} className="card-cosmic p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold mb-2">{repo.name}</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      {repo.description}
                    </p>
                  </div>
                  <Button variant="outline" size="sm" asChild>
                    <a href={repo.url} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </Button>
                </div>
                
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                    <span className="flex items-center">
                      <div className="w-3 h-3 bg-primary rounded-full mr-1"></div>
                      {repo.language}
                    </span>
                    <span className="flex items-center">
                      ⭐ {repo.stars}
                    </span>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-1">
                  {repo.topics.map((topic, topicIndex) => (
                    <Badge key={topicIndex} variant="secondary" className="text-xs">
                      {topic}
                    </Badge>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Datasets */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <Database className="h-8 w-8 text-secondary mr-3" />
            <h2 className="text-2xl font-bold">Datasets</h2>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-6">
            {datasets.map((dataset, index) => (
              <Card key={index} className="card-cosmic p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center mb-2">
                      <h3 className="text-lg font-semibold mr-2">{dataset.name}</h3>
                      <Badge variant="outline" className="text-xs">
                        {dataset.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      {dataset.description}
                    </p>
                  </div>
                </div>
                
                <div className="space-y-2 mb-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Size:</span>
                    <span className="font-medium">{dataset.size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Formats:</span>
                    <span className="font-medium">{dataset.format.join(", ")}</span>
                  </div>
                </div>
                
                <Button variant="outline" size="sm" className="w-full" asChild>
                  <a href={dataset.url} target="_blank" rel="noopener noreferrer">
                    <Download className="h-4 w-4 mr-2" />
                    Access Dataset
                  </a>
                </Button>
              </Card>
            ))}
          </div>
        </section>

        {/* Documentation */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <BookOpen className="h-8 w-8 text-accent mr-3" />
            <h2 className="text-2xl font-bold">Documentation</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            {documentation.map((doc, index) => {
              const Icon = doc.icon;
              return (
                <Card key={index} className="card-cosmic p-6">
                  <div className="flex items-start space-x-4">
                    <div className="p-3 bg-accent/20 rounded-lg">
                      <Icon className="h-6 w-6 text-accent" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-semibold">{doc.title}</h3>
                        <Badge variant="outline" className="text-xs">
                          {doc.type}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-4">
                        {doc.description}
                      </p>
                      <Button variant="outline" size="sm" asChild>
                        <a href={doc.url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="h-4 w-4 mr-2" />
                          View
                        </a>
                      </Button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        </section>

        {/* Video Resources */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <Video className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">Video Resources</h2>
          </div>
          
          <div className="grid lg:grid-cols-3 gap-6">
            {videos.map((video, index) => (
              <Card key={index} className="card-cosmic p-6">
                <div className="aspect-video bg-muted/30 rounded-lg mb-4 flex items-center justify-center">
                  <Video className="h-12 w-12 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{video.title}</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  {video.description}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    Duration: {video.duration}
                  </span>
                  <Button variant="outline" size="sm" asChild>
                    <a href={video.url} target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Community */}
        <section>
          <Card className="card-cosmic p-8 text-center">
            <Users className="h-12 w-12 text-primary mx-auto mb-4" />
            <h2 className="text-2xl font-bold mb-4 text-gradient-nebula">
              Join Our Community
            </h2>
            <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
              Connect with other researchers and developers working on AI applications 
              in astronomy. Share insights, ask questions, and collaborate on new projects.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button className="btn-stellar">
                <Users className="h-4 w-4 mr-2" />
                Join Slack Community
              </Button>
              <Button variant="outline">
                <Github className="h-4 w-4 mr-2" />
                GitHub Discussions
              </Button>
            </div>
          </Card>
        </section>
      </div>
    </div>
  );
}
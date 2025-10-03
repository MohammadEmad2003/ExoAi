import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Github, 
  Linkedin, 
  Mail, 
  ExternalLink,
  GraduationCap,
  Award,
  Users
} from "lucide-react";

const teamMembers = [
  {
    name: "Ahmed Samir",
    role: "Project Lead & ML Engineer",
    institution: "Cairo University",
    expertise: ["Deep Learning", "Quantum Computing", "Adversarial Machine Learning"],
    bio: "2+ years experience in machine learning applications.",
    image: "src/pages/myImage_1to1_400px.JPG",
    links: {
      github: "https://github.com/ahmed-samir11",
      linkedin: "https://linkedin.com/in/ahmed-samir-fcai-cu",
      email: "ahmedsamir1598@gmail.com"
    }
  },
  {
    name: "Dr. Michael Rodriguez",
    role: "Data Scientist",
    institution: "NASA Ames Research Center",
    expertise: ["Time Series Analysis", "Statistical Modeling", "Kepler Data"],
    bio: "NASA researcher specializing in Kepler and TESS data processing. Expert in astronomical time series analysis with focus on transit photometry.",
    image: "/api/placeholder/150/150",
    links: {
      github: "https://github.com/m-rodriguez",
      linkedin: "https://linkedin.com/in/michael-rodriguez-nasa",
      email: "michael.rodriguez@nasa.gov"
    }
  },
  {
    name: "Alex Kim",
    role: "Software Engineer",
    institution: "UC Berkeley",
    expertise: ["Python", "TensorFlow", "Data Pipelines", "Web Development"],
    bio: "MS Computer Science student with focus on scientific computing. Experienced in building scalable ML pipelines and interactive web applications.",
    image: "/api/placeholder/150/150",
    links: {
      github: "https://github.com/alex-kim-dev",
      linkedin: "https://linkedin.com/in/alex-kim-cs",
      email: "alex.kim@berkeley.edu"
    }
  }
];

const acknowledgments = [
  {
    organization: "NASA Exoplanet Archive",
    contribution: "Kepler and TESS datasets, confirmed planet catalogs",
    logo: "🛸"
  },
  {
    organization: "Lightkurve Team",
    contribution: "Open-source tools for Kepler/TESS data analysis",
    logo: "💡"
  },
  {
    organization: "STScI MAST",
    contribution: "Data hosting and archive services",
    logo: "🔭"
  },
  {
    organization: "Astropy Community",
    contribution: "Python libraries for astronomy and astrophysics",
    logo: "🐍"
  }
];

export default function Team() {
  return (
    <div className="min-h-screen py-12 px-6">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 text-gradient-stellar">
            Our Team
          </h1>
          <p className="text-xl text-muted-foreground">
            Contributors, roles, and acknowledgments
          </p>
        </div>

        {/* Team Members */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <Users className="h-8 w-8 text-primary mr-3" />
            <h2 className="text-2xl font-bold">Core Team</h2>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <Card key={index} className="card-cosmic p-6 text-center">
                <div className="mb-4">
                  <div className="w-24 h-24 bg-gradient-to-br from-primary/20 to-secondary/20 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <GraduationCap className="h-12 w-12 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold mb-1">{member.name}</h3>
                  <p className="text-sm text-primary font-medium mb-1">{member.role}</p>
                  <p className="text-sm text-muted-foreground">{member.institution}</p>
                </div>
                
                <div className="mb-4">
                  <div className="flex flex-wrap gap-1 justify-center">
                    {member.expertise.map((skill, skillIndex) => (
                      <Badge key={skillIndex} variant="secondary" className="text-xs">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  {member.bio}
                </p>
                
                <div className="flex justify-center space-x-2">
                  <Button variant="outline" size="sm" asChild>
                    <a href={member.links.github} target="_blank" rel="noopener noreferrer">
                      <Github className="h-4 w-4" />
                    </a>
                  </Button>
                  <Button variant="outline" size="sm" asChild>
                    <a href={member.links.linkedin} target="_blank" rel="noopener noreferrer">
                      <Linkedin className="h-4 w-4" />
                    </a>
                  </Button>
                  <Button variant="outline" size="sm" asChild>
                    <a href={`mailto:${member.links.email}`}>
                      <Mail className="h-4 w-4" />
                    </a>
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Acknowledgments */}
        <section className="mb-16">
          <div className="flex items-center mb-8">
            <Award className="h-8 w-8 text-secondary mr-3" />
            <h2 className="text-2xl font-bold">Acknowledgments</h2>
          </div>
          
          <Card className="card-cosmic p-8">
            <p className="text-muted-foreground mb-6 leading-relaxed">
              This project would not have been possible without the contributions of the 
              broader astronomical and open-source communities. We gratefully acknowledge 
              the following organizations and their invaluable resources:
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              {acknowledgments.map((ack, index) => (
                <div key={index} className="flex items-start space-x-4 p-4 bg-muted/30 rounded-lg">
                  <div className="text-2xl">{ack.logo}</div>
                  <div className="flex-1">
                    <h4 className="font-semibold mb-1">{ack.organization}</h4>
                    <p className="text-sm text-muted-foreground">{ack.contribution}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </section>

        {/* Collaboration */}
        <section>
          <Card className="card-cosmic p-8 text-center">
            <h2 className="text-2xl font-bold mb-4 text-gradient-nebula">
              Interested in Collaborating?
            </h2>
            <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
              We welcome collaborations with researchers, institutions, and organizations 
              interested in advancing AI applications in astronomy. Whether you have datasets, 
              expertise, or computational resources to contribute, we'd love to hear from you.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button className="btn-stellar">
                <Mail className="h-4 w-4 mr-2" />
                Contact Us
              </Button>
              <Button variant="outline">
                <ExternalLink className="h-4 w-4 mr-2" />
                Join Our Slack
              </Button>
            </div>
          </Card>
        </section>
      </div>
    </div>
  );
}
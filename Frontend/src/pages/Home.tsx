import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, Telescope, Brain, Database, Target } from "lucide-react";
import { Link } from "react-router-dom";
import { AnimatedExoplanets } from "@/components/AnimatedExoplanets";
import { useTranslation } from 'react-i18next';

export default function Home() {
  const { t } = useTranslation();
  
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-24 px-6">
        <AnimatedExoplanets />
        <div className="container mx-auto text-center relative z-10">
          <div className="cosmic-float inline-block mb-6">
            <Telescope className="h-20 w-20 text-primary mx-auto stellar-pulse" />
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="text-gradient-stellar">{t('home.hero.title1')}</span>
            <br />
            <span className="text-gradient-nebula">{t('home.hero.title2')}</span>
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            {t('home.hero.description')}
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/demo">
              <Button className="btn-stellar">
                {t('home.hero.tryModel')}
                <ArrowRight className="ml-2 h-4 w-4 icon-mirror" />
              </Button>
            </Link>
            <Link to="/approach">
              <Button variant="outline" className="border-primary/30 hover:bg-primary/10">
                {t('home.hero.learnMore')}
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12 text-gradient-stellar">
            {t('home.highlights.title')}
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="card-cosmic p-6 text-center">
              <Database className="h-12 w-12 text-primary mx-auto mb-4 nebula-glow" />
              <h3 className="text-xl font-semibold mb-3">{t('home.highlights.datasets.title')}</h3>
              <p className="text-muted-foreground">
                {t('home.highlights.datasets.description')}
              </p>
            </Card>

            <Card className="card-cosmic p-6 text-center">
              <Brain className="h-12 w-12 text-secondary mx-auto mb-4 cosmic-float" />
              <h3 className="text-xl font-semibold mb-3">{t('home.highlights.deepLearning.title')}</h3>
              <p className="text-muted-foreground">
                {t('home.highlights.deepLearning.description')}
              </p>
            </Card>

            <Card className="card-cosmic p-6 text-center">
              <Target className="h-12 w-12 text-accent mx-auto mb-4 stellar-pulse" />
              <h3 className="text-xl font-semibold mb-3">{t('home.highlights.accuracy.title')}</h3>
              <p className="text-muted-foreground">
                {t('home.highlights.accuracy.description')}
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Challenge Context */}
      <section className="py-20 px-6 bg-card/30">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-8 text-gradient-nebula">
            {t('home.challenge.title')}
          </h2>
          <p className="text-lg text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            {t('home.challenge.description')}
          </p>
          <div className="mt-8">
            <Link to="/challenge">
              <Button className="btn-cosmic">
                {t('home.challenge.explore')}
                <ArrowRight className="ml-2 h-4 w-4 icon-mirror" />
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
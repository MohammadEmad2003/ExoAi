import { Routes, Route } from 'react-router-dom';
import { Navigation } from '@/components/Navigation';
import Home from './Home';
import Demo from './Demo';
import DataUpload from './DataUpload';
import Onboarding from './Onboarding';
import ResearchHub from './ResearchHub';
import Quantum from './Quantum';
import Adversarial from './Adversarial';
import Visualizations from './Visualizations';
import Team from './Team';
import Resources from './Resources';
import NotFound from './NotFound';

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/research" element={<ResearchHub />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/upload" element={<DataUpload />} />
        <Route path="/quantum" element={<Quantum />} />
        <Route path="/adversarial" element={<Adversarial />} />
        <Route path="/visualizations" element={<Visualizations />} />
        <Route path="/team" element={<Team />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </div>
  );
};

export default Index;

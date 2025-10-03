import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import Home from "./pages/Home";
import Onboarding from "./pages/Onboarding";
import ResearchHub from "./pages/ResearchHub";
import Challenge from "./pages/Challenge";
import Approach from "./pages/Approach";
import Demo from "./pages/Demo";
import DataUpload from "./pages/DataUpload";
import Quantum from "./pages/Quantum";
import Adversarial from "./pages/Adversarial";
import Visualizations from "./pages/Visualizations";
import Team from "./pages/Team";
import Resources from "./pages/Resources";
import NotFound from "./pages/NotFound";

// Initialize i18n
import './i18n';

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="onboarding" element={<Onboarding />} />
            <Route path="research" element={<ResearchHub />} />
            <Route path="challenge" element={<Challenge />} />
            <Route path="approach" element={<Approach />} />
            <Route path="demo" element={<Demo />} />
            <Route path="upload" element={<DataUpload />} />
            <Route path="quantum" element={<Quantum />} />
            <Route path="adversarial" element={<Adversarial />} />
            <Route path="visualizations" element={<Visualizations />} />
            <Route path="team" element={<Team />} />
            <Route path="resources" element={<Resources />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

export function AnimatedExoplanets() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Animated exoplanets with orbital motion */}
      <div className="exoplanet exoplanet-1" />
      <div className="exoplanet exoplanet-2" />
      <div className="exoplanet exoplanet-3" />
      <div className="exoplanet exoplanet-4" />
      
      {/* Additional floating cosmic elements */}
      <div className="absolute top-10 right-10 w-2 h-2 bg-primary rounded-full stellar-pulse opacity-60" />
      <div className="absolute bottom-20 left-20 w-1.5 h-1.5 bg-secondary rounded-full nebula-glow opacity-40" />
      <div className="absolute top-1/3 left-1/4 w-3 h-3 bg-accent rounded-full cosmic-float opacity-50" />
      <div className="absolute bottom-1/3 right-1/3 w-2.5 h-2.5 bg-primary rounded-full stellar-pulse opacity-45" />
    </div>
  );
}
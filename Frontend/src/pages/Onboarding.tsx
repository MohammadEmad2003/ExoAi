import React, { useMemo, useState, useRef, useEffect } from "react";
import * as THREE from "three";
import { useTranslation } from "react-i18next";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  OrbitControls,
  PointerLockControls,
  Html,
  useGLTF,
} from "@react-three/drei";
import {
  ArrowRight,
  ArrowLeft,
  Upload,
  Settings,
  Play,
  Download,
  BookOpen,
  Target,
  Zap,
  Shield,
  CheckCircle,
  Sparkles,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import DataUploader from "@/components/DataUploader";
import HyperparamPanel from "@/components/HyperparamPanel";
import ModelCard from "@/components/ModelCard";

const STEPS = [
  {
    id: "welcome",
    title: "Welcome to Cosmic Analysts ExoAI",
    description: "Your guided journey to advanced machine learning",
    icon: Sparkles,
  },
  {
    id: "upload",
    title: "Upload Your Data",
    description: "Start by uploading your dataset for analysis",
    icon: Upload,
  },
  {
    id: "configure",
    title: "Configure Training",
    description: "Adjust model parameters with guided recommendations",
    icon: Settings,
  },
  {
    id: "train",
    title: "Train Your Model",
    description: "Watch your model learn from the data",
    icon: Play,
  },
  {
    id: "evaluate",
    title: "Evaluate Results",
    description: "Review performance and robustness metrics",
    icon: Target,
  },
  {
    id: "deploy",
    title: "Export & Deploy",
    description: "Get your production-ready model",
    icon: Download,
  },
];

const DEMO_DATASETS = [
  {
    name: "Exoplanet Classification",
    description:
      "Classify exoplanets based on stellar and orbital characteristics",
    size: "2,847 samples",
    features: 12,
    target: "planet_type",
    difficulty: "Beginner",
  },
  {
    name: "Stellar Properties",
    description: "Predict stellar mass and luminosity from observational data",
    size: "5,234 samples",
    features: 18,
    target: "stellar_mass",
    difficulty: "Intermediate",
  },
  {
    name: "Galaxy Classification",
    description: "Classify galaxies into morphological types",
    size: "12,456 samples",
    features: 24,
    target: "galaxy_type",
    difficulty: "Advanced",
  },
];

// Mock data for demonstration
const MOCK_MODEL_INFO = {
  name: "ExoAI-TabKANet",
  version: "1.0.0",
  architecture: "TabKANet (KAN + Transformer)",
  created_at: "2024-01-15 14:30:00",
  training_time: "12m 34s",
  dataset_size: 2847,
  parameters: 125000,
  framework: "PyTorch",
};

const MOCK_METRICS = {
  accuracy: 0.924,
  precision: 0.918,
  recall: 0.912,
  f1_score: 0.915,
  auc_roc: 0.967,
  loss: 0.234,
  val_loss: 0.267,
};

const MOCK_ROBUSTNESS = {
  adversarial_accuracy: 0.856,
  calibration_error: 0.123,
  uncertainty_score: 0.089,
  attack_success_rate: 0.144,
};

const MOCK_PERFORMANCE = {
  inference_time_ms: 2.3,
  throughput_samples_sec: 4347,
  memory_usage_mb: 156,
  model_size_mb: 12.4,
};

export default function Onboarding() {
  /**
   * Onboarding page
   * - Shows interactive 3D models for Exoplanet (Exo) and Kepler
   * - Free Move: enables pointer lock and keyboard movement (Arrow keys or WASD).
   *   * How to use Free Move:
   *     1. Click "Enter Free Move" (button at top-left of the Exoplanet viewer).
   *     2. Click inside the canvas to lock the pointer / enable camera look.
   *     3. Move with Arrow keys or WASD. Move speed is modest; adjust in code if needed.
   *     4. Press ESC to release pointer lock and stop looking around.
   * - Kepler viewer uses orbit controls for rotate/zoom and includes a short info panel.
   */
  const { t, i18n } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [uploadedData, setUploadedData] = useState(null);
  const [freeMove, setFreeMove] = useState(false);
  const [pointerLocked, setPointerLocked] = useState(false);
  const modelsRef = useRef<HTMLDivElement | null>(null);
  const exoRef = useRef<HTMLDivElement | null>(null);
  const stepIntoRef = useRef<HTMLDivElement | null>(null);
  const userGuideRef = useRef<HTMLDivElement | null>(null);

  // small helper to scroll an element into view with an offset so the element appears slightly below the top
  const scrollToWithOffset = (
    el: HTMLElement | null | undefined,
    offset = 96
  ) => {
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top, behavior: "smooth" });
  };

  // 3D model component (use same logic as src/3DModel.tsx)
  const Model: React.FC<{ src?: string; label?: string }> = ({ src }) => {
    const gltf = useGLTF(src || "/exo.glb") as any;
    useEffect(() => {
      if (!gltf?.scene) return;
      gltf.scene.traverse((child: any) => {
        if (child.isMesh && child.material) {
          const materials = Array.isArray(child.material)
            ? child.material
            : [child.material];
          materials.forEach((mat: any) => {
            try {
              // Only set properties when they exist on the material to avoid
              // breaking materials provided by glTF extensions or custom shaders.
              if (mat == null) return;
              if (mat.isMaterial) {
                if ("emissive" in mat) mat.emissive = new THREE.Color(0x000000);
                if ("roughness" in mat) mat.roughness = 0.5;
                if ("metalness" in mat) mat.metalness = 0.3;
                // Ensure three updates internal program/uniforms
                mat.needsUpdate = true;
              }
            } catch (err) {
              // Don't crash the entire scene if a material can't be mutated
              // (some glTF materials come as NodeMaterials or custom shader materials)
              // Log for debugging in development only.
              // eslint-disable-next-line no-console
              console.warn("skipped material mutation for a glTF mesh:", err);
            }
          });
        }
      });
    }, [gltf]);

    return gltf?.scene ? (
      <primitive object={gltf.scene} scale={[1.5, 1.5, 1.5]} />
    ) : null;
  };

  // FreeMove controls (WASD + Arrow keys) copied from 3DModel.tsx
  const FreeMoveControls: React.FC<{ enabled: boolean }> = ({ enabled }) => {
    const { camera } = useThree();
    const velocity = useRef(new THREE.Vector3());
    const direction = useRef(new THREE.Vector3());
    const moveForward = useRef(false);
    const moveBackward = useRef(false);
    const moveLeft = useRef(false);
    const moveRight = useRef(false);
    const enabledRef = useRef(enabled);

    useEffect(() => {
      enabledRef.current = enabled;
    }, [enabled]);

    useEffect(() => {
      const handleKeyDown = (event: KeyboardEvent) => {
        if (!enabledRef.current) return;
        switch (event.code) {
          case "ArrowUp":
          case "KeyW":
            event.preventDefault();
            moveForward.current = true;
            break;
          case "ArrowLeft":
          case "KeyA":
            event.preventDefault();
            moveLeft.current = true;
            break;
          case "ArrowDown":
          case "KeyS":
            event.preventDefault();
            moveBackward.current = true;
            break;
          case "ArrowRight":
          case "KeyD":
            event.preventDefault();
            moveRight.current = true;
            break;
        }
      };

      const handleKeyUp = (event: KeyboardEvent) => {
        if (!enabledRef.current) return;
        switch (event.code) {
          case "ArrowUp":
          case "KeyW":
            moveForward.current = false;
            break;
          case "ArrowLeft":
          case "KeyA":
            moveLeft.current = false;
            break;
          case "ArrowDown":
          case "KeyS":
            moveBackward.current = false;
            break;
          case "ArrowRight":
          case "KeyD":
            moveRight.current = false;
            break;
        }
      };

      document.addEventListener("keydown", handleKeyDown);
      document.addEventListener("keyup", handleKeyUp);

      return () => {
        document.removeEventListener("keydown", handleKeyDown);
        document.removeEventListener("keyup", handleKeyUp);
      };
    }, []);

    useFrame((_, delta) => {
      if (!enabled) return;

      velocity.current.set(0, 0, 0);
      const speed = 0.5;

      if (moveForward.current) velocity.current.z -= speed * delta;
      if (moveBackward.current) velocity.current.z += speed * delta;
      if (moveLeft.current) velocity.current.x -= speed * delta;
      if (moveRight.current) velocity.current.x += speed * delta;

      direction.current.copy(velocity.current).applyEuler(camera.rotation);
      camera.position.add(direction.current);
    });

    return enabled ? <PointerLockControls /> : null;
  };

  const currentStepData = STEPS[currentStep];
  const progress = ((currentStep + 1) / STEPS.length) * 100;

  // Scroll to models when entering the welcome (guided) step
  useEffect(() => {
    if (currentStepData.id === "welcome") {
      // small delay to allow layout transition
      setTimeout(() => {
        scrollToWithOffset(modelsRef.current || undefined, 96);
      }, 200);
    }
  }, [currentStepData.id]);

  // When entering Free Move, scroll the Exo viewer into the center of the viewport
  useEffect(() => {
    if (freeMove) {
      setTimeout(() => {
        scrollToWithOffset(exoRef.current || undefined, 120);
      }, 180);
    }
  }, [freeMove]);

  // Track pointer lock state so we can show a click-to-start overlay
  useEffect(() => {
    const handlePointerLockChange = () => {
      setPointerLocked(!!document.pointerLockElement);
    };

    document.addEventListener("pointerlockchange", handlePointerLockChange);
    document.addEventListener("pointerlockerror", handlePointerLockChange);

    return () => {
      document.removeEventListener(
        "pointerlockchange",
        handlePointerLockChange
      );
      document.removeEventListener("pointerlockerror", handlePointerLockChange);
    };
  }, []);

  const nextStep = () => {
    if (currentStep < STEPS.length - 1) {
      setCompletedSteps((prev) => [...prev, currentStep]);
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setTrainingProgress(0);

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          setCompletedSteps((prev) => [...prev, currentStep]);
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  const selectDemoDataset = (datasetName: string) => {
    setSelectedDataset(datasetName);
    // Simulate data loading
    setTimeout(() => {
      setUploadedData(true as any);
      nextStep();
    }, 1000);
  };

  const renderStepContent = () => {
    switch (currentStepData.id) {
      case "welcome": {
        return (
          <div className="space-y-10">
            <div className="text-center">
              <h1 className="text-3xl md:text-4xl font-bold mb-2 text-foreground mx-auto">
                {t("onboarding.welcome.heading", "Guided Onboarding")}
              </h1>
              <p className="text-base md:text-lg text-foreground/80 max-w-2xl mx-auto">
                {t(
                  "onboarding.welcome.subtitle",
                  "Explore the Kepler and Exo models interactively."
                )}
              </p>
            </div>

            {/* Space header and gamification banner placed above both model viewers */}
            <div className="mx-auto w-full max-w-5xl transition-all duration-500">
              <div className="relative rounded-xl overflow-hidden bg-gradient-to-r from-slate-900 via-[#050816] to-sky-900 p-6 text-white mb-6 border">
                {/* decorative stars/planet */}
                <svg
                  className="absolute inset-0 w-full h-full opacity-20 -z-10"
                  preserveAspectRatio="none"
                  viewBox="0 0 800 200"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <radialGradient id="g" cx="50%" cy="50%">
                      <stop offset="0%" stopColor="#ffd" stopOpacity="0.1" />
                      <stop offset="100%" stopColor="#fff" stopOpacity="0" />
                    </radialGradient>
                  </defs>
                  <rect width="800" height="200" fill="url(#g)" />
                  <g fill="#fff">
                    <circle cx="40" cy="30" r="1.6" />
                    <circle cx="120" cy="50" r="1" />
                    <circle cx="220" cy="20" r="1.2" />
                    <circle cx="320" cy="70" r="0.9" />
                    <circle cx="520" cy="30" r="1.4" />
                    <circle cx="700" cy="60" r="1" />
                  </g>
                  <g transform="translate(640,36)">
                    <circle
                      cx="0"
                      cy="0"
                      r="24"
                      fill="#7dd3fc"
                      opacity="0.06"
                    />
                    <circle
                      cx="0"
                      cy="0"
                      r="12"
                      fill="#60a5fa"
                      opacity="0.12"
                    />
                  </g>
                </svg>

                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <h1
                        className="text-3xl md:text-4xl font-extrabold tracking-tight bg-clip-text text-transparent"
                        style={{
                          backgroundImage:
                            "linear-gradient(90deg, #06b6d4, rgba(255,255,255,0.5))",
                        }}
                      >
                        {t(
                          "onboarding.banner.title",
                          "Step Into Our 3D Universe"
                        )}
                      </h1>
                      {/* Dev-only: show active i18n language to help debug missing translations */}
                      {process.env.NODE_ENV === "development" && (
                        <span className="text-sm text-white/60">
                          lang: {i18n.language}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-white/80 max-w-2xl">
                      {t(
                        "onboarding.banner.description",
                        'Fly around exoplanets, discover alien worlds, and uncover hidden patterns — a playful, gamified experience that makes exploring astronomy and ML data feel like an adventure. Click "Enter Free Move" to pilot the camera yourself and roam the system.'
                      )}
                    </p>
                    <div className="mt-3 flex items-center gap-2">
                      <Badge variant="secondary">
                        {t(
                          "onboarding.banner.badges.interactive",
                          "Interactive"
                        )}
                      </Badge>
                      <Badge variant="outline">
                        {t("onboarding.banner.badges.immersive", "Immersive")}
                      </Badge>
                      <Badge variant="outline">
                        {t(
                          "onboarding.banner.badges.learnByPlaying",
                          "Learn-by-Playing"
                        )}
                      </Badge>
                    </div>
                  </div>

                  <div className="flex-shrink-0 flex items-center gap-2">
                    <Button
                      onClick={() => setFreeMove((v) => !v)}
                      className="px-4 py-2 transition-transform hover:scale-105"
                    >
                      {freeMove
                        ? t("onboarding.banner.exitFreeMove", "Exit Free Move")
                        : t(
                            "onboarding.banner.enterFreeMove",
                            "Enter Free Move"
                          )}
                    </Button>
                    <Button
                      variant="outline"
                      className="px-3 py-2"
                      onClick={() =>
                        scrollToWithOffset(
                          userGuideRef.current || undefined,
                          96
                        )
                      }
                    >
                      {t("onboarding.banner.viewGuide", "View Guide")}
                    </Button>
                  </div>
                </div>
              </div>

              {/* Models grid */}
              <div
                ref={modelsRef}
                className={`mx-auto transition-all duration-500 ${
                  freeMove
                    ? "w-full h-[90vh]"
                    : "grid grid-cols-1 md:grid-cols-2 gap-6"
                }`}
              >
                {/* Exo model (fullscreen when freeMove enabled) */}
                <div
                  ref={exoRef}
                  className={`relative rounded-xl bg-black/70 border overflow-hidden ${
                    freeMove ? "h-full col-span-2" : "h-[420px]"
                  }`}
                >
                  <h2 className="text-white text-lg font-semibold p-3">
                    {t("onboarding.models.exoplanet", "Exoplanet")}
                  </h2>

                  {/* Exit button shown when exo fills the screen (freeMove active) */}
                  {freeMove && (
                    <div className="absolute top-3 right-3 z-30">
                      <button
                        className="px-2 py-1 text-xs rounded-md border bg-background/90 hover:bg-primary transition-colors"
                        onClick={() => setFreeMove(false)}
                      >
                        {t("onboarding.banner.exitFreeMove", "Exit Free Move")}
                      </button>
                    </div>
                  )}

                  <Canvas camera={{ position: [0, 3.5, 0.8], fov: 50 }}>
                    <color attach="background" args={["#090b12"]} />
                    <ambientLight intensity={0.9} />
                    <directionalLight position={[3, 5, 2]} intensity={1.4} />
                    <group position={[0, -0.5, 0]}>
                      <Model src="/exo.glb" />
                    </group>
                    {!freeMove && (
                      <OrbitControls
                        enableDamping
                        enableZoom
                        target={[0, -0.5, 0]}
                      />
                    )}
                    <FreeMoveControls enabled={freeMove} />
                  </Canvas>

                  {/* Click-to-start overlay when Free Move is active but pointer lock not yet engaged */}
                  {freeMove && !pointerLocked && (
                    <div
                      onClick={() => {
                        // Try to request pointer lock on the canvas inside this exoRef
                        const canvas = exoRef.current?.querySelector(
                          "canvas"
                        ) as HTMLCanvasElement | null;
                        if (canvas && canvas.requestPointerLock) {
                          canvas.requestPointerLock();
                        }
                      }}
                      className="absolute inset-0 z-40 flex items-center justify-center cursor-pointer"
                    >
                      <div className="bg-black/60 rounded-md px-4 py-2 text-center text-white">
                        {t(
                          "onboarding.banner.clickToStart",
                          "Click on the screen to start"
                        )}
                        <div className="text-xs text-white/70 mt-1">
                          {t(
                            "onboarding.banner.pressEscToExit",
                            "Press ESC to exit"
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Kepler model (only visible when not in free move) */}
                {!freeMove && (
                  <div className="rounded-xl bg-black/70 border overflow-hidden relative">
                    <h2 className="text-white text-lg font-semibold p-3">
                      {t("onboarding.models.kepler", "Kepler")}
                    </h2>

                    {/* Kepler Canvas (kept compact so documentation can sit below) */}
                    <div className="h-[260px] md:h-[300px] w-full">
                      <Canvas camera={{ position: [0, 1.4, 6], fov: 50 }}>
                        <color attach="background" args={["#090b12"]} />
                        <ambientLight intensity={0.9} />
                        <directionalLight
                          position={[3, 5, 2]}
                          intensity={1.4}
                        />
                        <group position={[0, -0.5, 0]}>
                          <Model src="/kepler.glb" />
                        </group>
                        <OrbitControls
                          enableDamping
                          enableZoom
                          target={[0, -0.5, 0]}
                        />
                      </Canvas>
                    </div>

                    {/* Kepler documentation was moved below the models grid for clearer layout */}
                  </div>
                )}
              </div>
              {/* Kepler documentation panel placed below both model viewers */}
              {!freeMove && (
                <div className="mx-auto max-w-5xl mt-6 p-6 rounded-lg bg-background/5 border text-white/90 text-left">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-lg font-semibold">
                        {t(
                          "onboarding.kepler.title",
                          "🌍 Exoplanet: Kepler-186f"
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground mt-1">
                        {t("onboarding.kepler.planetTypeLabel", "Planet Type:")}{" "}
                        <span className="font-medium">
                          {t("onboarding.kepler.planetType", "Super Earth")}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <div className="text-xs text-muted-foreground">
                        {t(
                          "onboarding.kepler.distanceLabel",
                          "Distance from Earth"
                        )}
                      </div>
                      <div className="font-medium">
                        {t(
                          "onboarding.kepler.distanceValue",
                          "~579 light years"
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">
                        {t("onboarding.kepler.textureLabel", "Texture Size")}
                      </div>
                      <div className="font-medium">
                        {t(
                          "onboarding.kepler.textureValue",
                          "512×512 (optimized)"
                        )}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">
                        {t("onboarding.kepler.verticesLabel", "Vertices Count")}
                      </div>
                      <div className="font-medium">
                        {t("onboarding.kepler.verticesValue", "304 (low-poly)")}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">
                        {t("onboarding.kepler.meshLabel", "Mesh")}
                      </div>
                      <div className="font-medium">
                        {t(
                          "onboarding.kepler.meshValue",
                          "Performance-focused"
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 text-sm text-white/85">
                    <div className="font-semibold">
                      {t(
                        "onboarding.kepler.aboutTitle",
                        "🪐 About Kepler-186f"
                      )}
                    </div>
                    <p className="mt-1 text-xs text-white/80">
                      {t(
                        "onboarding.kepler.paragraph1",
                        "Kepler-186f orbits a red dwarf in the constellation Cygnus and is notable for being the first Earth-sized planet discovered in the habitable zone of its star — the region where liquid water could exist on the surface."
                      )}
                    </p>
                    <p className="mt-2 text-xs text-white/80">
                      {t(
                        "onboarding.kepler.paragraph2",
                        "Classified as a Super Earth, Kepler-186f is larger and more massive than Earth but smaller than Neptune, likely with a rocky surface. Because its star is cooler and dimmer, the planet receives less stellar energy; if an atmosphere is present it may trap heat and moderate surface temperatures."
                      )}
                    </p>
                    <p className="mt-2 text-xs text-white/80">
                      {t(
                        "onboarding.kepler.paragraph3",
                        "The discovery of Kepler-186f marked an important milestone in the search for potentially habitable worlds and helps inform how we model exoplanet climates and detection strategies."
                      )}
                    </p>
                  </div>
                </div>
              )}
              {/* 3D Universe User Guide (below Kepler info) */}
              {!freeMove && (
                <div
                  ref={userGuideRef}
                  className="mx-auto max-w-5xl mt-6 p-6 rounded-lg bg-background/6 border text-white/90 text-left"
                >
                  <h3 className="text-lg font-semibold">
                    {t(
                      "onboarding.userGuide.title",
                      "🌌 3D Universe – User Guide"
                    )}
                  </h3>
                  <p className="mt-2 text-sm text-white/80">
                    {t(
                      "onboarding.userGuide.welcome",
                      "Welcome to the 3D Universe Explorer, an interactive environment where you can explore distant exoplanets like Kepler-186f."
                    )}
                  </p>

                  <div className="mt-4">
                    <div className="font-semibold">
                      {t(
                        "onboarding.userGuide.basicControls",
                        "🖱 Basic Controls"
                      )}
                    </div>
                    <div className="mt-2 text-sm">
                      <div className="font-semibold">Orbit Mode (default):</div>
                      <ul className="list-disc ml-5 mt-1 text-xs text-white/80">
                        <li>
                          {t(
                            "onboarding.userGuide.controls.leftMouse",
                            "Left Mouse Button (Drag): Rotate around the universe"
                          )}
                        </li>
                        <li>
                          {t(
                            "onboarding.userGuide.controls.rightMouse",
                            "Right Mouse Button (Drag): Pan the camera view"
                          )}
                        </li>
                        <li>
                          {t(
                            "onboarding.userGuide.controls.scroll",
                            "Scroll Wheel: Zoom in and out smoothly"
                          )}
                        </li>
                      </ul>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="font-semibold">
                      {t(
                        "onboarding.userGuide.freeMoveTitle",
                        "🎮 Free Move Mode (Exoplanet Universe Only)"
                      )}
                    </div>
                    <div className="mt-2 text-sm text-white/80">
                      {t(
                        "onboarding.userGuide.freeMoveDescription",
                        "When inside an exoplanet universe, you can switch to Free Move Mode for a more immersive experience. After clicking the Enter Free Move button, click the 3D model to lock pointer and begin moving."
                      )}
                    </div>
                    <div className="mt-2 text-sm">
                      <div className="font-semibold">
                        {t(
                          "onboarding.userGuide.enableDisable",
                          "Enable / Disable Free Move Mode:"
                        )}
                      </div>
                      <div className="text-xs text-white/80">
                        {t(
                          "onboarding.userGuide.enableInstructions",
                          "Click the 'Enter Free Move' button in the top-left corner. To exit, click 'Exit Free Move'."
                        )}
                      </div>
                    </div>

                    <div className="mt-2 text-sm">
                      <div className="font-semibold">
                        {t(
                          "onboarding.userGuide.movementKeys",
                          "Movement Keys:"
                        )}
                      </div>
                      <div className="text-xs text-white/80">
                        {t(
                          "onboarding.userGuide.movementKeysList",
                          "W / ↑ Arrow: Move Forward · S / ↓ Arrow: Move Backward · A / ← Arrow: Move Left · D / → Arrow: Move Right"
                        )}
                      </div>
                    </div>

                    <div className="mt-2 text-sm">
                      <div className="font-semibold">
                        {t(
                          "onboarding.userGuide.mouseMovement",
                          "Mouse Movement:"
                        )}
                      </div>
                      <div className="text-xs text-white/80">
                        {t(
                          "onboarding.userGuide.mouseMovementParagraph",
                          "Look around in any direction while moving. This mode lets you walk or fly through space and get up close to different exoplanets."
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="font-semibold">
                      {t(
                        "onboarding.userGuide.exploringExoplanets",
                        "🌍 Exploring Exoplanets"
                      )}
                    </div>
                    <div className="mt-2 text-xs text-white/80">
                      {t(
                        "onboarding.userGuide.exploringExoplanetsParagraph",
                        "Each exoplanet model is labeled with its name. You can zoom in closely to inspect the surface. In Free Move Mode, fly between planets and explore them from any angle. Additional info panels (like planet type, distance from Earth, etc.) can be displayed alongside models."
                      )}
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="font-semibold">
                      {t("onboarding.userGuide.tips.title", "⚡ Tips")}
                    </div>
                    <ul className="list-disc ml-5 mt-1 text-xs text-white/80">
                      <li>
                        {t(
                          "onboarding.userGuide.tips.startInOrbit",
                          "Start in Orbit Mode to get a full overview."
                        )}
                      </li>
                      <li>
                        {t(
                          "onboarding.userGuide.tips.useFreeMove",
                          "Use Free Move Mode to dive deeper into individual planets."
                        )}
                      </li>
                      <li>
                        {t(
                          "onboarding.userGuide.tips.viewFullscreen",
                          "For best experience, view in fullscreen and on a desktop."
                        )}
                      </li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      }

      case "upload":
        return (
          <div className="space-y-6">
            <div className="text-left">
              <h2 className="text-2xl font-bold mb-2 text-foreground">
                {t("onboarding.upload.title")}
              </h2>
              <p className="text-foreground/80">
                {t("onboarding.upload.description")}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {DEMO_DATASETS.map((dataset) => (
                <Card
                  key={dataset.name}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedDataset === dataset.name
                      ? "ring-2 ring-primary"
                      : ""
                  }`}
                  onClick={() => selectDemoDataset(dataset.name)}
                >
                  <CardHeader>
                    <CardTitle className="text-base">{dataset.name}</CardTitle>
                    <CardDescription>{dataset.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>
                          {t("onboarding.upload.samplesLabel", "Samples:")}
                        </span>
                        <span className="font-medium">{dataset.size}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>
                          {t("onboarding.upload.featuresLabel", "Features:")}
                        </span>
                        <span className="font-medium">{dataset.features}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>
                          {t("onboarding.upload.targetLabel", "Target:")}
                        </span>
                        <span className="font-medium">{dataset.target}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>
                          {t(
                            "onboarding.upload.difficultyLabel",
                            "Difficulty:"
                          )}
                        </span>
                        <Badge
                          variant={
                            dataset.difficulty === "Beginner"
                              ? "default"
                              : dataset.difficulty === "Intermediate"
                              ? "secondary"
                              : "destructive"
                          }
                        >
                          {dataset.difficulty}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  {t("onboarding.upload.orUpload")}
                </span>
              </div>
            </div>

            <DataUploader
              onDataUploaded={(data) => {
                setUploadedData(data);
                nextStep();
              }}
            />
          </div>
        );

      case "configure":
        return (
          <div className="space-y-6">
            <div className="text-left">
              <h2 className="text-2xl font-bold mb-2 text-foreground">
                {t("onboarding.configure.title")}
              </h2>
              <p className="text-foreground/80">
                {t("onboarding.configure.description")}
              </p>
            </div>

            <HyperparamPanel
              mode="guided"
              onConfigChange={(config) => {
                // Handle config changes
              }}
            />
          </div>
        );

      case "train":
        return (
          <div className="space-y-6">
            <div className="text-left">
              <h2 className="text-2xl font-bold mb-2 text-foreground">
                {t("onboarding.train.title")}
              </h2>
              <p className="text-foreground/80">
                {t("onboarding.train.description")}
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Play className="h-5 w-5" />
                  {t("onboarding.train.progressTitle")}
                </CardTitle>
                <CardDescription>
                  {isTraining
                    ? t("onboarding.train.training")
                    : t("onboarding.train.ready")}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isTraining && (
                  <>
                    <div className="flex justify-between text-sm">
                      <span>
                        {t("onboarding.train.epoch")}{" "}
                        {Math.floor(trainingProgress / 10)} / 10
                      </span>
                      <span>{trainingProgress.toFixed(0)}%</span>
                    </div>
                    <Progress value={trainingProgress} />
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">
                          {t("onboarding.train.trainingLoss")}
                        </span>
                        <span className="ml-2 font-mono">
                          {(0.8 - (trainingProgress / 100) * 0.6).toFixed(4)}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">
                          {t("onboarding.train.validationAccuracy")}
                        </span>
                        <span className="ml-2 font-mono">
                          {(0.3 + (trainingProgress / 100) * 0.6).toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </>
                )}

                {!isTraining && trainingProgress === 0 && (
                  <Button onClick={startTraining} className="w-full">
                    <Play className="h-4 w-4 mr-2" />
                    {t("onboarding.buttons.startTraining")}
                  </Button>
                )}

                {!isTraining && trainingProgress === 100 && (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>
                      {t("onboarding.train.completed")}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {trainingProgress === 100 && (
              <div className="text-left">
                <Button onClick={nextStep}>
                  {t("onboarding.buttons.viewResults")}
                  <ArrowRight className="h-4 w-4 ml-2 icon-mirror" />
                </Button>
              </div>
            )}
          </div>
        );

      case "evaluate":
        return (
          <div className="space-y-6">
            <div className="text-left">
              <h2 className="text-2xl font-bold mb-2 text-foreground">
                {t("onboarding.evaluate.title")}
              </h2>
              <p className="text-foreground/80">
                {t("onboarding.evaluate.description")}
              </p>
            </div>

            <ModelCard
              modelInfo={MOCK_MODEL_INFO}
              metrics={MOCK_METRICS}
              robustness={MOCK_ROBUSTNESS}
              performance={MOCK_PERFORMANCE}
              onExport={(format) => {
                console.log("Exporting model in format:", format);
              }}
              onShare={() => {
                console.log("Sharing model");
              }}
            />
          </div>
        );

      case "deploy":
        return (
          <div className="space-y-6 text-left">
            <div>
              <h2 className="text-2xl font-bold mb-2 text-foreground">
                {t("onboarding.deploy.title")}
              </h2>
              <p className="text-foreground/80">
                {t("onboarding.deploy.description")}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Download className="h-5 w-5" />
                    Export Model
                  </CardTitle>
                  <CardDescription>
                    Download optimized model files for production deployment
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Button className="w-full" variant="outline">
                      Download TorchScript (.pt)
                    </Button>
                    <Button className="w-full" variant="outline">
                      Download ONNX (.onnx)
                    </Button>
                    <Button className="w-full" variant="outline" disabled>
                      Download TensorRT (.plan)
                      <Badge variant="secondary" className="ml-2">
                        GPU Required
                      </Badge>
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Continue Learning
                  </CardTitle>
                  <CardDescription>
                    Explore advanced features and experiment with different
                    models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button className="w-full">Switch to Advanced Mode</Button>
                  <Button className="w-full" variant="outline">
                    Try Quantum Computing
                  </Button>
                  <Button className="w-full" variant="outline">
                    Test Adversarial Robustness
                  </Button>
                </CardContent>
              </Card>
            </div>

            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                You've completed the guided tour! Your model achieved 92.4%
                accuracy with strong robustness metrics. Ready to tackle more
                complex challenges?
              </AlertDescription>
            </Alert>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Thin top progress bar */}
      <div className="h-1 w-full bg-muted">
        <div
          className="h-1 bg-primary transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="container mx-auto px-4 py-10">
        {/* Header */}
        <div className="mb-10 text-center">
          <h2 className="text-xs uppercase tracking-widest text-primary/80 mb-2">
            {t("navigation.guidedMode")}
          </h2>
          <h1 className="text-3xl md:text-4xl font-bold text-foreground">
            {t("onboarding.title")}
          </h1>
          <p className="text-foreground/70 mt-2">{t("onboarding.subtitle")}</p>
          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">
              {t("onboarding.buttons.previous", {
                defaultValue: `Step ${currentStep + 1} of ${STEPS.length}`,
              })}
            </Badge>
            <span>•</span>
            <span>{Math.round(progress)}% Complete</span>
          </div>
        </div>

        {/* Step Navigation - centered titles row */}
        <div className="mb-8">
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap items-center justify-center gap-6">
              {STEPS.map((step, index) => (
                <div
                  key={step.id}
                  className="flex flex-col items-center text-center"
                >
                  <div
                    className={`
                      flex items-center justify-center w-12 h-12 rounded-full border-2 transition-colors mb-2
                      ${
                        index === currentStep
                          ? "border-primary bg-primary text-primary-foreground"
                          : completedSteps.includes(index)
                          ? "border-green-500 bg-green-500 text-white"
                          : "border-muted bg-background text-muted-foreground"
                      }
                    `}
                  >
                    {completedSteps.includes(index) ? (
                      <CheckCircle className="h-5 w-5" />
                    ) : (
                      <step.icon className="h-5 w-5" />
                    )}
                  </div>
                  <div className="text-xs font-medium max-w-[160px]">
                    {t(`onboarding.steps.${step.id}.title`, step.title)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step Content */}
        <div className="mb-8">
          <Card className="min-h-[500px]">
            <CardContent className="p-8">{renderStepContent()}</CardContent>
          </Card>
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={prevStep}
            disabled={currentStep === 0}
          >
            <ArrowLeft className="h-4 w-4 mr-2 icon-mirror" />
            {t("onboarding.buttons.previous")}
          </Button>

          <Button
            onClick={nextStep}
            disabled={
              currentStep === STEPS.length - 1 ||
              (currentStepData.id === "upload" &&
                !uploadedData &&
                !selectedDataset) ||
              (currentStepData.id === "train" && trainingProgress < 100)
            }
          >
            {currentStep === STEPS.length - 1
              ? t("onboarding.buttons.finish")
              : t("onboarding.buttons.next")}
            <ArrowRight className="h-4 w-4 ml-2 icon-mirror" />
          </Button>
        </div>
      </div>
    </div>
  );
}

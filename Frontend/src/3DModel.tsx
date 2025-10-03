// src/App.tsx
import React, { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, PointerLockControls } from "@react-three/drei";
import * as THREE from "three";

/* --------------------
   Model loader
   -------------------- */
function Model() {
  const gltf = useGLTF("/exo.glb") as any;
  useEffect(() => {
    if (!gltf?.scene) return;
    gltf.scene.traverse((child: any) => {
      if (child.isMesh && child.material) {
        const mat = child.material;
        mat.emissive = new THREE.Color(0x000000);
        mat.roughness = 0.5;
        mat.metalness = 0.3;
        mat.needsUpdate = true;
      }
    });
  }, [gltf]);
  return gltf?.scene ? (
    <primitive object={gltf.scene} scale={[1.5, 1.5, 1.5]} />
  ) : null;
}

/* --------------------
   FreeMoveControls (WASD + mouse look)
   -------------------- */
function FreeMoveControls({ enabled }: { enabled: boolean }) {
  const { camera } = useThree();
  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const moveForward = useRef(false);
  const moveBackward = useRef(false);
  const moveLeft = useRef(false);
  const moveRight = useRef(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      switch (event.code) {
        case "ArrowUp":
        case "KeyW":
          moveForward.current = true;
          break;
        case "ArrowLeft":
        case "KeyA":
          moveLeft.current = true;
          break;
        case "ArrowDown":
        case "KeyS":
          moveBackward.current = true;
          break;
        case "ArrowRight":
        case "KeyD":
          moveRight.current = true;
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
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
}

/* --------------------
   CameraTracker (HUD)
   -------------------- */
function CameraTracker({
  setCoords,
}: {
  setCoords: React.Dispatch<
    React.SetStateAction<{
      x: string;
      y: string;
      z: string;
      zoom: string;
      qx: string;
      qy: string;
      qz: string;
      qw: string;
    }>
  >;
}) {
  const { camera } = useThree();
  useFrame(() => {
    const { x, y, z } = camera.position;
    const { x: qx, y: qy, z: qz, w: qw } = camera.quaternion;
    const zoom = camera.position.distanceTo(new THREE.Vector3(0, 0, 0));
    setCoords({
      x: x.toFixed(2),
      y: y.toFixed(2),
      z: z.toFixed(2),
      zoom: zoom.toFixed(2),
      qx: qx.toFixed(2),
      qy: qy.toFixed(2),
      qz: qz.toFixed(2),
      qw: qw.toFixed(2),
    });
  });
  return null;
}

/* --------------------
   Main App - Simplified
   -------------------- */
export default function App() {
  const [movementEnabled, setMovementEnabled] = useState<boolean>(false);
  const [coords, setCoords] = useState({
    x: "0",
    y: "0",
    z: "0",
    zoom: "0",
    qx: "0",
    qy: "0",
    qz: "0",
    qw: "1",
  });

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "white",
        overflow: "hidden",
      }}
    >
      {/* Only Free Move Button */}
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 20 }}>
        <button onClick={() => setMovementEnabled((m) => !m)}>
          {movementEnabled ? "Exit Free Move" : "Enter Free Move"}
        </button>
      </div>

      {/* HUD */}
      <div
        style={{
          position: "absolute",
          top: 80,
          left: 16,
          zIndex: 20,
          background: "rgba(0,0,0,0.7)",
          color: "lime",
          padding: "8px 12px",
          borderRadius: 6,
          fontFamily: "monospace",
        }}
      >
        <div>X: {coords.x}</div>
        <div>Y: {coords.y}</div>
        <div>Z: {coords.z}</div>
        <div>Zoom: {coords.zoom}</div>
        <div>QuatX: {coords.qx}</div>
        <div>QuatY: {coords.qy}</div>
        <div>QuatZ: {coords.qz}</div>
        <div>QuatW: {coords.qw}</div>
      </div>

      <Canvas
        camera={{
          fov: 50,
          position: [2, 1, 2],
          far: 1000,
          near: 0.01,
        }}
      >
        <color attach="background" args={["#f0f0f0"]} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} />
        <pointLight position={[0, 5, 10]} intensity={0.8} />

        <FreeMoveControls enabled={movementEnabled} />
        <Model />
        <CameraTracker setCoords={setCoords} />
      </Canvas>
    </div>
  );
}

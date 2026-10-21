/**
 * Line Following Drone — 3D Simulator Engine
 * 
 * Features:
 * - Three.js physics & graphics pipeline
 * - Accurate PID Controller matching pid_controller.py
 * - Body-frame velocity flight dynamics (vx forward, vy lateral PID steering, vz altitude)
 * - Downward camera FOV with OpenCV-style computer vision overlay
 * - Multiple switchable tracks (Oval, Figure 8, S-Curve, Complex)
 * - 3D Flight path trail visualization
 * - Multiple camera perspectives (Follow, Top-down, Free Orbit)
 * - Real-time telemetry dashboard & PID tuning
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ==============================================================================
// 1. PID Controller Class (Mirrors pid_controller.py)
// ==============================================================================

class PIDController {
    constructor(kp = 0.3, ki = 0.0, kd = 0.1, integralLimit = 50.0, outputLimit = 1.5) {
        this.kp = kp;
        this.ki = ki;
        this.kd = kd;
        this.integralLimit = integralLimit;
        this.outputLimit = outputLimit;

        this.integral = 0.0;
        this.previousError = 0.0;
        this.hasPrevious = false;

        // Telemetry breakdown
        this.lastP = 0.0;
        this.lastI = 0.0;
        this.lastD = 0.0;
        this.lastOutput = 0.0;
    }

    compute(error, dt) {
        if (!dt || dt <= 0) dt = 0.016;

        // Proportional
        this.lastP = this.kp * error;

        // Integral with anti-windup clamping
        this.integral += error * dt;
        this.integral = Math.max(-this.integralLimit, Math.min(this.integralLimit, this.integral));
        this.lastI = this.ki * this.integral;

        // Derivative
        let derivative = 0.0;
        if (this.hasPrevious) {
            derivative = (error - this.previousError) / dt;
        } else {
            this.hasPrevious = true;
        }
        this.lastD = this.kd * derivative;
        this.previousError = error;

        // Total Output with clamping
        let output = this.lastP + this.lastI + this.lastD;
        if (this.outputLimit !== null) {
            output = Math.max(-this.outputLimit, Math.min(this.outputLimit, output));
        }
        this.lastOutput = output;
        return output;
    }

    reset() {
        this.integral = 0.0;
        this.previousError = 0.0;
        this.hasPrevious = false;
        this.lastP = 0.0;
        this.lastI = 0.0;
        this.lastD = 0.0;
        this.lastOutput = 0.0;
    }
}

// ==============================================================================
// 2. Track Generators
// ==============================================================================

function generateTrackPoints(type) {
    const points = [];
    const numPoints = 200;

    switch (type) {
        case 'figure8': {
            const a = 12; // size
            for (let i = 0; i < numPoints; i++) {
                const t = (i / numPoints) * Math.PI * 2;
                // Lemniscate of Gerono / Bernoulli
                const x = a * Math.sin(t);
                const z = (a / 1.7) * Math.sin(t) * Math.cos(t);
                points.push(new THREE.Vector3(x, 0.02, z));
            }
            break;
        }
        case 'scurve': {
            const length = 28;
            for (let i = 0; i <= numPoints; i++) {
                const frac = i / numPoints;
                const x = (frac - 0.5) * length;
                const z = 6 * Math.sin(frac * Math.PI * 2);
                points.push(new THREE.Vector3(x, 0.02, z));
            }
            break;
        }
        case 'complex': {
            // Irregular closed racetrack with straightaway, chicane, hairpin
            const keypoints = [
                new THREE.Vector3(-12, 0.02, -8),
                new THREE.Vector3(0, 0.02, -10),
                new THREE.Vector3(12, 0.02, -8),
                new THREE.Vector3(14, 0.02, -2),
                new THREE.Vector3(8, 0.02, 2),
                new THREE.Vector3(13, 0.02, 7),
                new THREE.Vector3(6, 0.02, 10),
                new THREE.Vector3(-4, 0.02, 8),
                new THREE.Vector3(-10, 0.02, 4),
                new THREE.Vector3(-14, 0.02, -2),
            ];
            const curve = new THREE.CatmullRomCurve3(keypoints, true, 'catmullrom', 0.5);
            return curve.getSpacedPoints(numPoints);
        }
        case 'oval':
        default: {
            const rx = 14;
            const rz = 8;
            for (let i = 0; i < numPoints; i++) {
                const t = (i / numPoints) * Math.PI * 2;
                const x = rx * Math.cos(t);
                const z = rz * Math.sin(t);
                points.push(new THREE.Vector3(x, 0.02, z));
            }
            break;
        }
    }

    const isClosed = (type !== 'scurve');
    const curve = new THREE.CatmullRomCurve3(points, isClosed, 'catmullrom', 0.2);
    return curve.getSpacedPoints(numPoints * 2);
}

// ==============================================================================
// 3. Drone 3D Model Builder
// ==============================================================================

function buildDroneModel() {
    const droneGroup = new THREE.Group();

    // Central Avionics Pod (Body)
    const bodyGeo = new THREE.BoxGeometry(0.42, 0.12, 0.32);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.8,
        roughness: 0.25
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.castShadow = true;
    droneGroup.add(bodyMesh);

    // Top Canopy / Flight Controller cover
    const canopyGeo = new THREE.CylinderGeometry(0.14, 0.18, 0.06, 16);
    const canopyMat = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        metalness: 0.9,
        roughness: 0.1
    });
    const canopy = new THREE.Mesh(canopyGeo, canopyMat);
    canopy.position.y = 0.08;
    droneGroup.add(canopy);

    // Pixhawk Status LED on top (cyan pulse)
    const ledGeo = new THREE.SphereGeometry(0.025, 12, 12);
    const ledMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const statusLed = new THREE.Mesh(ledGeo, ledMat);
    statusLed.position.set(0, 0.12, 0);
    droneGroup.add(statusLed);

    // 4 Carbon Fiber Arms & Motors
    const armGeo = new THREE.CylinderGeometry(0.016, 0.016, 0.65, 12);
    const armMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        roughness: 0.4
    });

    // Arm 1 & 2 (Diagonal X configuration)
    const armX1 = new THREE.Mesh(armGeo, armMat);
    armX1.rotation.z = Math.PI / 2;
    armX1.rotation.y = Math.PI / 4;
    armX1.castShadow = true;
    droneGroup.add(armX1);

    const armX2 = new THREE.Mesh(armGeo, armMat);
    armX2.rotation.z = Math.PI / 2;
    armX2.rotation.y = -Math.PI / 4;
    armX2.castShadow = true;
    droneGroup.add(armX2);

    // Motor Pods & Propellers
    const motorGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.05, 16);
    const motorMat = new THREE.MeshStandardMaterial({
        color: 0x0284c7,
        metalness: 0.9,
        roughness: 0.2
    });

    const propGeo = new THREE.BoxGeometry(0.48, 0.006, 0.04);
    const propMat = new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        transparent: true,
        opacity: 0.75,
        roughness: 0.3
    });

    const propDiscGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.005, 24);
    const propDiscMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.18
    });

    const armDist = 0.23;
    const motorOffsets = [
        { x:  armDist, z:  armDist, isFront: true,  navColor: 0x10b981 }, // Front Left
        { x: -armDist, z:  armDist, isFront: true,  navColor: 0x10b981 }, // Front Right
        { x:  armDist, z: -armDist, isFront: false, navColor: 0xf43f5e }, // Rear Left
        { x: -armDist, z: -armDist, isFront: false, navColor: 0xf43f5e }, // Rear Right
    ];

    const propellers = [];

    motorOffsets.forEach((pos) => {
        // Motor Mount
        const motor = new THREE.Mesh(motorGeo, motorMat);
        motor.position.set(pos.x, 0.02, pos.z);
        motor.castShadow = true;
        droneGroup.add(motor);

        // Navigation LED under motor
        const navLightGeo = new THREE.SphereGeometry(0.018, 8, 8);
        const navLightMat = new THREE.MeshBasicMaterial({ color: pos.navColor });
        const navLight = new THREE.Mesh(navLightGeo, navLightMat);
        navLight.position.set(pos.x, -0.025, pos.z);
        droneGroup.add(navLight);

        // Propeller Group (rotates on Y)
        const propGroup = new THREE.Group();
        propGroup.position.set(pos.x, 0.05, pos.z);

        const propBlade1 = new THREE.Mesh(propGeo, propMat);
        propGroup.add(propBlade1);

        const propBlade2 = new THREE.Mesh(propGeo, propMat);
        propBlade2.rotation.y = Math.PI / 2;
        propGroup.add(propBlade2);

        // Blurred disk
        const propDisc = new THREE.Mesh(propDiscGeo, propDiscMat);
        propGroup.add(propDisc);

        droneGroup.add(propGroup);
        propellers.push(propGroup);
    });

    // Downward Camera Pod underneath
    const camPodGeo = new THREE.CylinderGeometry(0.035, 0.045, 0.06, 16);
    const camPodMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8 });
    const camPod = new THREE.Mesh(camPodGeo, camPodMat);
    camPod.position.set(0, -0.08, 0);
    droneGroup.add(camPod);

    // Camera Lens Ring
    const lensGeo = new THREE.TorusGeometry(0.025, 0.005, 8, 24);
    const lensMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.position.set(0, -0.11, 0);
    lens.rotation.x = Math.PI / 2;
    droneGroup.add(lens);

    // Camera FOV Visualizer Cone (semi-transparent scanning cone)
    const coneHeight = 3.0;
    const coneRadius = coneHeight * Math.tan(THREE.MathUtils.degToRad(35)); // ~70 deg FOV
    const coneGeo = new THREE.ConeGeometry(coneRadius, coneHeight, 32, 1, true);
    // Align cone apex to camera (0, -0.1, 0) pointing downward
    coneGeo.translate(0, -coneHeight / 2, 0);
    const coneMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.06,
        wireframe: false,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const fovCone = new THREE.Mesh(coneGeo, coneMat);
    fovCone.position.set(0, -0.1, 0);
    droneGroup.add(fovCone);

    // Downward Spotlight projection onto ground
    const camLight = new THREE.SpotLight(0x38bdf8, 2.5, 8, Math.PI / 4, 0.4, 1.2);
    camLight.position.set(0, -0.1, 0);
    camLight.target.position.set(0, -5, 0);
    droneGroup.add(camLight);
    droneGroup.add(camLight.target);

    return {
        mesh: droneGroup,
        propellers,
        statusLed,
        fovCone,
        camLight
    };
}

// ==============================================================================
// 4. Main Simulator Application
// ==============================================================================

class DroneSimulator {
    constructor() {
        this.container = document.getElementById('canvas-container');
        this.camCanvas = document.getElementById('drone-cam-canvas');
        this.camCtx = this.camCanvas ? this.camCanvas.getContext('2d') : null;

        // Simulation State
        this.isPlaying = false;
        this.simSpeed = 1.0;
        this.currentTrackType = 'oval';
        this.camMode = 'follow'; // 'follow' | 'top' | 'free'

        // Flight Parameters
        this.forwardSpeed = 0.3; // m/s (from config.FORWARD_SPEED)
        this.targetAltitude = 2.0; // m (from config.TAKEOFF_ALTITUDE)
        this.lateralSpeed = 0.0;
        this.distanceTraveled = 0.0;

        // Drone Physics & Kinematics
        this.dronePos = new THREE.Vector3(0, 0, 0);
        this.droneYaw = 0.0;
        this.dronePitch = 0.0;
        this.droneRoll = 0.0;
        this.currentAltitude = 0.0;

        // PID Controller
        this.pid = new PIDController(0.3, 0.0, 0.1, 50.0, 1.5);

        // Vision Tracking State
        this.lineDetected = true;
        this.currentError = 0.0; // Normalized [-1, 1]
        this.lostTimer = 0.0;

        // Trail points
        this.trailPoints = [];
        this.maxTrailLength = 400;

        // Performance / FPS
        this.fpsCount = 0;
        this.lastFpsTime = performance.now();
        this.clock = new THREE.Clock();

        this.initThree();
        this.initScene();
        this.initTrail();
        this.initDrone();
        this.initTrack(this.currentTrackType);
        this.initUI();
        this.animate();
    }

    initThree() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e17);
        this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.015);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Cameras
        this.aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(55, this.aspect, 0.1, 200);
        this.camera.position.set(0, 8, 14);

        // OrbitControls (for free mode)
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent below ground
        this.controls.enabled = false; // Disabled until 'free' mode

        // Window resize handler
        window.addEventListener('resize', () => this.onWindowResize());
    }

    initScene() {
        // Ambient Light
        const ambientLight = new THREE.AmbientLight(0x1e293b, 1.8);
        this.scene.add(ambientLight);

        // Directional Sun Light
        const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
        dirLight.position.set(15, 30, 20);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 100;
        dirLight.shadow.camera.left = -25;
        dirLight.shadow.camera.right = 25;
        dirLight.shadow.camera.top = 25;
        dirLight.shadow.camera.bottom = -25;
        dirLight.shadow.bias = -0.0005;
        this.scene.add(dirLight);

        // Ground Arena Floor
        const groundGeo = new THREE.PlaneGeometry(120, 120);
        const groundMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            roughness: 0.85,
            metalness: 0.15
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Floor Grid Overlay
        const grid = new THREE.GridHelper(120, 120, 0x38bdf8, 0x1e293b);
        grid.position.y = 0.005;
        grid.material.opacity = 0.35;
        grid.material.transparent = true;
        this.scene.add(grid);

        // Outer circular launch ring
        const ringGeo = new THREE.RingGeometry(24, 24.2, 64);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.2, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.01;
        this.scene.add(ring);
    }

    initTrack(trackType) {
        // Clean up existing track meshes
        if (this.trackLineMesh) this.scene.remove(this.trackLineMesh);
        if (this.trackBorderMesh) this.scene.remove(this.trackBorderMesh);
        if (this.trackStartMesh) this.scene.remove(this.trackStartMesh);

        this.trackPoints = generateTrackPoints(trackType);

        // Build continuous ribbon track geometry
        const lineGeo = new THREE.BufferGeometry().setFromPoints(this.trackPoints);
        
        // Ribbon strip mesh on ground (width = 0.28m, black tarmac stripe with high contrast)
        const ribbonGeom = new THREE.BufferGeometry();
        const positions = [];
        const uvs = [];
        const numPts = this.trackPoints.length;

        for (let i = 0; i < numPts; i++) {
            const curr = this.trackPoints[i];
            const next = this.trackPoints[(i + 1) % numPts];
            const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
            // Normal in XZ ground plane
            const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

            const halfWidth = 0.22; // 44cm track width
            const pLeft = curr.clone().addScaledVector(normal, halfWidth);
            const pRight = curr.clone().addScaledVector(normal, -halfWidth);

            positions.push(pLeft.x, 0.012, pLeft.z);
            positions.push(pRight.x, 0.012, pRight.z);

            const v = i / numPts;
            uvs.push(0, v);
            uvs.push(1, v);
        }

        const indices = [];
        for (let i = 0; i < numPts - 1; i++) {
            const i0 = i * 2;
            const i1 = i0 + 1;
            const i2 = i0 + 2;
            const i3 = i0 + 3;
            indices.push(i0, i1, i2);
            indices.push(i2, i1, i3);
        }

        // Close loop if closed track
        if (trackType !== 'scurve') {
            const last = (numPts - 1) * 2;
            indices.push(last, last + 1, 0);
            indices.push(0, last + 1, 1);
        }

        ribbonGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        ribbonGeom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        ribbonGeom.setIndex(indices);
        ribbonGeom.computeVertexNormals();

        // High-contrast line material (crisp black track with glowing neon edge borders)
        const ribbonMat = new THREE.MeshStandardMaterial({
            color: 0x020617,
            roughness: 0.9,
            metalness: 0.1,
            side: THREE.DoubleSide
        });
        this.trackLineMesh = new THREE.Mesh(ribbonGeom, ribbonMat);
        this.scene.add(this.trackLineMesh);

        // Center line guide (luminous white/cyan dotted line in center)
        const centerLineMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            linewidth: 2,
            transparent: true,
            opacity: 0.85
        });
        this.trackBorderMesh = new THREE.Line(lineGeo, centerLineMat);
        this.trackBorderMesh.position.y = 0.015;
        this.scene.add(this.trackBorderMesh);

        // Start / Finish Checkered Pad
        const startPt = this.trackPoints[0];
        const startPadGeo = new THREE.BoxGeometry(0.8, 0.02, 0.8);
        const startPadMat = new THREE.MeshStandardMaterial({
            color: 0x0284c7,
            metalness: 0.5,
            roughness: 0.3
        });
        this.trackStartMesh = new THREE.Mesh(startPadGeo, startPadMat);
        this.trackStartMesh.position.set(startPt.x, 0.01, startPt.z);
        this.scene.add(this.trackStartMesh);

        // Reset Drone position to start of new track
        this.resetDronePosition();
    }

    initDrone() {
        const drone = buildDroneModel();
        this.droneMesh = drone.mesh;
        this.propellers = drone.propellers;
        this.statusLed = drone.statusLed;
        this.fovCone = drone.fovCone;
        this.scene.add(this.droneMesh);
    }

    initTrail() {
        const trailMax = this.maxTrailLength;
        const trailPositions = new Float32Array(trailMax * 3);
        const trailGeo = new THREE.BufferGeometry();
        trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));

        const trailMat = new THREE.LineBasicMaterial({
            color: 0x10b981,
            linewidth: 2,
            transparent: true,
            opacity: 0.6
        });

        this.trailMesh = new THREE.Line(trailGeo, trailMat);
        this.scene.add(this.trailMesh);
    }

    resetDronePosition() {
        if (!this.trackPoints || this.trackPoints.length === 0) return;

        const start = this.trackPoints[0];
        const next = this.trackPoints[1];
        const dir = new THREE.Vector3().subVectors(next, start).normalize();

        this.dronePos.set(start.x, this.targetAltitude, start.z);
        this.droneYaw = Math.atan2(-dir.z, dir.x); // Align heading with track tangent
        this.dronePitch = 0.0;
        this.droneRoll = 0.0;
        this.currentAltitude = this.targetAltitude;
        this.lateralSpeed = 0.0;
        this.distanceTraveled = 0.0;

        this.pid.reset();
        this.trailPoints = [];
        this.updateTrail();

        if (this.droneMesh) {
            this.droneMesh.position.copy(this.dronePos);
            this.droneMesh.rotation.set(0, this.droneYaw, 0);
        }
    }

    // ==============================================================================
    // 5. Vision Sensor Simulation (Mirrors line_detection.py)
    // ==============================================================================

    updateVision() {
        // Find closest point on track to drone ground projection
        const droneGround = new THREE.Vector2(this.dronePos.x, this.dronePos.z);
        let minDistSq = Infinity;
        let closestIdx = 0;

        for (let i = 0; i < this.trackPoints.length; i++) {
            const p = this.trackPoints[i];
            const distSq = (droneGround.x - p.x) ** 2 + (droneGround.y - p.z) ** 2;
            if (distSq < minDistSq) {
                minDistSq = distSq;
                closestIdx = i;
            }
        }

        const closest = this.trackPoints[closestIdx];
        const nextIdx = (closestIdx + 1) % this.trackPoints.length;
        const next = this.trackPoints[nextIdx];

        // Track tangent direction in XZ plane
        const tangent = new THREE.Vector2(next.x - closest.x, next.z - closest.z).normalize();
        
        // Drone heading vector in XZ plane
        // In Three.js: yaw 0 points towards +X or forward
        const heading = new THREE.Vector2(Math.cos(this.droneYaw), -Math.sin(this.droneYaw));

        // Drone position relative to closest track point
        const toDrone = new THREE.Vector2(droneGround.x - closest.x, droneGround.y - closest.z);

        // Signed lateral distance: cross product of tangent and toDrone
        // Positive means drone is to the right of line, Negative means to the left
        const signedCross = tangent.x * toDrone.y - tangent.y * toDrone.x;
        const lateralOffsetMeters = signedCross;

        // Downward camera FOV visible ground width at current altitude:
        // FOV ~ 70 deg -> width = 2 * alt * tan(35deg) ~ 1.4 * altitude
        const visibleHalfWidth = Math.max(0.6, this.dronePos.y * Math.tan(THREE.MathUtils.degToRad(35)));

        // Camera error normalized between -1.0 and 1.0 (matches main.py: error / frame_half_width)
        // If line is to the right of camera center, error is positive.
        // If drone is shifted right (+lateralOffset), line appears to the LEFT of camera (-error).
        const rawNormalizedError = -lateralOffsetMeters / visibleHalfWidth;

        if (Math.abs(lateralOffsetMeters) < visibleHalfWidth) {
            this.lineDetected = true;
            this.currentError = Math.max(-1.0, Math.min(1.0, rawNormalizedError));
            this.lostTimer = 0.0;
        } else {
            // Line out of frame!
            this.lineDetected = false;
            this.lostTimer += 0.016;
        }

        // Tangent heading angle
        this.trackHeading = Math.atan2(-tangent.y, tangent.x);

        // Draw simulated OpenCV camera view
        this.renderCameraHUD(this.currentError, this.lineDetected);
    }

    renderCameraHUD(normError, detected) {
        if (!this.camCtx) return;

        const ctx = this.camCtx;
        const w = this.camCanvas.width;
        const h = this.camCanvas.height;
        const cx = w / 2;
        const cy = h / 2;

        // Dark simulated ground background
        ctx.fillStyle = '#0a0f1d';
        ctx.fillRect(0, 0, w, h);

        // Grid floor in FPV view
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 24) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 24) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Green Center reference vertical line (from main.py draw_overlay)
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, h);
        ctx.stroke();
        ctx.setLineDash([]);

        // Horizontal reticle
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(w, cy);
        ctx.stroke();

        if (detected) {
            // Compute centroid position in canvas coordinates
            const lineX = cx + (normError * (w / 2) * 0.85);

            // Draw line strip representation
            ctx.fillStyle = '#020617';
            ctx.fillRect(lineX - 16, 0, 32, h);
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(lineX, 0);
            ctx.lineTo(lineX, h);
            ctx.stroke();

            // Blue error vector from center to centroid (main.py)
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(lineX, cy);
            ctx.stroke();

            // Red centroid circle (main.py)
            ctx.fillStyle = '#f43f5e';
            ctx.beginPath();
            ctx.arc(lineX, cy, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Text Overlays
            const pixelErr = Math.round(normError * (w / 2));
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillStyle = '#f8fafc';
            ctx.fillText(`Error: ${pixelErr >= 0 ? '+' : ''}${pixelErr}px`, 8, 16);
            ctx.fillText(`PID:   ${this.pid.lastOutput >= 0 ? '+' : ''}${this.pid.lastOutput.toFixed(3)}`, 8, 30);

            // Status label
            ctx.fillStyle = '#10b981';
            ctx.font = 'bold 10px Inter, sans-serif';
            ctx.fillText('TRACKING', w - 66, 16);
        } else {
            // Line Lost Alert (from main.py: cv2.putText "LINE LOST")
            ctx.fillStyle = 'rgba(244, 63, 94, 0.2)';
            ctx.fillRect(0, 0, w, h);

            ctx.font = 'bold 16px "JetBrains Mono", monospace';
            ctx.fillStyle = '#f43f5e';
            ctx.textAlign = 'center';
            ctx.fillText('LINE LOST', cx, cy - 4);

            ctx.font = '10px Inter, sans-serif';
            ctx.fillStyle = '#94a3b8';
            ctx.fillText('HOVER SEARCHING', cx, cy + 16);
            ctx.textAlign = 'left';
        }
    }

    // ==============================================================================
    // 6. Physics & Flight Dynamics Update
    // ==============================================================================

    updateFlight(dt) {
        if (!this.isPlaying) return;

        // Effective delta time with simulation speed multiplier
        const effectiveDt = dt * this.simSpeed;

        // 1. Vision feedback
        this.updateVision();

        // 2. Control computation
        let pidOutput = 0.0;
        let vx = 0.0;
        let vy = 0.0;

        if (this.lineDetected) {
            // In main.py: pid_output = pid.compute(normalized_error)
            // fc.send_body_velocity(vx=config.FORWARD_SPEED, vy=pid_output, vz=0)
            pidOutput = this.pid.compute(this.currentError, effectiveDt);
            vx = this.forwardSpeed;
            // vy moves drone laterally: positive error (line to right) requires drone to move right (+vy)
            vy = pidOutput;
        } else {
            // Line lost behavior: hover in place and slowly damp velocity
            this.pid.reset();
            vx = 0.0;
            vy = 0.0;
        }

        // 3. Smooth Yaw Alignment: Align drone forward direction to track heading
        // Gently steer heading towards track tangent to handle curved racetracks
        let yawDiff = this.trackHeading - this.droneYaw;
        // Normalize angle to [-PI, PI]
        while (yawDiff > Math.PI) yawDiff -= Math.PI * 2;
        while (yawDiff < -Math.PI) yawDiff += Math.PI * 2;
        this.droneYaw += yawDiff * Math.min(1.0, 3.5 * effectiveDt);

        // 4. Transform body velocities (vx, vy) into world velocities (worldX, worldZ)
        const cosYaw = Math.cos(this.droneYaw);
        const sinYaw = Math.sin(this.droneYaw);
        // Drone local body frame:
        // Forward: +X in world rotated by Yaw
        // Right/Lateral: -Z in world rotated by Yaw
        const worldVx = (vx * cosYaw) + (vy * sinYaw);
        const worldVz = (-vx * sinYaw) + (vy * cosYaw);

        // 5. Update Drone World Position
        this.dronePos.x += worldVx * effectiveDt;
        this.dronePos.z += worldVz * effectiveDt;

        // 6. Altitude Hold
        const altDiff = this.targetAltitude - this.currentAltitude;
        this.currentAltitude += altDiff * Math.min(1.0, 4.0 * effectiveDt);
        this.dronePos.y = this.currentAltitude;

        // Distance accumulator
        this.distanceTraveled += Math.sqrt(worldVx ** 2 + worldVz ** 2) * effectiveDt;

        // 7. Dynamic Pitch & Roll Banking Angles
        // Moving forward tilts nose down (pitch). Lateral acceleration causes roll banking!
        const targetPitch = (vx / 2.0) * 0.15; // rad
        const targetRoll = (vy / 1.5) * -0.22; // bank into turn

        this.dronePitch += (targetPitch - this.dronePitch) * Math.min(1.0, 8.0 * effectiveDt);
        this.droneRoll += (targetRoll - this.droneRoll) * Math.min(1.0, 8.0 * effectiveDt);

        // Apply transformations to Drone 3D Mesh
        this.droneMesh.position.copy(this.dronePos);
        this.droneMesh.rotation.set(0, 0, 0);
        this.droneMesh.rotateY(this.droneYaw);
        this.droneMesh.rotateX(this.dronePitch);
        this.droneMesh.rotateZ(this.droneRoll);

        // Spin propellers
        const propSpeed = 45 * this.simSpeed;
        this.propellers.forEach((p, idx) => {
            const dir = (idx % 2 === 0) ? 1 : -1;
            p.rotation.y += propSpeed * effectiveDt * dir;
        });

        // Breadcrumb Trail Update
        this.recordTrailPoint(this.dronePos);
    }

    recordTrailPoint(pos) {
        if (!this.isPlaying) return;

        // Append every ~0.1m traveled
        const lastPt = this.trailPoints[this.trailPoints.length - 1];
        if (!lastPt || lastPt.distanceTo(pos) > 0.12) {
            this.trailPoints.push(pos.clone().add(new THREE.Vector3(0, -0.05, 0)));
            if (this.trailPoints.length > this.maxTrailLength) {
                this.trailPoints.shift();
            }
            this.updateTrail();
        }
    }

    updateTrail() {
        if (!this.trailMesh || !this.trailMesh.geometry) return;
        const positions = this.trailMesh.geometry.attributes.position.array;
        for (let i = 0; i < this.trailPoints.length; i++) {
            const pt = this.trailPoints[i];
            positions[i * 3] = pt.x;
            positions[i * 3 + 1] = pt.y;
            positions[i * 3 + 2] = pt.z;
        }
        this.trailMesh.geometry.setDrawRange(0, this.trailPoints.length);
        this.trailMesh.geometry.attributes.position.needsUpdate = true;
    }

    // ==============================================================================
    // 7. Camera System
    // ==============================================================================

    updateCamera() {
        if (this.camMode === 'free') {
            this.controls.enabled = true;
            this.controls.update();
            return;
        }

        this.controls.enabled = false;

        if (this.camMode === 'top') {
            // Overhead Tactical View following the drone
            const targetPos = new THREE.Vector3(this.dronePos.x, 18, this.dronePos.z);
            this.camera.position.lerp(targetPos, 0.08);
            this.camera.lookAt(this.dronePos.x, 0, this.dronePos.z);
        } else {
            // Chase / Follow Camera (3rd person chase)
            // Position behind drone according to droneYaw
            const followDist = 3.8;
            const followHeight = 2.0;

            const behindX = this.dronePos.x - followDist * Math.cos(this.droneYaw);
            const behindZ = this.dronePos.z + followDist * Math.sin(this.droneYaw);
            const targetCamPos = new THREE.Vector3(behindX, this.dronePos.y + followHeight, behindZ);

            this.camera.position.lerp(targetCamPos, 0.08);
            
            // Look slightly ahead of drone
            const lookTarget = this.dronePos.clone().add(
                new THREE.Vector3(1.2 * Math.cos(this.droneYaw), 0, -1.2 * Math.sin(this.droneYaw))
            );
            this.camera.lookAt(lookTarget);
        }
    }

    // ==============================================================================
    // 8. User Interface & Telemetry
    // ==============================================================================

    initUI() {
        // Buttons
        const btnPlay = document.getElementById('btn-play');
        const btnPause = document.getElementById('btn-pause');
        const btnReset = document.getElementById('btn-reset');

        btnPlay.addEventListener('click', () => {
            this.isPlaying = true;
            btnPlay.disabled = true;
            btnPause.disabled = false;
        });

        btnPause.addEventListener('click', () => {
            this.isPlaying = false;
            btnPlay.disabled = false;
            btnPause.disabled = true;
        });

        btnReset.addEventListener('click', () => {
            this.isPlaying = false;
            btnPlay.disabled = false;
            btnPause.disabled = true;
            this.resetDronePosition();
        });

        // Track Selector
        const trackSelect = document.getElementById('track-select');
        trackSelect.addEventListener('change', (e) => {
            this.currentTrackType = e.target.value;
            this.initTrack(this.currentTrackType);
        });

        // PID Sliders
        const sliderKp = document.getElementById('slider-kp');
        const sliderKi = document.getElementById('slider-ki');
        const sliderKd = document.getElementById('slider-kd');
        const valKp = document.getElementById('val-kp');
        const valKi = document.getElementById('val-ki');
        const valKd = document.getElementById('val-kd');

        sliderKp.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.pid.kp = val;
            valKp.textContent = val.toFixed(2);
        });

        sliderKi.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.pid.ki = val;
            valKi.textContent = val.toFixed(3);
        });

        sliderKd.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.pid.kd = val;
            valKd.textContent = val.toFixed(2);
        });

        // Flight Sliders
        const sliderSpeed = document.getElementById('slider-speed');
        const sliderAlt = document.getElementById('slider-alt');
        const sliderSimSpeed = document.getElementById('slider-simspeed');
        const valSpeed = document.getElementById('val-speed');
        const valAlt = document.getElementById('val-alt');
        const valSimSpeed = document.getElementById('val-simspeed');

        sliderSpeed.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.forwardSpeed = val;
            valSpeed.textContent = `${val.toFixed(2)} m/s`;
        });

        sliderAlt.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.targetAltitude = val;
            valAlt.textContent = `${val.toFixed(1)} m`;
        });

        sliderSimSpeed.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.simSpeed = val;
            valSimSpeed.textContent = `${val.toFixed(1)}×`;
        });

        // Camera Buttons
        const camButtons = document.querySelectorAll('.btn-cam');
        camButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                camButtons.forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
                this.camMode = btn.dataset.cam;

                if (this.camMode === 'free') {
                    this.controls.target.copy(this.dronePos);
                }
            });
        });

        // Cache Telemetry DOM Elements
        this.elStatus = document.getElementById('telem-status');
        this.elError = document.getElementById('telem-error');
        this.elPid = document.getElementById('telem-pid');
        this.elP = document.getElementById('telem-p');
        this.elI = document.getElementById('telem-i');
        this.elD = document.getElementById('telem-d');
        this.elSpeed = document.getElementById('telem-speed');
        this.elAlt = document.getElementById('telem-altval');
        this.elDist = document.getElementById('telem-dist');
        this.elFps = document.getElementById('fps-counter');
    }

    updateTelemetry() {
        if (!this.elStatus) return;

        // Status
        if (!this.isPlaying) {
            this.elStatus.textContent = 'IDLE';
            this.elStatus.className = 'telem-value status-idle';
        } else if (this.lineDetected) {
            this.elStatus.textContent = 'TRACKING';
            this.elStatus.className = 'telem-value status-tracking';
        } else {
            this.elStatus.textContent = 'LINE LOST';
            this.elStatus.className = 'telem-value status-lost';
        }

        // Values
        this.elError.textContent = (this.currentError >= 0 ? '+' : '') + this.currentError.toFixed(3);
        this.elPid.textContent = (this.pid.lastOutput >= 0 ? '+' : '') + this.pid.lastOutput.toFixed(3);
        this.elP.textContent = (this.pid.lastP >= 0 ? '+' : '') + this.pid.lastP.toFixed(3);
        this.elI.textContent = (this.pid.lastI >= 0 ? '+' : '') + this.pid.lastI.toFixed(3);
        this.elD.textContent = (this.pid.lastD >= 0 ? '+' : '') + this.pid.lastD.toFixed(3);

        const currentSpeed = this.isPlaying ? this.forwardSpeed : 0.0;
        this.elSpeed.textContent = `${currentSpeed.toFixed(2)} m/s`;
        this.elAlt.textContent = `${this.dronePos.y.toFixed(1)} m`;
        this.elDist.textContent = `${this.distanceTraveled.toFixed(1)} m`;
    }

    updateFPS() {
        this.fpsCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            const fps = Math.round((this.fpsCount * 1000) / (now - this.lastFpsTime));
            if (this.elFps) this.elFps.textContent = `${fps} FPS`;
            this.fpsCount = 0;
            this.lastFpsTime = now;
        }
    }

    onWindowResize() {
        this.aspect = window.innerWidth / window.innerHeight;
        this.camera.aspect = this.aspect;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const dt = Math.min(this.clock.getDelta(), 0.1);

        this.updateFlight(dt);
        this.updateCamera();
        this.updateTelemetry();
        this.updateFPS();

        this.renderer.render(this.scene, this.camera);
    }
}

// Start Simulator once DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    new DroneSimulator();
});

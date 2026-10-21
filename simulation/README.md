# 🚁 Line Following Drone — 3D Simulator

An interactive, high-fidelity 3D web simulator designed for laptop testing and tuning of autonomous line-following drone control algorithms.

No physical drone, Raspberry Pi, Pixhawk, or Gazebo installation required. Runs smoothly in any modern web browser on macOS.

---

## 🌟 Features

- **Accurate PID Implementation**: Uses the exact same proportional, anti-windup clamped integral, and derivative logic from [`pid_controller.py`](../pid_controller.py).
- **Body-Frame Flight Dynamics**: Implements forward velocity ($v_x$), lateral correction ($v_y$), and altitude hold ($v_z$) matching [`flight_controller.py`](../flight_controller.py).
- **OpenCV Vision Simulation**:
  - Downward camera field-of-view (FOV) sensor.
  - Picture-in-picture (PIP) simulated camera HUD rendering centroid detection, center line, error offset vector, and status overlays matching `draw_overlay()` in [`main.py`](../main.py).
- **Multiple Circuit Tracks**:
  - **Oval Circuit**: Classic continuous loop for tuning baseline $K_p$ and $K_d$.
  - **Figure 8**: Sharp turns and crossover intersection.
  - **S-Curve**: Rapid alternating directional transitions.
  - **Complex Circuit**: Mixed straightaways, chicanes, and hairpin turns.
- **Flight Path Breadcrumb Trail**: 3D ribbon tracing the drone's actual trajectory to visualize overshoot, oscillations, and settling time.
- **Camera Angles**:
  - **Follow View**: 3rd-person chase camera smoothly tracking the drone.
  - **Top View**: Tactical overhead perspective for tracking alignment.
  - **Free View**: OrbitControls allowing 360° pan, zoom, and tilt with your mouse.
- **Real-Time Telemetry**: Real-time breakdown of P, I, D contributions, error, altitude, speed, and distance traveled.

---

## 🚀 How to Run

### Method 1: Python Launcher (Recommended)

From the project root:
```bash
python3 simulation/run_sim.py
```
This will start a local HTTP server and automatically launch your default browser at `http://localhost:8000`.

### Method 2: Direct Python HTTP Server

```bash
cd simulation
python3 -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🎮 How to Use

1. Click **▶ Play** to launch the drone and start autonomous line following.
2. In the right control sidebar, experiment with tuning **Kp**, **Ki**, and **Kd** sliders in real time:
   - Notice how increasing $K_p$ too high causes lateral oscillations (hunting).
   - Notice how adding $K_d$ dampens oscillations and smooths cornering.
   - Observe how $K_i$ corrects steady-state drift on sweeping curves.
3. Switch tracks from the **Track** dropdown to test controller performance under different track geometries.
4. Switch camera views (**Follow**, **Top**, **Free**) to inspect drone dynamics and pitch/roll banking.
5. Watch the **Downward FPV Cam** in the top left to see how OpenCV detects the line centroid and calculates pixel error in real time.

#!/usr/bin/env python3
"""
Launcher script for the Line Following Drone 3D Simulator.
Serves the simulation web application and opens it in your default browser.
"""

import os
import sys
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


def find_free_port(start_port=8000, max_attempts=50):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def main():
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(sim_dir)

    port = find_free_port(8000)
    server_address = ('127.0.0.1', port)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print("  🚁 LINE FOLLOWING DRONE — 3D LAPTOP SIMULATOR")
    print("=" * 65)
    print(f"  • Serving from: {sim_dir}")
    print(f"  • Simulator URL: {url}")
    print("  • Controls:")
    print("      - Click '▶ Play' to start autonomous flight")
    print("      - Switch tracks (Oval, Figure 8, S-Curve, Complex)")
    print("      - Adjust Kp, Ki, Kd sliders in real-time")
    print("      - Switch camera: Follow, Top-down, Free Orbit")
    print("      - Watch simulated OpenCV downward camera FPV feed")
    print("=" * 65)
    print("  Press Ctrl+C to stop the simulator server.\n")

    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

    # Open browser automatically
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SIM] Simulator server stopped. Goodbye!")
        httpd.server_close()
        sys.exit(0)


if __name__ == '__main__':
    main()

<div align="center">

# 🤟 SignBridge: Standalone Voice-to-Sign Assistive System

### An autonomous embedded solution designed for educational accessibility, translating real-time speech into sign language gestures without external host dependencies.

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat&logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.com/)
[![OS](https://img.shields.io/badge/OS-Linux%20Debian-A81D33?style=flat&logo=debian&logoColor=white)](https://www.debian.org/)
[![Language](https://img.shields.io/badge/Language-Python%203-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Service](https://img.shields.io/badge/Service-systemd-orange?style=flat&logo=linux&logoColor=white)](#system-architecture)

</div>

---

## 📌 Overview

**SignBridge** is a dedicated, plug-and-play embedded assistive device designed to eliminate communication barriers in academic environments. Unlike traditional software solutions requiring teacher laptops or mobile device pairing, SignBridge operates as a completely **standalone system** powered by a single-board computer (Raspberry Pi).

Equipped with an independent microphone, it captures the instructor's vocal input, translates semantic concepts into corresponding American Sign Language (ASL) representations, and displays visual playback in real time on classroom screens or dedicated displays.

---

## ⚙️ Key Technical Features

- **Standalone & Headless Ready:** Operates independently without requiring teacher devices or smartphones.
- **Audio Acquisition & Processing:** Plug-and-play microphone integration with noise resilience for classroom environments.
- **Dynamic Semantic Mapping:** Maps parsed terms directly to curated sign language visual assets (`mapping.json`).
- **Low-Latency Visual Playback:** Lightweight playback pipeline optimized for resource-constrained ARM architectures.
- **Daemon Automation (`systemd`):** Configured as a background Linux service (`sign.service`) for automatic launch at boot and fail-safe restarts.
- **Remote Administration:** Full headless maintenance, diagnostics, and deployment via secure SSH protocol.

---

## 🏗️ System Architecture & Workflow

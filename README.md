<p align="center">
  <img src="assets/propfix-banner.png" alt="PropFix - AI-assisted photo editor" width="100%" />
</p>

# PropFix: AI-Assisted Photo Editing Electron Application

PropFix is a cross-platform desktop application that pairs a modern **Electron/Node.js** user interface with a **Python** back end to deliver intelligent photo-editing tools. The goal is to demonstrate how computer systems engineering, data science, and modern software architecture can be combined in a portable, product-grade application. As a **passion project**, the repository emphasizes reproducibility, documentation, and careful version control so it can serve as part of a professional portfolio.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Features](#features)  
3. [Technology Stack](#technology-stack)  
4. [Architecture](#architecture)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Data-Science Details](#data-science-details)  
8. [Model Training & Integration](#model-training--integration)  
9. [Version Control & Contribution Guidelines](#version-control--contribution-guidelines)  
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)

---

## Motivation

Modern photo editors often rely on a mix of manual sliders and presets. As a computer systems engineer with a data-science focus, I wanted to explore how far AI can automate high-quality edits while still keeping the user in control. Electron makes it straightforward to build cross-platform desktop apps, and Python brings mature image-processing and ML tooling. PropFix integrates them so the UI stays responsive while heavy lifting happens in a separate Python process.

---

## Features

PropFix is designed to evolve into a full-fledged AI-assisted photo editor. The current roadmap includes:

- **Photo management** — load single images or folders, display thumbnails, and organize your work.  
- **Automatic enhancements** — color balance, white balance, noise reduction via pre-trained deep-learning models.  
- **Object detection and segmentation** — identify regions (e.g., person/parts) for selective, region-aware edits.  
- **Generative editing** — inpainting and style transfer via modern diffusion/GAN-based approaches.  
- **Undo/Redo and history** — keep a timeline of edits so you can revert or replay.  
- **Data export** — save PNG/JPEG/TIFF and export JSON metadata describing the edit pipeline.

> UI niceties already implemented include a log pane, “Reset All,” sensible defaults, and responsive zoom controls (Fit/Fill/1:1). The Python side produces preview images (see `server/outputs/`).

---

## Technology Stack

PropFix intentionally uses a hybrid stack:

| Layer               | Technology / Libraries                                                | Rationale |
|---------------------|------------------------------------------------------------------------|-----------|
| **User Interface**  | **Electron** (HTML/CSS/JavaScript)                                    | Portable, modern desktop UI; access to filesystem and native menus while rendering a web-quality interface. |
| **Application Logic** | **Node.js**                                                          | Coordinates renderer events, filesystem I/O, and calls to the Python back end over HTTP. |
| **Back End**        | **Python 3** (Flask, Pillow/OpenCV, NumPy, etc.)                      | Robust image processing, segmentation/warping, and room for ML models. Exposed as an HTTP server so the UI remains responsive. |
| **Data Science**    | Deep-learning models (U-Net/Mask-RCNN, ESRGAN/DnCNN, diffusion, etc.) | Segmentation, super-resolution/denoising, inpainting, style transfer—extensible via Python modules. |
| **Version Control** | Git + Conventional Commits                                            | Clean history and semantic-style commit messages. |

---

## Architecture

The application uses a simple, testable separation of concerns:

1. **Renderer (Electron)**  
   Renders the UI, gathers user inputs (sliders, toggles), and issues HTTP requests to the local Python server.  
   Example request:

   ```js
   // classify → returns segments; warp → applies geometry and photo controls
   const res = await fetch('http://127.0.0.1:5001/process', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       mode: 'warp',                     // or 'classify'
       input: 'C:\\path\\to\\image.png', // absolute path from the UI
       controls: {
         geometry: { scaleX: 1.02, scaleY: 0.98, rotate: -1.5, offsetX: 0, offsetY: 0 },
         photo:    { contrast: 1.0, exposure: 0.0, saturation: 1.0, strength: 0.75 }
       }
     })
   });
   const data = await res.json();

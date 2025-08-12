<p align="center">
  <img src="assets/propfix-banner.png" alt="PropFix - AI-assisted photo editor" width="100%" />
</p>

# PropFix: AI-Assisted Photo Editing Electron Application# PropFix: AIâ€‘Assisted Photo Editing Electron Application

PropFix is a crossâ€‘platform desktop application that uses a modern Electron/Node.js user interface and Pythonâ€‘powered machineâ€‘learning backâ€‘end to deliver intelligent photo editing tools.  The projectâ€™s goal is to demonstrate how computer systems engineering, data science, and modern software architecture can be combined in a portable productâ€‘grade application.  As a **passion project**, the repository has been organised with an emphasis on reproducibility, documentation, and careful version control so that it can serve as part of a professional portfolio.

## Table of Contents

1. [Motivation](#motivation)
2. [Features](#features)
3. [TechnologyÂ Stack](#technology-stack)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Dataâ€‘Science Details](#data-science-details)
8. [ModelÂ Training & Integration](#model-training--integration)
9. [Version Control & Contribution Guidelines](#version-control--contribution-guidelines)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Motivation

Modern photo editors often depend on manual slider adjustments and filter presets.  As a computer systems engineer with an interest in data science, I wanted to explore how far AI can automate the process while still giving the user full control.  Electron provides a convenient way to build crossâ€‘platform desktop applications, but some capabilities (for example, sophisticated image processing or access to native operatingâ€‘system features) are not available in JavaScript alone.  A blog on how to execute Python scripts from Electron highlights that certain functionality is easier to implement in Python and that one can embed Python into Electron using packages such as `pythonâ€‘shell`ã€362342226299824â€ L18-L22ã€‘.  PropFix integrates a Python backâ€‘end to perform advanced image processing tasks and to host deepâ€‘learning models while keeping the frontâ€‘end responsive and lightweight.

## Features

PropFix is designed to evolve into a fullâ€‘fledged AIâ€‘assisted photo editor.  The current roadmap includes:

* **Photo management** â€“ load single images or entire folders, display thumbnails, and organise your work.  Inspiration comes from the communityâ€‘developed electron photo editor that lets users load folders of images, scroll through them and apply effectsã€101163776406757â€ L231-L241ã€‘.
* **Automatic enhancements** â€“ apply colourâ€‘balance correction, whiteâ€‘balance adjustment and noise reduction using preâ€‘trained deepâ€‘learning models.  Integration with models such as ESRGAN for superâ€‘resolution or DnCNN for denoising can be configured through the Python backâ€‘end.
* **Object detection and segmentation** â€“ identify people or objects in an image using convolutional neural networks (e.g., YOLO, MaskÂ Râ€‘CNN) and allow regionâ€‘specific edits such as background blurring or subject isolation.
* **Generative editing** â€“ implement inpainting or style transfer using diffusion models or generative adversarial networks.  The backâ€‘end exposes functions to call models such as StableÂ Diffusion to replace objects, fill holes, or transfer artistic styles.
* **Undo/Redo and history** â€“ maintain a history stack of edits so the user can revert or replay transformations.
* **Data export** â€“ save edited images in common formats (PNG, JPEG, TIFF) and export JSON metadata that records the sequence of operations.

## TechnologyÂ Stack

PropFix intentionally uses a hybrid technology stack:

| Layer            | Technology / Libraries                                                                                                                                                                       | Rationale                                                                                                                                                                                                     |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **User Interface** | **Electron** with HTML, CSS and JavaScriptÂ (frontâ€‘end)                                                                                                                                       | Electron uses Chromium and Node.js to build crossâ€‘platform desktop apps.  It allows access to the local file system and system notifications while rendering modern web interfaces.                            |
| **ApplicationÂ Logic** | **Node.js**                                                                                                                                                | Acts as the glue layer for the UI and backâ€‘end.  The `pythonâ€‘shell` npm package is used to spawn Python processes from Node and to communicate via standard input/outputã€362342226299824â€ L53-L63ã€‘.           |
| **Backâ€‘End**       | **PythonÂ 3** with packages such as NumPy, OpenCV, Pillow, PyTorch or TensorFlow                                                                                 | Provides access to advanced image processing and machineâ€‘learning models.  The blog article on executing Python scripts in Electron demonstrates how to build a simple Python script and call it from Nodeã€362342226299824â€ L68-L78ã€‘.  Python is also used for model training and data analysis. |
| **Data Science**   | Deepâ€‘learning models (e.g., convolutional neural networks, diffusion models), scikitâ€‘learn for classical image processing                                                                      | Responsible for automatic enhancements, segmentation, classification and generative tasks.  Models can be trained or fineâ€‘tuned on custom datasets.                                                            |
| **Version Control** | Git with the [ConventionalÂ Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification                                                             | Enforces a structured commit history.  The specification states that commit messages should follow the format `<type>[optional scope]: <description>` with optional body and footersã€385383022001910â€ L56-L64ã€‘. This enables automatic release notes and semantic versioning. |

## Architecture

The application follows a modular architecture:

1. **Renderer process (frontâ€‘end)**: An Electron window renders the user interface.  When a user opens an image or requests an AI operation, the renderer sends a message to the main process via IPC (interâ€‘process communication).

2. **Main process (Node.js)**: Handles file system access, spawns Python processes, and coordinates communication between the renderer and the Python backâ€‘end.  Communication with Python is handled via the `pythonâ€‘shell` package: you install it with `npm install python-shell`ã€362342226299824â€ L53-L63ã€‘.  A simple example of running a Python script from Node is shown below:

   ```js
   const { PythonShell } = require('python-shell');
   const pyshell = new PythonShell('python/image_processing.py');
   pyshell.on('message', (message) => {
     console.log(`Received from Python: ${message}`);
     // send result back to renderer
   });
   // send a command to Python (optional)
   pyshell.send(JSON.stringify({ command: 'enhance', params: { contrast: 1.2 } }));
   pyshell.end((err) => {
     if (err) throw err;
     console.log('Python process finished');
   });
   ```

   This pattern is based on the example in the Skcript article, where `pyshell.on('message')` receives messages from the Python script and `pyshell.end()` terminates the processã€362342226299824â€ L100-L136ã€‘.

3. **Python backâ€‘end**: Implements the imageâ€‘processing algorithms.  For example, the `python/image_processing.py` script may accept JSON commands on `stdin`, run the requested operation using OpenCV or a deepâ€‘learning model, and print the result or output image path to `stdout`.  The Python script can also flush its output using `sys.stdout.flush()` to ensure that Node receives the data promptlyã€362342226299824â€ L68-L82ã€‘.

This separation of concerns lets the UI remain responsive while heavy computations occur in a separate process.  It also allows the Python backâ€‘end to be tested independently.

## Installation

These instructions assume you have **Node.jsÂ 18+** and **PythonÂ 3.9+** installed.  For brevity, commands starting with `$` are executed in a shell.  When a step refers to â€œthe root of the projectâ€ it means the directory containing this `README.md`.

1. **Clone the repository**:

   ```sh
   $ git clone https://github.com/Bobtheotherone/PropFix.git
   $ cd PropFix
   ```

   (If you are viewing this project outside of GitHub, copy the `PropFix` directory included in this repository.)

2. **Install Node dependencies**.  The `package.json` defines Electron and other JavaScript dependencies.  Run:

   ```sh
   $ npm install
   ```

3. **Install Python dependencies**.  Create a virtual environment (recommended) and install packages listed in `requirements.txt`:

   ```sh
   $ python -m venv .venv
   $ source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
   (.venv) $ pip install -r requirements.txt
   ```

4. **Run the application**.  Start the Electron app using npm.  The electron photo editor example uses a similar sequenceâ€”clone the repo, change into it, install dependencies and run `npm start`ã€101163776406757â€ L231-L241ã€‘.  For PropFix, run:

   ```sh
   $ npm start
   ```

   The application will open a new window.  On first launch, the Python environment may take a moment to initialise.

## Usage

1. **Opening images**: Click â€œOpenâ€ and select an image file or directory.  The file explorer uses the `dialog` API from Electron to read local files.
2. **Applying edits**: Choose an edit (e.g., â€œAuto Enhanceâ€, â€œDenoiseâ€, â€œStyle Transferâ€).  The request is sent to the Python backâ€‘end and results are displayed when ready.
3. **Viewing history**: Each edit is recorded in a history panel.  Click on a previous state to revert.
4. **Saving**: Use â€œSave Asâ€¦â€ to write the edited image.  Metadata about the edits is stored in a JSON sidecar file.
5. **Extending**: You can implement new operations by editing `python/image_processing.py` and adding a corresponding button in `src/index.html`.

## Dataâ€‘Science Details

PropFixâ€™s AI capabilities are grounded in established dataâ€‘science techniques.  The backâ€‘end comprises:

* **Preâ€‘processing** â€“ conversion to standard colour spaces (RGB â†”ï¸Ž LAB), normalisation, histogram equalisation and noise estimation.
* **Classical techniques** â€“ algorithms such as Canny edge detection, bilateral filtering and unsharp masking.  These operations run quickly and provide baseline enhancement.
* **Deepâ€‘learning models**:
  * **Superâ€‘resolution / denoising** â€“ models like ESRGAN and DnCNN upscale images and reduce noise.  These networks can be fineâ€‘tuned on custom datasets.
  * **Segmentation** â€“ models such as Uâ€‘Net or MaskÂ Râ€‘CNN identify regions (e.g., subject vs. background) so that edits can be applied selectively.
  * **Style transfer and inpainting** â€“ generative models (e.g., StableÂ Diffusion, VGGâ€‘based style transfer) allow creative modifications.  Integration requires sending the image and prompt parameters to Python functions that wrap these models.

Each model is encapsulated in a Python class under `python/model.py`.  Training scripts and notebooks (under `docs/training/`) demonstrate how to train or fineâ€‘tune the models using widely available datasets (such as COCO for segmentation or DIV2K for superâ€‘resolution).  To reproduce the training, activate the virtual environment and run `python docs/training/train_super_resolution.py`.  Keep in mind that training may require a GPU and several hours of computation.

## ModelÂ Training & Integration

1. **Datasets**: PropFix does not ship with datasets due to their size.  In the `data/README.md` file you will find links to recommended datasets and scripts to download them.
2. **Training**: Use the scripts in `docs/training/` to train models.  For example, `train_super_resolution.py` downloads the DIV2K dataset, performs data augmentation, defines a network in PyTorch or TensorFlow, and saves the trained weights to `models/`.
3. **Integration**: Trained models are loaded in `python/model.py`.  The `image_processing.py` script uses a simple dispatch system to call the appropriate model and to process the input image.  The Node side sends JSON commands specifying which model to use and any parameters.
4. **Evaluation**: Data science is iterative.  Evaluate model performance using PSNR, SSIM or other metrics.  You can run `python docs/training/evaluate.py --model models/esrgan.pth` to compute metrics on a validation set.

## Version Control & Contribution Guidelines

PropFix follows best practices for reproducible research and maintainability:

* **Git** is used for version control.  The `.gitignore` file combines the official Python templateâ€”which ignores byteâ€‘compiled files, build artefacts and virtual environmentsã€35231548267374â€ L0-L24ã€‘â€”and the Node templateâ€”which omits `node_modules/`, log files and build outputsã€925898819000351â€ L0-L42ã€‘.
* **Commit messages** follow the [ConventionalÂ Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.  A commit message should start with a type (e.g., `feat`, `fix`, `docs`) followed by a short description as mandated by the specificationã€385383022001910â€ L56-L64ã€‘.  Additional context can be provided in the body, separated from the header by a blank lineã€385383022001910â€ L152-L158ã€‘.  This structure allows for automated changelog generation and semantic versioning.
* **Branching model**: Use the `main` branch for stable releases.  Development occurs on feature branches (e.g., `feat/auto-enhance`) that branch off `main` and are merged via pull requests.  Each pull request should reference an issue describing the feature or bug.
* **Code style**: Python code must follow [PEPÂ 8](https://peps.python.org/pep-0008/).  JavaScript/TypeScript should be formatted with [Prettier](https://prettier.io/) and linted using ESLint.  Include type annotations where possible.
* **Tests**: Unit tests reside in `python/tests/` and `src/tests/`.  Run Python tests with `pytest` and JavaScript tests with `jest`.
* **Documentation**: Update `README.md` and any relevant documentation in `docs/` when adding new features.  All public functions should contain docstrings/comments explaining their behaviour.

## License

This project is licensed under the terms of the **MIT License**.  See the [LICENSE](LICENSE) file for details.  The license permits reuse, modification and distribution of the software with proper attribution.

## Acknowledgements

PropFix builds upon the efforts of many openâ€‘source communities.  In particular:

* The openâ€‘source **Electron Photo Editor** demonstrates how to construct a photo editor using Electron and CamanJS and inspired the initial folderâ€‘loading and effect application featuresã€101163776406757â€ L231-L241ã€‘.
* The **Skcript tutorial** on executing Python scripts in Electron showed how to bridge Node and Python using the `pythonâ€‘shell` packageã€362342226299824â€ L53-L63ã€‘.  The example code and description of interâ€‘process communication informed our own architectureã€362342226299824â€ L100-L136ã€‘.
* The maintainers of the [Python.gitignore](https://github.com/github/gitignore/blob/master/Python.gitignore) and [Node.gitignore](https://github.com/github/gitignore/blob/master/Node.gitignore) templates provided patterns that protect sensitive files and unnecessary build artefactsã€35231548267374â€ L0-L24ã€‘ã€925898819000351â€ L0-L42ã€‘.

This repository is an ongoing projectâ€”feel free to fork it, explore the code, and contribute improvements![ ! [ C I ] ( . . . ) ] ( & ) 
 
 


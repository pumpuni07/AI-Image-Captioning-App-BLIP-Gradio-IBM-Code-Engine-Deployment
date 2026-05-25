# AI Image Captioning App: BLIP, Gradio & IBM Code Engine Deployment

**Author:** Jack Pumpuni Frimpong-Manso  
**Affiliation:** Environmental Data Scientist | Guest Scientist, ZMT Bremen | AI Engineering  
**Date:** 2026  
**License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)  
**Source:** IBM Skills Network — Guided Project

---

## What This Project Does

This project builds a complete, end-to-end AI image captioning application — from model inference through a live web interface to cloud deployment with a public URL.

Given any image, the app automatically generates a natural language description of its contents using **BLIP** (Bootstrapping Language-Image Pre-training), a state-of-the-art vision-language model from Hugging Face. The app is served through a **Gradio** web interface and deployed to **IBM Cloud** via **Code Engine** as a containerised, publicly accessible application.

---

## Quick Demo

```
Input  → [ image of a dog sitting on a wooden floor next to a window ]
Output → "the image of a dog sitting on a wooden floor next to a window"

Input  → [ Wikipedia page URL ]
Output → captions.txt with one caption per image found on the page
```

---

## Project Structure

```
ai-image-captioning/
│
├── image_cap.py                     # Script 1: Single image captioning (CLI)
├── automate_url_captioner.py        # Script 2: Batch URL-based captioning
├── image_captioning_app.py          # Script 3: Gradio web application (local)
│
├── myapp/                           # Deployment package for IBM Code Engine
│   ├── demo.py                      # Gradio app entry point for container
│   ├── requirements.txt             # Runtime dependencies
│   └── Dockerfile                   # Container image blueprint
│
├── captions.txt                     # Output: auto-generated captions from URL captioner
├── AI_Image_Captioning_Report.md    # Full technical report
├── AI_Image_Captioning_Report.pdf   # Downloadable PDF report
└── README.md                        # This file
```

---

## The Three Scripts

### Script 1 — `image_cap.py` (Single Image)

Loads a local image file, runs it through the BLIP model, and prints a caption to the terminal.

```bash
python3 image_cap.py
```

### Script 2 — `automate_url_captioner.py` (Batch URL Captioning)

Scrapes all images from a given webpage, captions each one, and writes results to `captions.txt`. Uses **BeautifulSoup** for HTML parsing and handles `src`, `data-src`, and `srcset` attributes, URL normalisation, and SVG/icon filtering automatically.

```bash
python3 automate_url_captioner.py
```

**Output format:**
```
https://upload.wikimedia.org/.../IBM_logo.png: the image of a blue ibm logo on a white background
```

### Script 3 — `image_captioning_app.py` (Gradio Web App)

Wraps the BLIP pipeline in a Gradio interface — drag-and-drop image upload, instant caption output, no command line required.

```bash
python3 image_captioning_app.py
# Open http://localhost:7860 in your browser
```

---

## Technology Stack

| Component | Tool | Role |
|-----------|------|------|
| Vision-Language Model | Salesforce BLIP (`blip-image-captioning-base`) | Generates image captions |
| ML Framework | Hugging Face Transformers 4.38.2 | Model loading and inference |
| Deep Learning Backend | PyTorch 2.2.1 | Tensor computation |
| Web Interface | Gradio 5.23.2 | Browser-based user interface |
| Web Scraping | BeautifulSoup 4 | HTML parsing for URL captioner |
| Image Processing | Pillow | Image loading and RGB conversion |
| Containerisation | Docker | Container image packaging |
| Cloud Platform | IBM Code Engine | Serverless cloud deployment |
| Container Registry | IBM Cloud Container Registry | Stores the built container image |

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip
- IBM Cloud account (for cloud deployment only)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-image-captioning.git
cd ai-image-captioning

# 2. Create and activate a virtual environment
pip3 install virtualenv
virtualenv my_env
source my_env/bin/activate        # Linux / macOS
# my_env\Scripts\activate         # Windows

# 3. Install dependencies
pip install gradio==5.23.2 transformers==4.38.2 \
            torch==2.2.1 requests==2.31.0 bs4==0.0.2
```

> The BLIP model weights (~990 MB) are downloaded automatically on first run.

---

## Cloud Deployment (IBM Code Engine)

A full step-by-step deployment walkthrough is documented in [`AI_Image_Captioning_Report.md`](./AI_Image_Captioning_Report.md#8-deployment-with-ibm-code-engine). The summary is:

```bash
# 1. Create the build configuration
ibmcloud ce build create --name build-local-dockerfile1 \
    --build-type local --size large \
    --image us.icr.io/${SN_ICR_NAMESPACE}/myapp1 \
    --registry-secret icr-secret

# 2. Build and push the container image (~3–5 min)
ibmcloud ce buildrun submit --name buildrun-local-dockerfile1 \
    --build build-local-dockerfile1 --source .

# 3. Deploy the application
ibmcloud ce application create --name demo1 \
    --image us.icr.io/${SN_ICR_NAMESPACE}/myapp1 \
    --registry-secret icr-secret --es 2G \
    --port 7860 --minscale 1

# 4. Get the public URL
ibmcloud ce app get --name demo1 --output url
```

---

## Business Applications

| Industry | Use Case |
|----------|----------|
| News & Media | Auto-generate alt text for article images — improves accessibility and SEO |
| E-commerce | Generate product descriptions from product photos |
| Healthcare | Label and document medical imaging for records |
| Security | Real-time text descriptions of CCTV footage |
| Archives & Libraries | Auto-catalogue and tag historical photo collections |
| Social Media | Generate captions for scheduled visual content |

---

## Key Results

- BLIP generates grammatically coherent captions for natural photographs, illustrations, and screenshots.
- The URL captioner handles relative URLs, `srcset` attributes, SVG filtering, and corrupt file errors gracefully.
- The Gradio app launches locally in seconds with a single command.
- IBM Code Engine deployment produces a permanent public HTTPS URL with no server management required.

---

## Full Report

The complete technical report — covering the BLIP model architecture, annotated code walkthroughs for all three scripts, business scenario analysis, results, conclusions, and the full deployment guide — is available in two formats:

- [`AI_Image_Captioning_Report.md`](./AI_Image_Captioning_Report.md) — renders natively on GitHub
- [`AI_Image_Captioning_Report.pdf`](./AI_Image_Captioning_Report.pdf) — downloadable document

---

## References

- Hugging Face BLIP Model: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.* ICML 2022.
- [Gradio Documentation](https://gradio.app/docs)
- [IBM Code Engine Documentation](https://cloud.ibm.com/docs/codeengine)
- IBM Skills Network — *Give Meaningful Names to Your Photos with AI* (Guided Project)

---

*Jack Pumpuni Frimpong-Manso — Environmental Data Scientist | AI Engineering | Bremen, Germany*

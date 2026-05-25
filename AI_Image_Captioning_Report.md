# AI Image Captioning App with BLIP and Gradio

**Author:** Jack Pumpuni Frimpong-Manso  
**Date:** 2026  
**Lab License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)  
**Source:** IBM Skills Network — Guided Project

---

## Table of Contents

1. [Introduction and Objectives](#1-introduction-and-objectives)
2. [Environment Setup](#2-environment-setup)
3. [The BLIP Model — How It Works](#3-the-blip-model--how-it-works)
4. [Code Walkthrough](#4-code-walkthrough)
   - 4.1 [Script 1 — Single Image Captioning (`image_cap.py`)](#41-script-1--single-image-captioning-image_cappy)
   - 4.2 [Script 2 — Automated URL Captioner (`automate_url_captioner.py`)](#42-script-2--automated-url-captioner-automate_url_captionerpy)
   - 4.3 [Script 3 — Gradio Web App (`image_captioning_app.py`)](#43-script-3--gradio-web-app-image_captioning_apppy)
5. [Business Scenario and Use Cases](#5-business-scenario-and-use-cases)
6. [Results](#6-results)
7. [Conclusions and Next Steps](#7-conclusions-and-next-steps)
8. [Deployment with IBM Code Engine](#8-deployment-with-ibm-code-engine)
   - 8.1 [Required Files](#81-required-files)
   - 8.2 [File Contents](#82-file-contents)
   - 8.3 [Creating the Code Engine Project](#83-creating-the-code-engine-project)
   - 8.4 [Building the Container Image](#84-building-the-container-image)
   - 8.5 [Deploying the Containerised Application](#85-deploying-the-containerised-application)
   - 8.6 [Accessing the Deployed Application](#86-accessing-the-deployed-application)
   - 8.7 [Deployment Architecture Summary](#87-deployment-architecture-summary)

---

## 1. Introduction and Objectives

Images contain rich information that standard data systems and search engines cannot easily interpret. Converting visual content into machine-readable text — image captioning — bridges this gap and unlocks a broad set of practical applications.

### Why Image Captioning Matters

| Benefit | Description |
|---------|-------------|
| **Accessibility** | Helps visually impaired users understand visual content via screen readers |
| **SEO Enhancement** | Enables search engines to index image content, driving organic traffic |
| **Content Discovery** | Allows efficient analysis and categorisation of large image databases |
| **Social Media & Advertising** | Automates engaging description generation for visual content |
| **Security** | Provides real-time descriptions of activities in video footage |
| **Education and Research** | Assists in interpreting and documenting visual materials |
| **Multilingual Support** | Captions can be generated in multiple languages for international audiences |
| **Data Organisation** | Helps manage and categorise large sets of visual data |
| **Time Efficiency** | Automated captioning is significantly faster than manual efforts |
| **User Engagement** | Detailed, accurate captions make visual content more informative |

### Learning Objectives

By the end of this project, you will be able to:

- Implement an image captioning tool using the **BLIP model** from Hugging Face's Transformers library
- Use **Gradio** to build a user-friendly web interface for the captioning application
- Apply the tool in a real-world **news and media business scenario**
- Adapt the code for **automated URL-based captioning** of entire web pages

---

## 2. Environment Setup

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Internet connection (for model download on first run)

### Step-by-Step Installation

**Step 1 — Create and activate a virtual environment:**

```bash
pip3 install virtualenv
virtualenv my_env
source my_env/bin/activate        # Linux / macOS
# my_env\Scripts\activate         # Windows
```

**Step 2 — Install all required libraries:**

```bash
pip install langchain==0.1.11 \
            gradio==5.23.2 \
            transformers==4.38.2 \
            bs4==0.0.2 \
            requests==2.31.0 \
            torch==2.2.1
```

> Note: Installation takes approximately 5 minutes depending on your connection speed. The BLIP model weights (~990 MB) are downloaded automatically on first run.

### Library Reference

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.38.2 | Hugging Face model hub — BLIP processor and model |
| `gradio` | 5.23.2 | Web interface for the captioning app |
| `torch` | 2.2.1 | PyTorch backend for model inference |
| `Pillow` | — | Image loading and RGB conversion |
| `requests` | 2.31.0 | HTTP requests for URL-based scraping |
| `bs4` | 0.0.2 | BeautifulSoup — HTML parsing for URL captioner |
| `langchain` | 0.1.11 | LangChain framework (available for extension) |

---

## 3. The BLIP Model — How It Works

### What is Hugging Face Transformers?

Hugging Face is an organisation focused on natural language processing (NLP) and AI, widely known for its open-source `transformers` library. The library provides thousands of pre-trained models for tasks including translation, summarisation, text generation, and vision-language understanding. It has made state-of-the-art models (BERT, GPT-2, GPT-3, and others) accessible to researchers and developers worldwide.

### What is BLIP?

**BLIP** (Bootstrapping Language-Image Pre-training) is a vision-language model that enables computers to understand and generate natural language descriptions based on images. It is pre-trained on a large corpus of image-text pairs and can perform:

- **Image captioning** — generate a textual description of any image
- **Visual question answering (VQA)** — answer natural language questions about an image
- **Image-text retrieval** — match images with relevant text descriptions

The model used in this project is `Salesforce/blip-image-captioning-base`, available on the Hugging Face model hub.

### Key Components

**`AutoProcessor`**

The processor wraps a BLIP image processor and an OPT/T5 tokenizer into a single object. It handles both image preprocessing and text tokenization, preparing inputs for the model in the correct format.

> A **tokenizer** is a tool in NLP that breaks text into smaller units (tokens) — such as words or subwords — so that models can analyse and understand the text.

**`BlipForConditionalGeneration`**

This model class performs conditional text generation given an image and an optional text prompt. It generates a sequence of tokens that are then decoded into human-readable text. This makes it suitable for image captioning and visual question answering tasks.

### Inference Pipeline

```
Input Image
     │
     ▼
AutoProcessor ──► Tensor inputs (pixel_values + input_ids)
     │
     ▼
BlipForConditionalGeneration.generate()
     │
     ▼
Output token sequence
     │
     ▼
processor.decode() ──► Human-readable caption string
```

---

## 4. Code Walkthrough

### 4.1 Script 1 — Single Image Captioning (`image_cap.py`)

This script demonstrates the core captioning pipeline on a single local image.

```python
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Step 1: Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Step 2: Load and preprocess the image
img_path = "YOUR_IMAGE_NAME.jpeg"
image = Image.open(img_path).convert('RGB')

# Step 3: Prepare inputs for the model
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Step 4: Generate the caption
outputs = model.generate(**inputs, max_length=50)

# Step 5: Decode tokens to human-readable text
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

**Run:**
```bash
python3 image_cap.py
```

**Key design decisions:**

- `convert('RGB')` — ensures consistent 3-channel input regardless of source format (RGBA, greyscale, etc.)
- `return_tensors="pt"` — returns PyTorch tensors, required by the model
- `**inputs` — unpacks the dictionary as keyword arguments to `model.generate()`
- `skip_special_tokens=True` — removes model control tokens from the output string
- `max_length=50` — caps caption length at 50 tokens for concise output

---

### 4.2 Script 2 — Automated URL Captioner (`automate_url_captioner.py`)

This script automatically scrapes all images from a given web page URL and writes captions for each one to a `captions.txt` output file.

```python
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Target URL
url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
img_elements = soup.find_all("img")

with open("captions.txt", "w", encoding="utf-8") as caption_file:
    for idx, img_element in enumerate(img_elements, start=1):

        # Extract image URL from src, data-src, or srcset attributes
        img_url = img_element.get("src") or img_element.get("data-src")
        if not img_url and img_element.has_attr("srcset"):
            img_url = img_element["srcset"].split()[0]
        if not img_url:
            continue

        # Skip SVG vector graphics (not supported by PIL)
        if img_url.endswith(".svg") or ".svg" in img_url:
            continue

        # Normalise relative URLs to absolute
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = "https://en.wikipedia.org" + img_url
        elif not img_url.startswith("http"):
            continue

        try:
            r = requests.get(img_url, timeout=10, headers=headers)
            raw_image = Image.open(BytesIO(r.content))

            # Skip very small icons and spacer images
            if raw_image.size[0] * raw_image.size[1] < 200:
                continue

            raw_image = raw_image.convert("RGB")
            text = "the image of"
            inputs = processor(images=raw_image, text=text, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_url}: {caption}\n")
            print(f"[{idx}] Caption saved")

        except OSError:
            continue   # Skip SVG, ICO, or corrupt files PIL cannot open
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            continue
```

**Output (`captions.txt` format):**
```
https://upload.wikimedia.org/.../IBM_logo.png: the image of a blue ibm logo on a white background
https://upload.wikimedia.org/.../IBM_Rochester.jpg: the image of a large building with a blue sky
...
```

**Key engineering decisions:**

| Decision | Reason |
|----------|--------|
| `User-Agent` header | Prevents request blocking by servers that reject bot traffic |
| SVG skip logic | PIL cannot open SVG files; skipping avoids `OSError` crashes |
| Pixel area threshold `< 200` | Eliminates tracking pixels, spacer GIFs, and tiny icons |
| `srcset` fallback | Modern responsive images often omit `src` and use `srcset` instead |
| `BytesIO` wrapping | Allows PIL to open image data downloaded as bytes without writing to disk |
| `except OSError` separately | Silently skips unreadable images; other exceptions still print for debugging |

---

### 4.3 Script 3 — Gradio Web App (`image_captioning_app.py`)

This script wraps the BLIP captioning pipeline in a Gradio web interface, allowing any user to upload an image and receive a caption through a browser.

```python
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Gradio passes images as NumPy arrays; convert to PIL RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Tokenize and preprocess
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # Generate caption tokens
    outputs = model.generate(**inputs, max_length=50)

    # Decode to string
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Build the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="A simple web app for generating captions for images using the BLIP model."
)

iface.launch()
```

**Run:**
```bash
python3 image_captioning_app.py
```

**Gradio Interface Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fn` | `caption_image` | The function to wrap with a UI |
| `inputs` | `gr.Image()` | Image upload component; returns NumPy array |
| `outputs` | `"text"` | Text display component for the caption |
| `title` | `"Image Captioning"` | Displayed at the top of the web page |
| `description` | String | Instructional text shown below the title |

> **Note:** Gradio passes uploaded images as NumPy arrays to the function, which is why `Image.fromarray(input_image)` is needed before processing — unlike Script 1 which loads directly from a file path.

---

## 5. Business Scenario and Use Cases

### News and Media Agency

A news agency publishing hundreds of articles daily faces a significant bottleneck when manually writing image captions for every visual. Integrating this captioning pipeline transforms the workflow:

```
Journalist selects article images
           │
           ▼
Images fed into captioning program (automate_url_captioner.py or Gradio app)
           │
           ▼
Suggested captions generated automatically
           │
           ▼
Editor reviews and approves / refines captions
           │
           ┌──────────────────────┬──────────────────────┐
           ▼                      ▼                      ▼
    Alt text added           SEO keywords           Caption published
    to online article        indexed by             below image
    (Accessibility)          search engines
```

**Dual business impact:**

**Enhanced Accessibility**
Approved captions are integrated as `alt` text for images in online articles. Visually impaired readers using screen readers can understand the context of every image, ensuring a comparable content experience and adherence to inclusive design principles (WCAG guidelines).

**Improved SEO**
Search engines such as Google index image `alt` text during page crawling. Relevant, keyword-rich captions improve the article's position in image and web search results, driving organic traffic without additional advertising spend.

### Additional Industries

| Industry | Application |
|----------|-------------|
| **E-commerce** | Auto-generate product descriptions from product images |
| **Healthcare** | Assist in documenting medical images for records |
| **Security & Surveillance** | Real-time textual description of CCTV footage events |
| **Social Media Management** | Generate post descriptions for scheduled visual content |
| **Archive & Libraries** | Automatically catalogue and tag historical photo collections |
| **Education** | Generate descriptive labels for scientific and educational images |

---

## 6. Results

### Script Outputs

**`image_cap.py` — Single image:**

The model receives a local image and a short text prompt (`"the image of"`) and produces a natural language caption of up to 50 tokens, printed to the terminal.

Example output:
```
the image of a dog sitting on a wooden floor next to a window
```

**`automate_url_captioner.py` — Batch URL captioning:**

All images found on the target webpage (after filtering SVGs, tiny icons, and corrupt files) are captioned and written to `captions.txt`. Each line follows the format:

```
<image_url>: <generated caption>
```

A status message `[N] Caption saved` is printed for each successfully processed image, and errors are reported without halting the full run.

**`image_captioning_app.py` — Gradio web interface:**

The web application launches on `http://localhost:7860` (or the provided CloudIDE URL). Users interact through:

- An image upload box (drag-and-drop or file browser)
- A text output field displaying the generated caption
- A Submit button to trigger inference

The interface is accessible in any modern browser with no additional configuration.

### Model Performance Notes

- The `blip-image-captioning-base` model generates grammatically coherent captions for most natural photographs, illustrations, and screenshots.
- Performance is strongest on images with clear subjects (people, objects, landscapes) and weaker on abstract graphics, charts, and diagrams.
- Caption quality improves with higher-resolution input images.
- The optional `Salesforce/blip2-opt-2.7b` (BLIP-2) model offers significantly improved captioning quality at the cost of approximately 10 GB of storage and higher compute requirements.

---

## 7. Conclusions and Next Steps

### Key Takeaways

- The **BLIP model** from Hugging Face provides a powerful, ready-to-use vision-language pipeline that requires no training from scratch.
- **Gradio** makes deploying an ML model as an interactive web application a matter of a few lines of code, dramatically lowering the barrier to sharing AI tools.
- **BeautifulSoup** enables scalable automation: rather than captioning images one at a time, the URL captioner processes entire web pages in a single run.
- The pipeline is modular — each script builds on the same processor/model core and can be adapted independently.

### Limitations

- The dataset used to pre-train BLIP is not domain-specific; captions for highly specialised imagery (medical scans, satellite images, engineering schematics) may be generic.
- The current URL captioner operates synchronously; processing pages with many images is slow. Asynchronous or parallel downloading would improve throughput.
- The Gradio app runs locally by default; making it publicly accessible requires deployment to a cloud platform.
- The `base` model variant prioritises speed over accuracy; the `large` or BLIP-2 variants produce richer captions but require more compute.

### Next Steps

1. **Deploy to the cloud** — Use IBM Code Engine, Hugging Face Spaces, or a containerised Kubernetes cluster to give the app a permanent public URL.
2. **Upgrade to BLIP-2** — Switch to `Salesforce/blip2-opt-2.7b` for higher-quality captions on complex images (requires GPU and ~10 GB storage).
3. **Add multilingual output** — Pipe the English caption through a translation model (e.g. Helsinki-NLP MarianMT) to generate captions in multiple languages.
4. **Batch processing** — Extend Script 2 with `concurrent.futures.ThreadPoolExecutor` to download and caption multiple images in parallel.
5. **Fine-tuning** — Fine-tune BLIP on a domain-specific dataset (e.g. medical imaging or satellite imagery) to improve caption relevance for specialised applications.
6. **API wrapper** — Wrap the captioning function in a FastAPI endpoint to allow integration into existing content management systems (CMS) or editorial workflows.

---

## 8. Deployment with IBM Code Engine

IBM Code Engine is a fully managed, serverless platform that runs containerised workloads — web apps, microservices, event-driven functions, and batch jobs — all hosted within the same Kubernetes infrastructure. It lets you focus entirely on writing code, not on managing servers.

### 8.1 Required Files

Three files must exist in your `myapp/` directory before building a container image:

| File | Purpose |
|------|---------|
| `demo.py` | Gradio application — the Python script that launches the web interface |
| `requirements.txt` | All Python library dependencies the app needs |
| `Dockerfile` | Blueprint that instructs the container runtime how to assemble the image |

**Create the directory and files:**

```bash
mkdir myapp && cd myapp
touch demo.py Dockerfile requirements.txt
```

---

### 8.2 File Contents

#### `requirements.txt`

```
requests
gradio==5.23.2
```

Test locally before containerising:

```bash
pip3 install -r requirements.txt
```

---

#### `demo.py`

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(server_name="0.0.0.0", server_port=7860)
```

Test locally:

```bash
python3 demo.py
```

The app will be available at `http://0.0.0.0:7860/`. Press `Ctrl+C` to stop.

---

#### `Dockerfile`

```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "demo.py"]
```

**What each instruction does:**

| Instruction | Explanation |
|-------------|-------------|
| `FROM python:3.10` | Inherits the official Python 3.10 base image with all required Python tooling |
| `WORKDIR /app` | Sets `/app` as the default working directory for all subsequent commands |
| `COPY requirements.txt` | Copies the dependencies file into the image before installing |
| `RUN pip3 install ...` | Installs all listed dependencies into the image (no cache to reduce image size) |
| `COPY . .` | Copies all remaining source files (including `demo.py`) into the image |
| `CMD ["python", "demo.py"]` | Defines the command that runs when the container starts |

---

### 8.3 Creating the Code Engine Project

IBM Code Engine projects group applications, jobs, and builds together and manage resource access. In the IBM Skills Network environment a project is pre-configured. Launch the Code Engine CLI from the IDE and verify your project:

```bash
ibmcloud ce project get --name "YOUR_PROJECT_NAME"
```

---

### 8.4 Building the Container Image

**Step 1 — Navigate to the source directory:**

```bash
cd myapp
```

**Step 2 — Create the build configuration:**

```bash
ibmcloud ce build create --name build-local-dockerfile1 \
    --build-type local --size large \
    --image us.icr.io/${SN_ICR_NAMESPACE}/myapp1 \
    --registry-secret icr-secret
```

| Argument | Value | Purpose |
|----------|-------|---------|
| `--name` | `build-local-dockerfile1` | Name of the build configuration |
| `--build-type` | `local` | Source is the local directory, not a Git repo |
| `--size` | `large` | Allocates more CPU, memory, and disk for large model pipelines |
| `--image` | `us.icr.io/${SN_ICR_NAMESPACE}/myapp1` | Target registry and image tag |
| `--registry-secret` | `icr-secret` | Credentials for pushing to IBM Cloud Container Registry |

**Step 3 — Submit and run the build:**

```bash
ibmcloud ce buildrun submit --name buildrun-local-dockerfile1 \
    --build build-local-dockerfile1 \
    --source .
```

> This packs your local source into an archive, uploads it to IBM Cloud Container Registry, and builds the image. Allow approximately 3–5 minutes to complete.

**Step 4 — Monitor build progress:**

```bash
ibmcloud ce buildrun get -n buildrun-local-dockerfile1
```

When `Status` shows `Succeeded`, the container image has been built and pushed to your registry namespace.

---

### 8.5 Deploying the Containerised Application

With the image in the registry, deploy the app with a single command:

```bash
ibmcloud ce application create --name demo1 \
    --image us.icr.io/${SN_ICR_NAMESPACE}/myapp1 \
    --registry-secret icr-secret \
    --es 2G \
    --port 7860 \
    --minscale 1
```

**Argument reference:**

| Argument | Value | Reason |
|----------|-------|--------|
| `--name` | `demo1` | Name of the deployed application |
| `--image` | `us.icr.io/.../myapp1` | Container image to pull and run |
| `--registry-secret` | `icr-secret` | Registry access credentials |
| `--es` | `2G` | Ephemeral storage — 2 GB needed for large model files |
| `--port` | `7860` | Gradio's default port; public traffic is routed here |
| `--minscale` | `1` | Keeps at least one instance running; prevents cold-start delays |

Allow 3–5 minutes for deployment to complete.

---

### 8.6 Accessing the Deployed Application

Retrieve the public URL of your running app:

```bash
ibmcloud ce app get --name demo1 --output url
```

Open the returned URL in any browser to access the live Gradio application. The app is now publicly accessible without any local server running.

For a custom domain, IBM Code Engine supports Cloudflare-based domain configuration — see the IBM Cloud documentation on configuring custom domains for Code Engine applications.

---

### 8.7 Deployment Architecture Summary

```
Local Source Files (myapp/)
    demo.py | requirements.txt | Dockerfile
           │
           ▼
ibmcloud ce build create        ← Define build configuration
           │
           ▼
ibmcloud ce buildrun submit     ← Pack, upload, and build image
           │
           ▼
IBM Cloud Container Registry    ← Stores built container image
  us.icr.io/${NAMESPACE}/myapp1
           │
           ▼
ibmcloud ce application create  ← Pull image and deploy app
           │
           ▼
Public URL (HTTPS)              ← Accessible from any browser worldwide
```

---

## Repository Structure

```
ai-image-captioning/
│
├── myapp/
│   ├── demo.py                       # Gradio app for Code Engine deployment
│   ├── requirements.txt              # Runtime dependencies
│   └── Dockerfile                    # Container image blueprint
│
├── image_cap.py                      # Script 1: Single image captioning
├── automate_url_captioner.py         # Script 2: Automated URL-based captioning
├── image_captioning_app.py           # Script 3: Gradio web application
├── captions.txt                      # Output: generated captions from URL captioner
└── README.md                         # This report
```

---

## References

- Hugging Face BLIP Model: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Hugging Face Transformers Documentation: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Gradio Documentation: [gradio.app/docs](https://gradio.app/docs)
- Original Paper: Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.* ICML 2022.
- IBM Skills Network Guided Project — *Give Meaningful Names to Your Photos with AI*
- Lab License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

*Jack Pumpuni Frimpong-Manso — Environmental Data Scientist | AI Engineering | Bremen, Germany*

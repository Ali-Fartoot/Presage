Sure, here's the improved version of your README file:

---

# Presage

Presage is an API-driven application utilizing **FastAPI** and **Large Language Models (LLMs)** to perform fortune-telling based on users' palm images. It leverages `llm-cpp-server` and `Lang-Segment-Anything (SAM-Lang)` to process hand images and generate insightful analyses.

## Getting Started

To run the application, follow these steps:

### Step One: Environment Setup

Create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step Two: Model Downloads

1. **Download the LLM Model and CLIP-like Model in GGUF Format:**

   - The default LLM model is **MiniCPM-V-2_6**.
   - Rename the LLM model to `ggml.gguf` and the CLIP model to `clip.gguf`.
   - Place both models in a folder named `models` at the root of the project.

2. **Download and Install SAM-Lang:**

   ```bash
   mkdir models
   cd models
   git clone https://github.com/luca-medeiros/lang-segment-anything
   cd lang-segment-anything
   pip install -e .
   ```

## Running the Application

After completing the preparation steps, you can start the LLM server and the API server using the following command:

```bash
make run USE_GPU=false
```

This command will start both servers:

- **FastAPI server** on port **8000**
- **llm-cpp-server** on port **5333**

To run the LLM model with GPU acceleration (using half of the model layers on the GPU), use:

```bash
make run USE_GPU=true
```

> **Note:** Ensure that your system has a compatible GPU and the necessary drivers installed to utilize GPU acceleration.

## Testing the Application

To run unit tests and end-to-end tests, execute:

```bash
make test
```

This command will perform both component-level tests and full pipeline tests to ensure the application is functioning correctly.

You can now send requests to `http://localhost:8000` or use the `./run.sh` script to send a request using `curl`.

## How It Works

The following UML diagram illustrates the project's workflow:

![UML Diagram](./example/UML.jpg)

1. **Validation:** The program first validates whether the provided image contains a hand.
2. **Segmentation:** It segments the hand from the image, removing other parts to improve performance for both the LLM (as the fortune teller) and the edge detector.
3. **Edge Detection:** An edge detector draws lines over the hand in the image, highlighting the palm lines.
4. **Fortune Telling:** The processed hand image is input into the LLM to generate the fortune-telling analysis.

## Result

Below is the final result of the image processing:

![Final Image](./example/final.jpeg)

At the end, the application returns a JSON response containing the fortune-telling analysis:

```json
{
  "filename": "test.jpg",
  "analysis_result": "In examining the intricate lines of this individual's hand, it appears that they possess a complex and multifaceted nature. The numerous branching patterns suggest a life filled with diverse experiences and opportunities for personal growth.\n\nLooking into their future, I see a journey marked by both challenges and triumphs. These individuals often find themselves at crossroads where decisions must be made based on intuition rather than logic alone. Their ability to navigate through these complex situations will lead them down paths they never expected but ultimately bring about greater fulfillment in life.\n\nTheir past experiences have taught them resilience, allowing them to bounce back from setbacks stronger and more knowledgeable with each passing day. As a result of this perseverance, new doors open up for exploration and adventure as their creativity blooms like an ever-expanding garden filled with possibilities waiting to be discovered.\n\nUltimately, I predict that these individuals will achieve great success by following their heart's desires while maintaining balance in all aspects of life—emotional, spiritual, intellectual, social—and continuing on this path brings happiness beyond measure."
}
```

## Tests

The testing strategy includes two approaches:

1. **Component Testing:** Each component (as depicted in the UML diagram) is individually tested to ensure it functions correctly in isolation.
2. **End-to-End Testing:** An image is selected and sent as a request to test the entire pipeline. Running `make test` executes all tests, verifying both individual components and the integrated system.

## Timeline

The project development was completed over **5 days**, divided as follows:

- **3 days:** Module programming
- **1 day:** Writing tests
- **1 day:** Writing documentation


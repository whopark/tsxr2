Product Requirements Document (PRD): Gemini-Powered Chest X-ray Analysis & Reporting System

1. Executive Vision and Product Scope

The strategic vision for this system is the implementation of a high-reliability hybrid intelligence pipeline that integrates local, specialized vision models (TSXr) with state-of-the-art Large Multimodal Models (LMMs) like Gemini 1.5 Pro and Med-Gemini. By offloading initial lesion detection to a localized, deterministic layer and utilizing the cloud-based LMM for clinical reasoning and synthesis, we transform raw DICOM data into structured, actionable insights. This architecture is specifically designed to integrate seamlessly with existing hospital PACS (Picture Archiving and Communication Systems) and EMR (Electronic Medical Records) via a robust FastAPI backend, ensuring interoperability within established clinical workflows.

The system serves as a subordinate AI assistant intended to alleviate "Reporting Fatigue" by generating high-quality draft reports. It is explicitly positioned as a decision-support tool rather than a diagnostic replacement; the final clinical interpretation always remains the responsibility of the human specialist.

Product Positioning

Feature	Traditional CAD (Computer-Aided Detection)	Gemini-Powered Generative Reporting
Primary Output	Simple detection (bounding boxes/heatmaps)	Structured clinical reasoning and narrative summaries
Data Integration	Image data only	Image + Local AI results + Clinical Context (Symptoms/Age)
Ideal Role Division	Screening and "Finding the finding"	Reasoning and "Interpret the finding and summarize impact"
Clinical Value	Visual aid/Alerting tool	Draft report generation and decision support

This transition from simple detection to clinical reasoning is engineered to meet the rigorous demands of healthcare providers operating in high-pressure environments.


--------------------------------------------------------------------------------


2. Target Audience and Clinical Use Cases

Success in medical AI requires a focus on reducing the documentation burden and cognitive load that characterize modern radiology and general practice. By providing an "AI Draft" as the starting point, we shift the clinician’s role from a manual reporter to a high-level editor, optimizing professional throughput.

Primary Personas

* The Radiologist: Faces extreme volume and repetitive reporting tasks. This system addresses "reporting fatigue" by providing a pre-structured draft, allowing the specialist to focus on complex diagnostic nuances rather than manual transcription.
* The General Practitioner (GP): Requires a concise "clinical summary" to support referral decisions or patient communication. The system translates technical visual findings into an accessible narrative grounded in patient context.

Impact on Productivity The automated draft workflow significantly minimizes the "blank page" hurdle. By presenting a grounded clinical narrative, the system reduces the time-per-case while improving report consistency. This efficiency is achieved through a technical architecture that prioritizes safety through grounding.


--------------------------------------------------------------------------------


3. Technical Architecture: The Hybrid Pipeline

To ensure medical-grade reliability and explainability, the system adopts a hybrid architecture (Local Vision + Cloud LMM). This ensures that all generative outputs are strictly grounded in pixel-level evidence identified by the local layer. The backend is built on a FastAPI framework utilizing PyTorch for model inference and pydicom for medical imaging standard compliance.

The Three-Stage Pipeline

1. Input and Preprocessing: DICOM ingestion and normalization for model compatibility.
2. Local TSXr Analysis: A specialized CNN/ViT-based vision model (e.g., DenseNet121) performing screening and feature extraction.
3. Gemini Reasoning Layer: Gemini 1.5 Pro or Med-Gemini synthesizes the image, TSXr data, and clinical context into a structured report.

Preprocessing Requirements

* DICOM-to-Image Conversion: Utilizing pydicom to load raw data, applying Hounsfield Unit (HU) windowing for optimal contrast, and converting to PNG/JPEG.
* PHI (Protected Health Information) Removal: Mandatory anonymization of DICOM tags (Name, ID, Accession Number) to ensure HIPAA/GDPR compliance prior to cloud transmission.
* Resolution Normalization: Resizing images to a fixed 512x512 resolution with mandatory channel normalization to meet model input specifications.

The Local Vision Layer (TSXr) The TSXr layer acts as the system's primary screening mechanism. It identifies lesion candidates and generates structured "Hints" for the LMM. These hints include multi-label classification probabilities, severity levels (mild/moderate/severe), and localization data (bounding boxes or Grad-CAM heatmaps). This ensures the LLM reasoning is grounded in local analysis, effectively eliminating the risk of hallucinated findings.


--------------------------------------------------------------------------------


4. Data Schemas and Interoperability

Rigid JSON schemas serve as the "source of truth," ensuring clinical reliability and facilitating seamless EMR integration.

TSXr Output Schema (Grounding Data)

{
  "metadata": {
    "model_version": "tsxr-v2.1",
    "timestamp": "2023-10-27T10:00:00Z"
  },
  "image_info": {
    "dimensions": [512, 512],
    "view": "PA"
  },
  "findings": [
    {
      "label": "Pneumonia",
      "probability": 0.89,
      "severity": "moderate",
      "side": "right",
      "bbox": [120, 210, 160, 260]
    }
  ],
  "global_scores": {
    "abnormality_score": 0.92,
    "confidence_index": 0.88
  },
  "quality_checks": {
    "rotation": "low",
    "inspiration": "adequate"
  }
}


Gemini Final Report Schema (Draft Output)

{
  "findings": "Structured description of visual findings based on TSXr hints.",
  "impression": "Clinical summary and primary diagnostic conclusion.",
  "recommendations": "Suggested follow-up actions (e.g., CT correlation, follow-up in 6 weeks)."
}


Architectural Impact of Structured Output These schemas enforce a strict boundary for the AI’s output, preventing the inclusion of conversational filler and ensuring all data is machine-readable for Quality Management Systems (QMS).


--------------------------------------------------------------------------------


5. Gemini 1.5 Pro / Med-Gemini Logic & Prompting

The reasoning layer must balance inferential power with strict clinical constraints. For high-complexity reasoning, Med-Gemini is the preferred tier.

System Prompt Requirements

* Role Identification: "You are a specialized medical reporting assistant."
* Diagnostic Prohibition: Explicitly forbid providing a "Final Diagnosis."
* Output Constraint: Mandate Strict JSON Only. Forbid the use of Markdown code blocks (e.g., ` ` `json) to ensure raw parsing success.

User Prompt Structure The multimodal input includes:

1. X-ray Image: The 512x512 preprocessed image.
2. TSXr JSON Results: Detailed lesion hints and quality scores.
3. Patient Context: Age, gender, and primary symptoms (e.g., "70yo Male, chronic cough").

Safety Constraints The prompt mandates a cautious tone (e.g., "findings correlate with," "suggestive of") and includes mandatory disclaimers in every output.


--------------------------------------------------------------------------------


6. Reliability Layer: JSON Validation and Retry Logic

In a clinical environment, the "Safety Net" layer is critical for maintaining system stability and clinician trust.

Three-Step Validation Pipeline

1. Error Detection (Validator): Validates the LLM response against the required schema, identifying missing keys or type mismatches.
2. Correction Prompting: If validation fails, the system sends an automated correction request back to the LLM (e.g., "Missing 'recommendations' key. Please re-generate the JSON structure correctly.")
3. Fallback: If logic fails after 3 attempts, the system provides a plain text summary or a Failure State UI.

The "status: retrying" Flag During the correction loop, the FastAPI backend must return a "status": "retrying" flag. This enables the frontend to manage the UX without signaling a system failure.


--------------------------------------------------------------------------------


7. UI/UX Strategy: Professional Clinical Visualization

The UI is designed to provide clarity and decision support while actively mitigating "AI anxiety" through transparent state management.

UI States

1. Success State: Visualization of the X-ray, Grad-CAM heatmap overlays, and the structured "AI Draft" report.
2. Processing State: To prevent user alarm, the UI must not display "Error" or "Failure" messages during the auto-fix loop. Instead, the interface displays "Analyzing image details..." or "Synthesizing findings..." while the backend handles retries.
3. Failure State: Triggered only after the 3rd retry attempt fails. Displays a "Retry" button or an "AI Text Summary" (text-only fallback) to maintain workflow continuity.

Mandatory Disclaimer The following text must be prominently displayed on all output screens: "Final interpretation by a specialist is required. This is an AI-generated draft for clinical decision support."


--------------------------------------------------------------------------------


8. Quality Management and Regulatory Compliance

Audit logs are a strategic necessity for Medical Device Quality Management Systems (QMS) and continuous improvement.

Critical Logging Points The system logs the following for every transaction to the Admin/QMS dashboard:

* Raw LLM Responses: For pattern analysis of formatting errors.
* Retry Counts: To monitor system latency and model stability.
* TSXr Grounding Data: To audit the accuracy of the local vision layer.
* Prompt Versioning: Essential for clinical traceability and regulatory compliance during model updates.

Final Summary Statement This system achieves a new standard in medical AI by separating visual detection from clinical reasoning. By grounding the reasoning of Gemini in the precision of local TSXr models and wrapping the pipeline in a robust, multi-tier reliability layer, we provide a tool that significantly reduces clinician workload while maintaining the highest levels of safety and technological excellence.

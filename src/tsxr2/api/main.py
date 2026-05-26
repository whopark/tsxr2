"""TSXr2 FastAPI application."""

import base64
import io
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from tsxr2.annotation import (
    AnnotationConfig,
    annotate_other_bone_findings,
    annotate_rib_findings,
)
from tsxr2.dicom_writer import (
    add_fracture_findings_to_dicom,
    extract_findings_from_dicom,
    save_annotated_dicom,
)
from tsxr2.gemini import GeminiClient, build_report_prompt, parse_gemini_response
from tsxr2.preprocessing import load_dicom, normalize_image, remove_phi
from tsxr2.reliability import (
    assess_confidence,
    generate_fallback_report,
    validate_dicom,
    validate_image_quality,
)
from tsxr2.rib_detection import run_full_rib_analysis, load_yolo_detector
from tsxr2.rib_detection.yolo_detector import DEFAULT_MODEL_PATH as YOLO_MODEL_PATH
from tsxr2.schemas import (
    AnnotatedImageOutput,
    ConfidenceInfo,
    FullReportResponse,
    GeminiReport,
    QualityInfo,
    ReliabilityInfo,
    RibAnalysisResponse,
    TSXrOutput,
    ValidationInfo,
)
from tsxr2.tsxr import format_tsxr_output, load_tsxr_model, run_inference
from tsxr2.tsxr.model_loader import TSXrModel

# Global model cache
_model: TSXrModel | None = None
_device: torch.device | None = None
_yolo_detector = None  # Cached YOLO rib detector

# Metrics tracking
_startup_time: float | None = None
_request_counts: dict[str, int] = {
    "analyze": 0,
    "report": 0,
    "full_report": 0,
    "preprocess": 0,
    "analyze_ribs": 0,
}

# Cache for modified DICOM files (session_id -> dicom_bytes)
# In production, use Redis or similar persistent storage
_modified_dicom_cache: dict[str, bytes] = {}
_dicom_cache_counter: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model, _device, _startup_time
    _startup_time = time.time()
    _model, _device = load_tsxr_model()
    yield
    # Cleanup
    _model = None
    _device = None


API_DESCRIPTION = """
# TSXr2 - Gemini-Powered Chest X-ray Analysis & Reporting System

A hybrid AI pipeline that combines local vision model inference with
Gemini's reasoning capabilities for comprehensive chest X-ray analysis.

## Features

* **DICOM Processing** - Load, validate, and preprocess DICOM chest X-rays
* **Multi-label Classification** - Detect 14 pathological findings using DenseNet121
* **Rib Fracture Detection** - Systematic scanning L1→L10, R1→R10 with arrow annotations
* **Clinical Report Generation** - Generate structured reports via Gemini API
* **Reliability Assessment** - Confidence scoring and quality validation
* **Automatic Fallback** - Template-based reports when Gemini is unavailable

## Endpoints

### Analysis Endpoints
- `/analyze` - Run TSXr model inference only
- `/analyze-ribs` - **Rib fracture detection with arrow annotations**
- `/report` - Generate Gemini clinical report
- `/full-report` - **Complete pipeline with reliability info** (recommended)

### Utility Endpoints
- `/preprocess` - Preprocess DICOM for external use
- `/health` - Simple health check
- `/health/detailed` - Component status
- `/metrics` - System metrics

## Authentication

This API requires a valid `GEMINI_API_KEY` environment variable for
report generation endpoints.
"""

tags_metadata = [
    {
        "name": "Analysis",
        "description": "Chest X-ray analysis and report generation endpoints.",
    },
    {
        "name": "Health",
        "description": "Health checks and system monitoring.",
    },
    {
        "name": "Utility",
        "description": "Utility endpoints for preprocessing and metrics.",
    },
]

app = FastAPI(
    title="TSXr2",
    description=API_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    contact={
        "name": "TSXr2 Team",
    },
    license_info={
        "name": "MIT",
    },
)


class PreprocessResponse(BaseModel):
    """Response model for /preprocess endpoint."""

    image_base64: str
    metadata: dict[str, Any]
    anonymized: bool


def get_model() -> tuple[TSXrModel, torch.device]:
    """Get the loaded model and device."""
    global _model, _device
    if _model is None or _device is None:
        # Fallback: load model if not loaded via lifespan
        _model, _device = load_tsxr_model()
    return _model, _device


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Simple health check endpoint.

    Returns a basic status indicating the API is running.
    Use `/health/detailed` for component-level status.
    """
    return {"status": "healthy"}


@app.get("/health/detailed", tags=["Health"])
async def health_detailed() -> dict:
    """Detailed health check with component status.

    Returns status of all system components including:
    - Model loading status
    - Device information (CPU/GPU)
    - Timestamp

    Returns:
        Dict with overall status and component details.
    """
    global _startup_time

    # Ensure model is loaded (uses fallback if not loaded via lifespan)
    try:
        model, device = get_model()
        model_loaded = model is not None
        model_status = "healthy" if model_loaded else "unhealthy"
        device_info = str(device) if device else "unknown"
    except Exception:
        model_loaded = False
        model_status = "unhealthy"
        device_info = "unknown"

    # Build component status
    components = {
        "model": {
            "status": model_status,
            "loaded": model_loaded,
            "device": device_info,
        },
    }

    # Overall status is healthy only if all components are healthy
    overall_status = "healthy" if model_loaded else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }


@app.get("/metrics", tags=["Utility"])
async def get_metrics() -> dict:
    """Get system metrics and counters.

    Returns request counts, uptime, and system information.

    Returns:
        Dict with metrics data.
    """
    global _startup_time, _request_counts, _device

    uptime = time.time() - _startup_time if _startup_time else 0

    return {
        "uptime_seconds": round(uptime, 2),
        "requests": _request_counts.copy(),
        "device": str(_device) if _device else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/preprocess", response_model=PreprocessResponse, tags=["Utility"])
async def preprocess_dicom(file: UploadFile = File(...)) -> PreprocessResponse:
    """Preprocess a DICOM file for analysis.

    Accepts a DICOM file upload and returns:
    - Base64-encoded normalized image (512x512 PNG)
    - Extracted metadata (with PHI removed)
    - Anonymization status

    Args:
        file: DICOM file upload.

    Returns:
        PreprocessResponse with processed image and metadata.

    Raises:
        HTTPException: If file is invalid or processing fails.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load DICOM
        dicom_data = load_dicom(tmp_path)

        # Remove PHI from dataset (for audit logging, not returned in response)
        _ = remove_phi(dicom_data.original_dataset, keep_dates=True)

        # Normalize image for model input
        normalized = normalize_image(
            dicom_data.pixel_array,
            window_center=dicom_data.metadata.get("window_center"),
            window_width=dicom_data.metadata.get("window_width"),
        )

        # Convert to PNG and base64 encode
        img = Image.fromarray(normalized)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Extract safe metadata
        safe_metadata = {
            "modality": dicom_data.metadata.get("modality"),
            "body_part": dicom_data.metadata.get("body_part"),
            "view_position": dicom_data.metadata.get("view_position"),
            "rows": dicom_data.metadata.get("rows"),
            "columns": dicom_data.metadata.get("columns"),
            "bits_stored": dicom_data.metadata.get("bits_stored"),
        }

        return PreprocessResponse(
            image_base64=image_base64,
            metadata=safe_metadata,
            anonymized=True,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid DICOM file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/analyze", response_model=TSXrOutput, tags=["Analysis"])
async def analyze_dicom(file: UploadFile = File(...)) -> TSXrOutput:
    """Analyze a DICOM chest X-ray using the TSXr model.

    Accepts a DICOM file upload and returns:
    - TSXr model findings (multi-label classification)
    - Global abnormality and confidence scores
    - Image quality assessment

    Args:
        file: DICOM file upload.

    Returns:
        TSXrOutput with analysis results matching PRD schema.

    Raises:
        HTTPException: If file is invalid or analysis fails.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load DICOM
        dicom_data = load_dicom(tmp_path)

        # Normalize image for model input
        normalized = normalize_image(
            dicom_data.pixel_array,
            window_center=dicom_data.metadata.get("window_center"),
            window_width=dicom_data.metadata.get("window_width"),
        )

        # Get model and run inference
        model, device = get_model()
        inference_result = run_inference(model, normalized, device)

        # Format to TSXrOutput schema
        view_position = dicom_data.metadata.get("view_position", "PA") or "PA"
        output = format_tsxr_output(
            inference_result,
            image_dimensions=(512, 512),
            view_position=view_position,
        )

        return output

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid DICOM file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/report", response_model=GeminiReport, tags=["Analysis"])
async def generate_report(file: UploadFile = File(...)) -> GeminiReport:
    """Generate a clinical report from a DICOM chest X-ray.

    Full pipeline: DICOM → TSXr analysis → Gemini report generation.

    Accepts a DICOM file upload and returns:
    - Clinical findings narrative
    - Impression summary
    - Recommendations

    Args:
        file: DICOM file upload.

    Returns:
        GeminiReport with clinical narrative.

    Raises:
        HTTPException: If file is invalid, analysis fails, or Gemini unavailable.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load DICOM
        dicom_data = load_dicom(tmp_path)

        # Normalize image for model input
        normalized = normalize_image(
            dicom_data.pixel_array,
            window_center=dicom_data.metadata.get("window_center"),
            window_width=dicom_data.metadata.get("window_width"),
        )

        # Get model and run inference
        model, device = get_model()
        inference_result = run_inference(model, normalized, device)

        # Format to TSXrOutput schema
        view_position = dicom_data.metadata.get("view_position", "PA") or "PA"
        tsxr_output = format_tsxr_output(
            inference_result,
            image_dimensions=(512, 512),
            view_position=view_position,
        )

        # Build prompt and call Gemini
        prompt = build_report_prompt(tsxr_output)
        gemini_client = GeminiClient()
        response_text = gemini_client.generate(prompt)

        # Parse response into GeminiReport
        report = parse_gemini_response(response_text)

        return report

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid DICOM file")
    except ValueError as e:
        # Gemini API key not configured
        raise HTTPException(status_code=503, detail=f"Gemini service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/full-report", response_model=FullReportResponse, tags=["Analysis"])
async def generate_full_report(file: UploadFile = File(...)) -> FullReportResponse:
    """Generate a comprehensive analysis report from a DICOM chest X-ray.

    Full pipeline with reliability assessment:
    1. Validate DICOM file
    2. Run TSXr model analysis
    3. Assess confidence and quality
    4. Generate Gemini report (with fallback if unavailable)

    Args:
        file: DICOM file upload.

    Returns:
        FullReportResponse with TSXr output, Gemini report, and reliability info.

    Raises:
        HTTPException: If file is invalid or processing fails.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Step 1: Validate DICOM file
        validation_result = validate_dicom(tmp_path)
        validation_info = ValidationInfo(
            is_valid=validation_result.is_valid,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
        )

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid DICOM file: {'; '.join(validation_result.errors)}",
            )

        # Step 2: Load and preprocess DICOM
        dicom_data = load_dicom(tmp_path)
        normalized = normalize_image(
            dicom_data.pixel_array,
            window_center=dicom_data.metadata.get("window_center"),
            window_width=dicom_data.metadata.get("window_width"),
        )

        # Step 3: Run TSXr model inference
        model, device = get_model()
        inference_result = run_inference(model, normalized, device)

        view_position = dicom_data.metadata.get("view_position", "PA") or "PA"
        tsxr_output = format_tsxr_output(
            inference_result,
            image_dimensions=(512, 512),
            view_position=view_position,
        )

        # Step 4: Assess confidence and quality
        confidence_assessment = assess_confidence(
            confidence_index=tsxr_output.global_scores.confidence_index,
            abnormality_score=tsxr_output.global_scores.abnormality_score,
        )
        quality_validation = validate_image_quality(tsxr_output.quality_checks)

        confidence_info = ConfidenceInfo(
            level=confidence_assessment.level,
            warnings=confidence_assessment.warnings,
            requires_review=confidence_assessment.requires_review,
        )
        quality_info = QualityInfo(
            is_acceptable=quality_validation.is_acceptable,
            issues=quality_validation.issues,
        )

        # Step 5: Generate Gemini report (with fallback)
        used_fallback = False
        try:
            prompt = build_report_prompt(tsxr_output)
            gemini_client = GeminiClient()
            response_text = gemini_client.generate(prompt)
            gemini_report = parse_gemini_response(response_text)
        except Exception:
            # Use fallback report generation
            gemini_report = generate_fallback_report(tsxr_output)
            used_fallback = True

        # Build reliability info
        reliability_info = ReliabilityInfo(
            confidence=confidence_info,
            quality=quality_info,
            used_fallback=used_fallback,
        )

        return FullReportResponse(
            tsxr_output=tsxr_output.model_dump(),
            gemini_report=gemini_report.model_dump(),
            reliability=reliability_info,
            validation=validation_info,
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid DICOM file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full report generation failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/analyze-ribs", response_model=RibAnalysisResponse, tags=["Analysis"])
async def analyze_ribs(
    file: UploadFile = File(...),
    include_annotations: bool = True,
    show_intact_ribs: bool = False,
    embed_in_dicom: bool = False,
    use_yolo: bool = True,
) -> RibAnalysisResponse:
    """Analyze a DICOM chest X-ray for rib fractures with systematic scanning.

    Performs systematic rib fracture detection following clinical protocol:
    1. Scan left ribs: L1 → L10
    2. Scan right ribs: R1 → R10
    3. Check other bones (clavicle, scapula, spine)

    For each detected fracture, adds arrow annotations pointing to the
    fracture location on the image.

    Args:
        file: DICOM file upload.
        include_annotations: If True, return annotated image with arrows.
        show_intact_ribs: If True, annotate all ribs (not just fractures).
        embed_in_dicom: If True, embed findings in DICOM private tags.
            Use /analyze-ribs/dicom to download the modified DICOM file.
        use_yolo: If True (default), use YOLOv8 model for real detection.
            Falls back to simulation if model not available.

    Returns:
        RibAnalysisResponse with:
        - Systematic scan results for each rib
        - Detected fractures with confidence scores
        - Annotated image (if requested)
        - dicom_modified flag indicating if embedding was performed
        - Processing metrics

    Raises:
        HTTPException: If file is invalid or analysis fails.
    """
    global _request_counts, _yolo_detector
    _request_counts["analyze_ribs"] += 1

    start_time = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load DICOM
        dicom_data = load_dicom(tmp_path)

        # Normalize image for analysis
        normalized = normalize_image(
            dicom_data.pixel_array,
            window_center=dicom_data.metadata.get("window_center"),
            window_width=dicom_data.metadata.get("window_width"),
        )

        # Run rib fracture detection
        # Try YOLO model first if requested, fall back to simulation
        use_simulation = True
        yolo_detector_to_use = None

        if use_yolo and YOLO_MODEL_PATH.exists():
            try:
                # Load YOLO detector if not cached
                if _yolo_detector is None:
                    _yolo_detector = load_yolo_detector(
                        model_path=YOLO_MODEL_PATH,
                        conf_threshold=0.01,  # Low threshold for medical imaging
                    )
                yolo_detector_to_use = _yolo_detector
                use_simulation = False
            except Exception:
                # Fall back to simulation if YOLO fails to load
                pass

        rib_analysis = run_full_rib_analysis(
            model=None,
            image=normalized,
            device=None,
            use_simulation=use_simulation,
            yolo_detector=yolo_detector_to_use,
        )

        # Prepare response
        annotated_output = None

        if include_annotations:
            # Convert normalized array to PIL Image
            pil_image = Image.fromarray(normalized)

            # Configure annotation rendering
            config = AnnotationConfig(
                show_intact_ribs=show_intact_ribs,
                show_labels=True,
            )

            # Annotate rib fractures (and suspicious ribs)
            findings_to_annotate = [
                f for f in rib_analysis.rib_findings
                if f.fracture_status in ("fractured", "suspicious") or show_intact_ribs
            ]

            annotated_image, annotations = annotate_rib_findings(
                pil_image,
                findings_to_annotate,
                config,
            )

            # Also annotate other bone findings (clavicle, scapula)
            if rib_analysis.other_bone_findings:
                # Build fracture status map from simulated detections
                from tsxr2.rib_detection import simulate_other_bone_detection
                other_bone_detections = simulate_other_bone_detection(normalized)
                fracture_status_map = {
                    f"{d.bone_name}_{d.side}": d.fracture_status
                    for d in other_bone_detections
                }

                annotated_image, other_annotations = annotate_other_bone_findings(
                    annotated_image,  # Use already annotated image
                    rib_analysis.other_bone_findings,
                    config,
                    fracture_status_map,
                )
                annotations.extend(other_annotations)

            # Encode images to base64
            original_buffer = io.BytesIO()
            pil_image.save(original_buffer, format="PNG")
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode("utf-8")

            annotated_buffer = io.BytesIO()
            annotated_image.save(annotated_buffer, format="PNG")
            annotated_base64 = base64.b64encode(annotated_buffer.getvalue()).decode("utf-8")

            annotated_output = AnnotatedImageOutput(
                original_image_base64=original_base64,
                annotated_image_base64=annotated_base64,
                annotations=annotations,
                image_dimensions=(pil_image.width, pil_image.height),
            )

        # Embed findings in DICOM if requested
        dicom_modified = False
        dicom_session_id = None

        if embed_in_dicom:
            global _modified_dicom_cache, _dicom_cache_counter
            _dicom_cache_counter += 1
            dicom_session_id = f"rib_analysis_{_dicom_cache_counter}"

            # Get annotations for embedding
            embed_annotations = annotations if include_annotations else None

            # Save modified DICOM to temporary file
            modified_dicom_path = tmp_path.parent / f"modified_{tmp_path.name}"
            save_annotated_dicom(
                original_dataset=dicom_data.original_dataset,
                output_path=modified_dicom_path,
                analysis_output=rib_analysis,
                annotations=embed_annotations,
                new_series=True,
            )

            # Cache the modified DICOM bytes
            with open(modified_dicom_path, "rb") as f:
                _modified_dicom_cache[dicom_session_id] = f.read()

            # Clean up modified file
            if modified_dicom_path.exists():
                modified_dicom_path.unlink()

            dicom_modified = True

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return RibAnalysisResponse(
            rib_analysis=rib_analysis,
            annotated_image=annotated_output,
            dicom_modified=dicom_modified,
            dicom_session_id=dicom_session_id,
            processing_time_ms=processing_time_ms,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid DICOM file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rib analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/analyze-ribs/dicom/{session_id}", tags=["Analysis"])
async def download_modified_dicom(session_id: str) -> Response:
    """Download DICOM file with embedded AI findings.

    After calling /analyze-ribs with embed_in_dicom=True, use the returned
    session_id to download the modified DICOM file.

    The modified DICOM includes:
    - Private tags (0x0099 block) with JSON-encoded findings
    - ImageComments field with human-readable summary
    - New Series/SOP Instance UIDs

    Args:
        session_id: Session ID returned from /analyze-ribs.

    Returns:
        DICOM file as binary response.

    Raises:
        HTTPException: If session_id is invalid or expired.
    """
    global _modified_dicom_cache

    if session_id not in _modified_dicom_cache:
        raise HTTPException(
            status_code=404,
            detail=f"DICOM session '{session_id}' not found or expired. "
            "Call /analyze-ribs with embed_in_dicom=True first.",
        )

    dicom_bytes = _modified_dicom_cache[session_id]

    # Optional: remove from cache after download to prevent memory bloat
    # del _modified_dicom_cache[session_id]

    return Response(
        content=dicom_bytes,
        media_type="application/dicom",
        headers={
            "Content-Disposition": f"attachment; filename={session_id}.dcm",
        },
    )


@app.get("/analyze-ribs/extract/{session_id}", tags=["Analysis"])
async def extract_dicom_findings(session_id: str) -> dict:
    """Extract AI findings from a cached modified DICOM.

    Reads the private tags from the modified DICOM and returns
    the extracted findings as JSON.

    Args:
        session_id: Session ID returned from /analyze-ribs.

    Returns:
        Dict containing extracted findings from DICOM private tags.

    Raises:
        HTTPException: If session_id is invalid or no findings present.
    """
    import pydicom

    global _modified_dicom_cache

    if session_id not in _modified_dicom_cache:
        raise HTTPException(
            status_code=404,
            detail=f"DICOM session '{session_id}' not found or expired.",
        )

    # Load DICOM from cached bytes
    dicom_bytes = _modified_dicom_cache[session_id]
    dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))

    # Extract findings using our utility
    findings = extract_findings_from_dicom(dataset)

    if findings is None:
        raise HTTPException(
            status_code=404,
            detail="No TSXr2 AI findings found in DICOM metadata.",
        )

    return {
        "session_id": session_id,
        "findings": findings,
        "image_comments": getattr(dataset, "ImageComments", None),
    }
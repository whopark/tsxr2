"""Test TSXr local vision model components."""

import torch


def test_load_model_returns_densenet():
    """load_tsxr_model should return a DenseNet121 model in eval mode."""
    from tsxr2.tsxr import load_tsxr_model

    model, device = load_tsxr_model()

    # Should be a DenseNet model
    assert "densenet" in model.__class__.__name__.lower() or hasattr(model, "features")
    # Should be in eval mode
    assert not model.training
    # Device should be returned
    assert device in [torch.device("cuda"), torch.device("cpu")]


def test_model_has_correct_output_shape():
    """TSXr model should output 14 class probabilities."""
    from tsxr2.tsxr import load_tsxr_model

    model, device = load_tsxr_model()

    # Create dummy input (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Should output 14 probabilities
    assert output.shape == (1, 14)
    # Probabilities should be in [0, 1] due to sigmoid
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_model_accepts_512x512_input():
    """TSXr model should accept 512x512 input images."""
    from tsxr2.tsxr import load_tsxr_model

    model, device = load_tsxr_model()

    # Create 512x512 input (common medical imaging size)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Should still output 14 probabilities
    assert output.shape == (1, 14)


# --- Inference Pipeline Tests ---


def test_run_inference_returns_predictions():
    """run_inference should return prediction dict with probabilities."""
    import numpy as np

    from tsxr2.tsxr import load_tsxr_model, run_inference

    model, device = load_tsxr_model()

    # Create a dummy normalized image (512x512x3 uint8, like from preprocessing)
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    result = run_inference(model, dummy_image, device)

    # Should return dict with predictions
    assert "probabilities" in result
    assert "labels" in result
    assert len(result["probabilities"]) == 14
    assert len(result["labels"]) == 14
    # Probabilities should be in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in result["probabilities"])


def test_run_inference_returns_global_scores():
    """run_inference should return abnormality_score and confidence_index."""
    import numpy as np

    from tsxr2.tsxr import load_tsxr_model, run_inference

    model, device = load_tsxr_model()
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    result = run_inference(model, dummy_image, device)

    assert "abnormality_score" in result
    assert "confidence_index" in result
    assert 0.0 <= result["abnormality_score"] <= 1.0
    assert 0.0 <= result["confidence_index"] <= 1.0


def test_run_inference_filters_by_threshold():
    """run_inference should only return findings above threshold."""
    import numpy as np

    from tsxr2.tsxr import load_tsxr_model, run_inference

    model, device = load_tsxr_model()
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # High threshold should filter out most findings
    result = run_inference(model, dummy_image, device, threshold=0.9)

    # All returned findings should be above threshold
    for finding in result["findings"]:
        assert finding["probability"] >= 0.9


# --- Output Formatter Tests ---


def test_format_tsxr_output_returns_valid_schema():
    """format_tsxr_output should return a valid TSXrOutput schema."""
    import numpy as np

    from tsxr2.schemas import TSXrOutput
    from tsxr2.tsxr import format_tsxr_output, load_tsxr_model, run_inference

    model, device = load_tsxr_model()
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    inference_result = run_inference(model, dummy_image, device)
    output = format_tsxr_output(
        inference_result,
        image_dimensions=(512, 512),
        view_position="PA",
    )

    # Should be a valid TSXrOutput instance
    assert isinstance(output, TSXrOutput)
    assert output.metadata.model_version is not None
    assert output.image_info.dimensions == (512, 512)
    assert output.image_info.view == "PA"
    assert output.global_scores.abnormality_score >= 0.0

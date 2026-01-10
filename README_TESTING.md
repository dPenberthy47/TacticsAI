# TacticsAI Testing Guide

This guide covers testing and validation for the TacticsAI project.

## Quick Validation

For a fast sanity check that all components are working:

```bash
python scripts/validate_pipeline.py
```

This runs 10 essential checks:
- ✓ Core imports
- ✓ Dataset creation
- ✓ Enhanced features (12D nodes, 6D edges)
- ✓ GAT model architecture
- ✓ Model forward pass
- ✓ Attention-based zones
- ✓ Prediction service
- ✓ Battle zones determinism
- ✓ Insights generator
- ✓ Tactical dominance calculation

**Expected output:** All checks should pass with ✓

## Comprehensive Testing

### Run All Tests

```bash
cd backend
pytest
```

### Run with Coverage

```bash
pytest --cov=backend --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Run Specific Test Classes

```bash
# Test data sources only
pytest tests/test_integration.py::TestDataSources -v

# Test model architecture only
pytest tests/test_integration.py::TestModelArchitecture -v

# Test prediction service only
pytest tests/test_integration.py::TestPredictionService -v
```

### Run Tests by Marker

```bash
# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests (if available)
pytest -m gpu
```

## Test Organization

### Test Categories

**Integration Tests** (`tests/test_integration.py`):
- End-to-end pipeline validation
- Multi-component interaction
- Real data source connectivity

**Test Classes**:
- `TestDataSources` - Validate all 4 data fetchers
- `TestEnhancedFeatures` - Check 12D nodes, 6D edges
- `TestModelArchitecture` - Verify GAT layers, attention
- `TestPredictionService` - End-to-end predictions
- `TestInsightsGeneration` - LLM and fallback insights
- `TestDatasetCreation` - Dataset from all sources
- `TestGraphBuilder` - Graph construction
- `TestTrainer` - Training infrastructure

## Common Issues

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'data'`

**Solution:**
```bash
cd backend
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest
```

### Missing Dependencies

**Problem:** `ImportError: torch_geometric not installed`

**Solution:**
```bash
pip install -r requirements.txt
```

### API Tests Failing

**Problem:** FBref/Understat tests timeout

**Solution:** These tests may fail due to network issues or website changes. The test suite is designed to gracefully handle this. Core functionality doesn't depend on these sources.

## Test Data

Tests use:
- **Synthetic data** for model architecture tests
- **StatsBomb free data** (2015/16 PL season) for integration tests
- **Mock data** for API service tests

No API keys required for basic testing.

## CI/CD Integration

For GitHub Actions or similar:

```yaml
- name: Run tests
  run: |
    cd backend
    pytest --cov=backend --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Performance Benchmarks

Expected test runtimes (on M1 Mac):
- `validate_pipeline.py`: ~5-10 seconds
- Full test suite: ~30-60 seconds
- With coverage: ~45-90 seconds

## Writing New Tests

### Template for Unit Tests

```python
def test_my_feature():
    """Test description."""
    from module import feature

    result = feature(input_data)

    assert result is not None
    assert result.property == expected_value
```

### Template for Integration Tests

```python
class TestMyFeature:
    """Test my feature integration."""

    def test_end_to_end(self):
        """Test complete workflow."""
        # Setup
        from data import load_data
        from models import train_model

        # Execute
        data = load_data()
        model = train_model(data)

        # Verify
        assert model.accuracy > 0.5
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest -vv -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Specific Test

```bash
pytest tests/test_integration.py::test_imports -vv
```

### See Print Statements

```bash
pytest -s  # Shows print() and logger output
```

## Continuous Validation

Recommended workflow:

1. **Before committing**: `python scripts/validate_pipeline.py`
2. **Before PR**: `pytest`
3. **After major changes**: `pytest --cov=backend`
4. **Before deploy**: `python scripts/validate_pipeline.py` in production environment

## Contact

For test failures or questions:
- Check logs: `backend/logs/`
- Review this guide
- Open an issue with test output

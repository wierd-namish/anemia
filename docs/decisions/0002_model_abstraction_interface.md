# ADR 0002: Model Abstraction Interface & Model Registry

## Status
Accepted

## Context
In previous iterations, model inference was tightly coupled to specific PyTorch or Scikit-Learn invocations in the API layer. Adding a new model required editing monolithic inference scripts.

## Decision
Introduce a formal `BaseModel` abstract base class in `anemia_ai.core.interfaces` and a `ModelRegistry` in `anemia_ai.models.registry`.
All model architectures (EfficientNet-B0, JetX-GT MLP, future Vision Transformers) implement `BaseModel.predict()` and `BaseModel.is_ready()`.

The API and application services depend exclusively on the `BaseModel` and `InferencePipeline` interfaces rather than concrete model implementations.

## Consequences
- Adding future AI models (e.g. Swin Transformer, MobileNetV4) requires only creating a new class in `src/anemia_ai/models/` and registering it with `register_model()`.
- Zero changes are needed in the API routers, request/response handlers, or UI pipelines.

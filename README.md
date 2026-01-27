FinanceLLM – End-to-End Deployed Financial Language Model
Github Code link
Demo Video
Overview
FinanceLLM is an end-to-end deployed, domain-specific Financial Language Model designed to answer finance-focused queries such as company disclosures, annual reports, strategy explanations, and financial signal extraction.
The project goes beyond model training and focuses on real-world inference, deployment, latency analysis, and production constraints, highlighting the gap between theoretical ML and building usable AI systems.
________________________________________
Key Highlights
•	End-to-end LLM fine-tuning → inference → deployment
•	CPU-based production inference (cost-aware setup)
•	Dockerized FastAPI service, deployed on Railway
•	Practical latency breakdown and optimization
•	Hands-on experience with model loading, failures, crashes, and recovery
________________________________________
Model & Training
•	Base Model: Qwen-2 (0.5B Instruct)
•	Fine-Tuning Method: LoRA (Low-Rank Adaptation)
•	Domain: Financial instruction-following
•	Dataset: Proprietary financial instruction dataset curated from structured financial disclosures and reports
•	Post-training:
o	LoRA weights merged into base model
o	Final merged model used for standalone inference (no adapter dependency)
________________________________________
Inference Architecture
•	Framework: FastAPI + Uvicorn
•	Inference Mode: CPU-based (cost-efficient)
•	Deployment Platform: Railway
•	Containerization: Docker (image built, tested, and deployed)
•	Serving Pattern: Single-instance inference service
________________________________________
Latency & Performance
Observed latency depends on prompt length and output size:
•	Typical Latency: ~5–20 seconds
•	Prompt Size: ~100–250 tokens
•	Key Contributors to Latency:
o	Tokenization & prompt length
o	Prefill phase (context processing)
o	Decode phase (token generation)
o	CPU-bound execution
Optimization Experiments
•	Separated latency into prefill vs decode stages
•	Reduced prompt verbosity to lower tokenization cost
•	Enforced deterministic decoding
•	Experimented with CPU quantization during inference (measurable latency reduction)
•	Evaluated performance vs cost trade-offs
________________________________________
Deployment Learnings
This project involved solving real deployment issues, including:
•	Dependency mismatches
•	Model loading failures
•	Runtime crashes after successful startup
•	Cold-start latency
•	Port and process misconfigurations
•	Template & routing errors in production
•	Differences between local and cloud execution
These challenges provided practical insight into why production ML is harder than training models or reading theory.
________________________________________
Scalability Considerations (Future Work)
The current deployment is intentionally kept cost-efficient, but the project identifies clear scaling paths:
•	Model serving via optimized inference engines (e.g., vLLM / ONNX Runtime)
•	GPU-backed inference for higher throughput
•	Async batching and request queues
•	Horizontal scaling with replicas
•	CI/CD pipelines for automated deployment
•	RAG integration using the fine-tuned model as the generation backbone
________________________________________
Tech Stack
•	Python
•	Hugging Face Transformers
•	LoRA / PEFT
•	FastAPI
•	Docker
•	Railway (deployment)
•	CPU-based inference
________________________________________
Key Takeaway
This project demonstrates practical GenAI engineering:
Building, deploying, debugging, and optimizing a real LLM system is fundamentally different from just training a model or studying theory.
FinanceLLM reflects hands-on experience with end-to-end AI system design, deployment trade-offs, and production realities.


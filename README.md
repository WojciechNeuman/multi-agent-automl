# 🤖 Multi-Agent AutoML

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-008489?style=for-the-badge&logo=pydantic&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)

---

## 📝 Project Overview

**Multi-Agent AutoML** is a system that automates the process of building and optimizing machine learning models using an agent-based architecture and LLM (Large Language Model, e.g., OpenAI GPT) support. Each stage of the ML pipeline (feature selection, model selection, tuning, evaluation) is handled by a specialized agent that communicates with an LLM and makes decisions based on data, history, and previous iteration results.

The project now includes a modern **React-based UI** for interactive pipeline configuration, real-time log streaming, and visualization of results and pipeline structure.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **OpenAI GPT (API) / LLM** – prompt generation, result interpretation, decision automation
- **scikit-learn** – ML models, pipelines, preprocessing
- **Pydantic** – data validation and serialization (request/response schemas)
- **Loguru** – logging
- **pytest** – unit testing
- **dotenv** – API key and configuration management
- **skops** – ML pipeline serialization
- **Jupyter Notebook** – testing and demonstration
- **Django** – backend API and SSE log streaming
- **React** – frontend user interface (in `frontend_ui/`)
- **Node.js & npm** – frontend dependencies and development server

---

## 🧩 Architecture

```
┌────────────────────────────┐
│      PipelineController    │
└────────────┬───────────────┘
             │
   ┌─────────▼─────────┐
   │   FeatureAgent    │  ← feature selection & engineering
   └─────────┬─────────┘
             │
   ┌─────────▼───────────┐
   │ ModelSelectionAgent │  ← model & hyperparameter selection
   └─────────┬───────────┘
             │
   ┌─────────▼─────────┐
   │  EvaluationAgent  │  ← evaluation, overfitting analysis, recommendations
   └───────────────────┘
             │
   ┌─────────▼─────────────┐
   │      Django API       │  ← REST endpoints, SSE log streaming
   └─────────┬─────────────┘
             │
   ┌─────────▼─────────┐
   │     React UI      │  ← interactive frontend, log streaming, results
   └───────────────────┘
```

- **Iterative workflow**: Agents pass summaries and recommendations to each other, learning from previous successes and failures.
- **LLM**: Each agent uses an LLM for decision-making and explanations.
- **Automatic best model selection**: After a set number of iterations, the model with the best metric (e.g., lowest RMSE) is chosen.
- **UI**: Users can upload datasets, configure pipeline parameters, and observe the process and results in real time.

---

## 🚀 How It Works

1. **FeatureAgent**: Analyzes data and selects the best features for modeling.
2. **ModelSelectionAgent**: Chooses the ML model and its hyperparameters based on features and history.
3. **EvaluationAgent**: Evaluates the pipeline, detects overfitting, and recommends next steps (continue, switch model, switch features, stop).
4. **PipelineController**: Orchestrates the process, passes summaries between agents, logs history, and selects the best model.
5. **Django API**: Handles file uploads, pipeline execution, log streaming (SSE), and result endpoints.
6. **React UI**: Lets users upload data, configure the pipeline, view live logs, and see results and pipeline diagrams.

---

## 📦 Directory Structure

```
.
├── agents/                # LLM agents (feature, model, evaluation)
├── datasets/              # Example datasets
├── frontend_ui/           # React frontend (UI)
│   ├── public/
│   └── src/
├── models/                # Enums, feature/model specs
├── notebooks/             # Jupyter notebooks for testing and demo
├── pipeline_controller/   # Main pipeline controller
├── schemas/               # Pydantic schemas (request/response)
├── tests/                 # Unit tests (mocked, no OpenAI required)
├── utils/                 # Utilities (EDA, metrics)
├── web_multi_agent_automl/ # Django backend (API, SSE, pipeline_api)
│   └── pipeline_api/
└── .github/workflows/     # CI/CD configuration
```

---

## 🖥️ User Interface (UI)

- **Location:** `frontend_ui/`
- **Features:**
  - Upload CSV datasets
  - Configure target column, problem type, iterations, and metrics
  - Start pipeline and monitor progress in real time
  - Live log streaming (SSE)
  - View selected features, best model, metrics, and reasoning
  - Visualize pipeline structure and diagram

**To run the UI locally:**
```bash
make install-frontend-deps
make run-frontend
```
The app will be available at [http://localhost:3000](http://localhost:3000).

---

## 🧪 Testing

- Unit tests do **not** require OpenAI – all agent calls are mocked.
- Run tests:
  ```bash
  make test
  ```

---

## ⚡ Quickstart

1. **Create and activate the Conda environment**
   ```bash
   make env
   make activate
   ```

   If you need to update the environment after changing `environment.yml`:
   ```bash
   make update
   ```

2. **Add your OpenAI key to `.env`**
   ```
   API_KEY=sk-...
   DJANGO_DEBUG=true
   ```
   Outside debug mode also set `DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS`.

3. **Run the backend (Django) server**
   ```bash
   make run-server
   ```

4. **Run the frontend (React) development server**
   ```bash
   make install-frontend-deps
   make run-frontend
   ```

5. **Run the pipeline on a sample dataset (Python API)**
   ```python
   from pipeline_controller.pipeline_controller import PipelineController

   controller = PipelineController(
       dataset_path="datasets/titanic.csv",
       target_column="Survived",
       problem_type="classification",
       max_iterations=3,
       main_metric="accuracy"
   )
   controller.run_full_pipeline()
   ```

---

## 📚 Examples

See the notebooks in the `notebooks/` directory (e.g., `apartments_pipeline_controller_test.ipynb`) for sample pipeline runs.

---

## 🧠 Authors & License

Educational project, MIT License.  
Authors: [Tomasz Makowski](https://github.com/makowskitomasz), [Wojciech Neuman](https://github.com/WojciechNeuman)

---

## ⭐ Inspirations

- [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning)
- [OpenAI GPT](https://platform.openai.com/)
- [scikit-learn](https://scikit-learn.org/)
- [React](https://react.dev/)
- [Django](https://www.djangoproject.com/)

---

> **Questions or want to contribute?**  
> Open an issue or contact the author!

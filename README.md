# TacticsAI ⚽

**A Chess Engine for Football Tactics** — ML-powered formation success prediction for the Premier League

![TacticsAI Demo](docs/assets/demo.gif)

---

## Overview

TacticsAI predicts which tactical setup will dominate in Premier League matchups. Input two teams and their formations, and our Graph Neural Network analyzes historical data, player roles, and style of play to forecast tactical outcomes.

Built for coaches, analysts, fantasy managers, and serious football fans who want data-driven tactical insights without enterprise pricing.

## Features

- **Formation Matchup Predictor** — Select two teams and formations to get tactical dominance predictions with confidence scores
- **Key Battle Zones** — Visual pitch map highlighting areas of tactical advantage and vulnerability
- **Player Role Analysis** — Understand how individual players affect the overall tactical matchup
- **Historical Validation** — View past matches with similar setups and their actual outcomes

## Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Model** | PyTorch Geometric (Graph Attention Network) |
| **Backend** | FastAPI |
| **Frontend** | Next.js + Tailwind CSS |
| **Data Sources** | StatsBomb Open Data, FBref, Understat |
| **Monitoring** | Arize |
| **Deployment** | Vercel (Frontend) + Railway (Backend) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Team Selector │  │ Formation UI │  │ Pitch Visualization  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                        Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  POST /predict                                            │   │
│  │  Input: team_a, team_b, formation_a, formation_b          │   │
│  │  Output: dominance_score, key_battles, confidence         │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      ML Pipeline                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ Graph Builder  │──│ GAT Model      │──│ Prediction      │   │
│  │ (PyG Format)   │  │ (Trained)      │  │ Interpreter     │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## ML Methodology

### Why Graph Neural Networks?

Formations are inherently relational — a 4-3-3 isn't just 11 isolated positions, it's a network of tactical relationships. GNNs capture this structure naturally:

- **Nodes** = Players (with position, historical performance, style metrics)
- **Edges** = Tactical relationships (passing connections, pressing triggers, spatial distance)
- **Global Features** = Team-level attributes (possession style, pressing intensity, formation flexibility)

### Model Details

- **Architecture:** Graph Attention Network (GAT) via PyTorch Geometric
- **Input:** Two team graphs representing formations
- **Output:** Tactical dominance score (0-100) with confidence interval
- **Training Data:** 5+ seasons of Premier League matches from StatsBomb Open Data
- **Target Variable:** Derived from xG differential, possession in final third, and chances created

### Fallback

XGBoost classifier available for environments where GNN inference is impractical.

## Project Structure

```
tacticsai/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md      # Detailed technical architecture
│   ├── ML_METHODOLOGY.md    # Model design decisions & performance
│   └── API.md               # API documentation
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Feature-engineered data
│   └── scripts/             # Data collection & processing
├── models/
│   ├── training/            # Training notebooks & scripts
│   ├── evaluation/          # Model evaluation notebooks
│   └── saved/               # Trained model files
├── backend/
│   ├── api/                 # FastAPI endpoints
│   └── services/            # Business logic
├── frontend/
│   └── ...                  # Next.js application
├── tests/                   # Unit & integration tests
└── .github/
    └── workflows/           # CI/CD pipelines
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tacticsai.git
   cd tacticsai
   ```

2. **Set up the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Download training data**
   ```bash
   cd data/scripts
   python download_statsbomb.py
   ```

### Running Locally

1. **Start the backend**
   ```bash
   cd backend
   uvicorn api.main:app --reload --port 8000
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open** [http://localhost:3000](http://localhost:3000)

## API Reference

### `POST /predict`

Predict tactical dominance for a formation matchup.

**Request:**
```json
{
  "team_a": "Arsenal",
  "team_b": "Manchester City",
  "formation_a": "4-3-3",
  "formation_b": "4-2-3-1"
}
```

**Response:**
```json
{
  "dominance_score": 42.7,
  "confidence": 0.78,
  "key_battles": [
    {
      "zone": "left_wing",
      "advantage": "team_b",
      "description": "Overload on the flank"
    }
  ],
  "historical_matches": [
    {
      "date": "2023-04-01",
      "result": "1-1",
      "similarity": 0.89
    }
  ]
}
```

See [docs/API.md](docs/API.md) for complete documentation.

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy (Win/Draw/Loss) | ~62% |
| MAE (Dominance Score) | 8.3 |
| Training Matches | 1,900+ |
| Cross-Validation | 5-fold |

*Performance metrics from evaluation on held-out test set.*

## Roadmap

- [ ] Add Champions League data
- [ ] Real-time lineup integration (pre-match)
- [ ] Player substitution impact predictions
- [ ] Mobile app (React Native)
- [ ] API access tier for developers
- [ ] Additional leagues (La Liga, Bundesliga, Serie A)

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for open football data
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN implementation
- Duke AIPI program for inspiring this project

---

**Built with ❤️ and a passion for the beautiful game**# TacticsAI
AI-powered tactical analysis and decision support system

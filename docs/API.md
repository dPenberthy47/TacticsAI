# API Documentation

## Endpoints

### POST /predict
Predict tactical dominance between two teams

**Request Body:**
```json
{
  "team_a": "string",
  "team_b": "string",
  "formation_a": "string",
  "formation_b": "string"
}
```

**Response:**
```json
{
  "dominance_score": "float",
  "key_battles": "array",
  "confidence": "float"
}
```

## Authentication
(To be documented)

## Rate Limiting
(To be documented)

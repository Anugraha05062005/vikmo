# VIKMO DealerDesk

**Track: AI / ML Intern (`aiml`)**

> A dealer-management and commerce backend for an Indian automotive parts distributor, featuring relational product catalogue management, concurrency-safe draft order workflows, an AI-powered conversational dealer assistant with tool calling and grounded retrieval, 4-week demand forecasting, and automated reorder suggestions.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://www.djangoproject.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](https://www.postgresql.org/)
[![Smoke Test](https://img.shields.io/badge/Smoke%20Test-12%2F12%20PASS-brightgreen.svg)]()

---

## Video Walkthrough Placeholder

- **Video Walkthrough Link**: `https://youtu.be/placeholder-vikmo-dealerdesk-walkthrough` *(5-minute technical code walkthrough of assistant tool loop and forecast pipeline)*

---

## 1. Features Implemented

### Required Modules (AI / ML Track)
- **Part A1: Backend Core**:
  - Relational schema for Products, Inventory, Dealers, Orders, and Order Items.
  - Strict pagination (`page`, `page_size`), filtering (`search`, `category`, `vehicle`), and exact contract error envelope (`{"error": {"code": "...", "message": "...", "details": []}}`).
  - Draft order creation and listing with live catalogue pricing and input validation.
  - Manual inventory adjustments via `PUT /api/inventory/{sku}/`.
  - Idempotent data seeding with exact catalogue counts (600 products, 25 dealers).
- **Part C: Dealer Assistant (GenAI)**:
  - Multi-turn conversational endpoint `POST /api/assistant/chat/` with session isolation (`session_id`).
  - Multi-provider LLM support: Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), and Gemini (`gemini-2.5-flash`).
  - Native tool calling loop with 3 validated tools: `search_parts`, `check_stock`, and `create_draft_order`.
  - Anti-injection isolation treating catalogue descriptions as untrusted data.
  - Real-time grounding: prices and inventory figures are strictly queried live from PostgreSQL.
  - Automatic extraction and verification of `cited_skus`.
- **Part D: Demand Forecasting & Reorder Suggestions (ML)**:
  - 4-week future weekly demand forecast for 40 SKUs committed in `ml/forecast.csv` (160 rows).
  - Chronological validation (no target leakage) evaluated against 3 mandatory baselines:
    - Baseline 1 (Last observed value): MAE = 11.72
    - Baseline 2 (Recent 4-week average): MAE = 8.76
    - Baseline 3 (Same week last year): MAE = 12.28
    - **Our Proposed Model**: MAE = **7.12** (**18.8% improvement over best baseline**).
  - `GET /api/forecast/{sku}/` endpoint returning 4-week forecast.
  - `GET /api/reorder-suggestions/` calculating lead-time demand, safety stock, and reorder point using live PostgreSQL inventory.

### Bonus Modules Implemented
- **Part A2: Complete Order Engine**:
  - Full order lifecycle transitions (`DRAFT` → `CONFIRMED` → `DISPATCHED` → `DELIVERED`, and `CANCELLED`).
  - Concurrency-safe atomic stock deduction with row-level locks (`select_for_update()`) preventing overselling.
  - Stock restoration on cancellation executed exactly once.
  - Exact financial calculations: 18% GST per line rounded `ROUND_HALF_UP` to two decimal places.
  - Order numbers formatted as `ORD-YYYYMMDD-NNNN` (Asia/Kolkata timezone) generated atomically.
  - Header-based `Idempotency-Key` handling concurrent duplicate requests.
- **Part B: Supplier Synchronization**:
  - `POST /api/sync/supplier/` endpoint pulling external catalogue feeds.
  - Resilient pagination and retry strategy handling HTTP 429 (`Retry-After`) and HTTP 5xx with backoff (up to 5 attempts).
  - All-or-nothing transactional rollback returning `502 SUPPLIER_UNAVAILABLE` upon unrecoverable outage.
  - Deduplication across pages by latest `updated_at`.
  - Preserves local inventory stock without modification.

---

## 2. Technology Stack

- **Backend Framework**: Python 3.11+, Django 5.2+, Django REST Framework 3.15+
- **Database**: PostgreSQL (configured via `DATABASE_URL`) with `psycopg` (binary) 3.3+ and `dj-database-url`
- **Machine Learning & Time Series**: `scikit-learn`, `pandas`, `numpy`
- **GenAI / LLM Providers**: Groq SDK (`groq`), OpenAI SDK (`openai`), Google GenAI (`google-genai`)
- **HTTP Client**: `requests` 2.32+
- **Testing**: Django Test Suite (`TestCase`, `APIClient`), `smoke_test.py`

---

## 3. Project Directory Structure

```text
.
├── API_CONTRACT.md              # Exact API specification
├── DECISIONS.md                 # Technical review & architectural answers
├── README.md                    # Project documentation
├── requirements.txt             # Linux-compatible dependencies
├── manage.py                    # Django CLI entrypoint
├── smoke_test.py                # Automated smoke test suite
├── .env.example                 # Template for environment variables
├── core/                        # Main Django application
│   ├── models.py                # Relational models (Product, Inventory, Dealer, Order, etc.)
│   ├── views.py                 # REST API endpoints
│   ├── urls.py                  # API routes under /api/
│   ├── serializers.py           # DRF Serializers with exact formatting
│   ├── services.py              # Business logic for orders, stock & GST
│   ├── exceptions.py            # Contract-compliant error envelope handler
│   ├── pagination.py            # Standard page/page_size pagination
│   ├── retrieval.py             # Hybrid vector & attribute retrieval
│   ├── assistant.py             # LLM tool-calling agent & session memory
│   ├── reorder_service.py       # Reorder suggestion & forecast queries
│   ├── supplier_sync.py         # Resilient supplier integration engine
│   ├── tests.py                 # Automated unit and integration test suite
│   └── management/commands/
│       └── seed_data.py         # Idempotent CSV database seeder
├── data/                        # Original supplied datasets (read-only)
│   ├── catalogue.csv
│   ├── dealers.csv
│   ├── festival_calendar.csv
│   ├── lead_times.csv
│   ├── promo_calendar.csv
│   └── sales_history.csv
├── ml/                          # Forecasting scripts and deliverables
│   ├── train_and_forecast.py    # Leakage-free ML training and evaluation pipeline
│   └── forecast.csv             # 160-row final forecast deliverable
└── supplier_mock/               # Supplier API mock for local sync testing
    └── mock_supplier.py
```

---

## 4. Setup and Run Contract

### 4.1 Prerequisites
- Python 3.11+
- PostgreSQL 14+ running locally or hosted (Neon / Supabase)

### 4.2 Installation & Environment Setup
Clone the repository and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Configure your PostgreSQL URL and optional LLM provider key:
```ini
DATABASE_URL=postgres://postgres:postgres@localhost:5432/dealerdesk
SUPPLIER_API_URL=http://127.0.0.1:8099
SUPPLIER_API_KEY=dev-supplier-key
DJANGO_SECRET_KEY=dev-vikmo-dealerdesk-secret-key-2026
DJANGO_DEBUG=1

# Optional LLM Key (Groq, OpenAI, or Gemini)
GROQ_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
```

### 4.3 Database Migration and Seeding
```bash
python manage.py migrate --noinput
python manage.py seed_data
python manage.py seed_data      # Running twice confirms idempotency
```

### 4.4 Start Server
```bash
python manage.py runserver 0.0.0.0:8000 --noreload
```

---

## 5. Automated Testing

### 5.1 Django Unit & Integration Tests
Run the comprehensive test suite covering product catalog, inventory, order lifecycle, GST rounding, idempotency, assistant, and reorder endpoints:
```bash
python manage.py test core
```
*Result: 13 / 13 tests passed.*

### 5.2 Official Smoke Test
Execute the official smoke test for the AI/ML track:
```bash
python smoke_test.py --base-url http://localhost:8000 --track aiml
```
*Result: 10 / 10 checks passed.*

Execute the smoke test across all modules (including order engine and supplier sync):
```bash
python smoke_test.py --base-url http://localhost:8000 --all
```
*Result: 12 / 12 checks passed.*

---

## 6. API Endpoint Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health/` | System health check (`{"status": "ok"}`) |
| `GET` | `/api/products/` | Paginated product list with search/filter |
| `POST` | `/api/products/` | Create a new product (initial stock 0) |
| `GET` | `/api/products/{sku}/` | Product details and current live stock |
| `PATCH`| `/api/products/{sku}/` | Partial update (price, description, etc.) |
| `PUT` | `/api/inventory/{sku}/` | Set live warehouse inventory quantity |
| `GET` | `/api/dealers/` | Paginated list of registered dealers |
| `POST` | `/api/dealers/` | Register a new dealer |
| `GET` | `/api/dealers/{id}/` | Dealer details |
| `POST` | `/api/orders/` | Create a DRAFT order (`Idempotency-Key` supported) |
| `GET` | `/api/orders/` | Paginated orders list (filter by dealer or status) |
| `GET` | `/api/orders/{order_number}/` | Order details with GST breakdown |
| `PUT` | `/api/orders/{order_number}/` | Replace items in a DRAFT order |
| `POST` | `/api/orders/{order_number}/confirm/` | Confirm order & deduct live stock atomically |
| `POST` | `/api/orders/{order_number}/dispatch/` | Mark order as dispatched |
| `POST` | `/api/orders/{order_number}/deliver/` | Mark order as delivered |
| `POST` | `/api/orders/{order_number}/cancel/` | Cancel order (restores stock if confirmed) |
| `POST` | `/api/assistant/chat/` | Conversational dealer assistant with tool calling |
| `GET` | `/api/forecast/{sku}/` | 4-week demand forecast for a forecast SKU |
| `GET` | `/api/reorder-suggestions/` | Live stock reorder suggestions for 40 SKUs |
| `POST` | `/api/sync/supplier/` | Pull supplier catalogue with retries and rollback |

---

## 7. GenAI Assistant & Forecasting Architecture

### LLM Assistant Configuration
- **Default LLM Provider**: Groq (`llama-3.3-70b-versatile`) / OpenAI (`gpt-4o-mini`) / Gemini (`gemini-2.5-flash`).
- **Tool Calling**: Native function calling executing `search_parts`, `check_stock`, and `create_draft_order`.
- **Grounded Fallback**: Includes a robust server-side grounded query execution engine that handles tool selection and catalog search even if external API keys are unavailable.

### Forecasting Pipeline
- **Script**: `ml/train_and_forecast.py`
- **Output**: `ml/forecast.csv` (160 rows: 40 SKUs × 4 weeks: `2026-09-28` to `2026-10-19`).
- **Features**: Lagged sales (`lag_1` to `lag_4`), rolling statistics (`rolling_mean_4`, `rolling_std_4`), known promotion calendar flags (`promo_flag`), and festival distance (`weeks_to_diwali`).
- **Baselines**: Evaluated chronologically with zero data leakage against Last Observed (MAE: 11.72), Recent 4-week Average (MAE: 8.76), and Same Week Last Year (MAE: 12.28).

---

## 8. Known Limitations & Assumptions

1. **Credit Limit**: Dealers have a `credit_limit` stored in the database, but draft order creation does not block on credit exhaustion (as specified in the contract where no core rule uses it yet).
2. **Catalog Scope**: Demand forecasting is modeled for the 40 SKUs specified in `lead_times.csv`. Non-forecast SKUs return 404 on `/api/forecast/{sku}/`.
3. **Diwali Timing**: Pre-Diwali demand surge is indexed dynamically using signed weeks to Diwali (`weeks_to_diwali`), reflecting the lunar calendar shift across years.

import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat

app = FastAPI(title="Loan Comparison API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanOfferInput(BaseModel):
    lender: str = Field(..., description="Lender name")
    apr_nominal: confloat(ge=0) = Field(..., description="Nominal annual interest rate in percent, e.g. 8.5")
    establishment_fee: confloat(ge=0) = Field(0, description="One-time fee at start")
    term_fee: confloat(ge=0) = Field(0, description="Recurring monthly fee")


class LoanComparisonRequest(BaseModel):
    loan_amount: confloat(gt=0) = Field(..., description="Loan principal in NOK")
    term_months: int = Field(..., gt=0, description="Loan term in months")
    offers: List[LoanOfferInput]


class LoanOfferResult(BaseModel):
    lender: str
    monthly_payment: float
    monthly_cost_with_fees: float
    total_interest: float
    total_fees: float
    total_cost: float
    effective_rate_annual: float


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


def _compute_annuity_monthly(principal: float, annual_rate_percent: float, n_months: int) -> float:
    """Monthly annuity payment excluding monthly fees.
    annual_rate_percent is e.g. 8.5 for 8.5%.
    """
    r = (annual_rate_percent / 100.0) / 12.0
    if r == 0:
        return principal / n_months
    return principal * r / (1 - (1 + r) ** (-n_months))


def _total_interest(principal: float, annual_rate_percent: float, n_months: int, monthly_payment: float) -> float:
    """Compute total interest paid over term for an annuity loan (excluding term fees)."""
    r = (annual_rate_percent / 100.0) / 12.0
    if r == 0:
        return 0.0
    balance = principal
    total_interest = 0.0
    for _ in range(n_months):
        interest = balance * r
        principal_component = monthly_payment - interest
        balance -= principal_component
        total_interest += interest
    return max(total_interest, 0.0)


def _solve_effective_rate_annual(principal_plus_start_fee: float, monthly_payment_with_fees: float, n_months: int) -> float:
    """Approximate effective annual rate including fees by solving for monthly rate x where
    PV(annuity with payment=monthly_payment_with_fees, n=n_months, rate=x) = principal_plus_start_fee.
    Returns annualized rate: (1+x)^12 - 1.
    """
    # Binary search for monthly rate x in [0, 100%]
    low, high = 0.0, 1.0
    for _ in range(80):  # sufficient iterations for precision
        mid = (low + high) / 2
        if mid == 0:
            pv = monthly_payment_with_fees * n_months
        else:
            pv = monthly_payment_with_fees * (1 - (1 + mid) ** (-n_months)) / mid
        if pv > principal_plus_start_fee:
            # Discounted payments are large -> rate too low
            low = mid
        else:
            high = mid
    monthly_rate = (low + high) / 2
    annual_rate = (1 + monthly_rate) ** 12 - 1
    return annual_rate


@app.get("/api/default-offers", response_model=List[LoanOfferInput])
def default_offers():
    """Sample lenders and assumptions (illustrative only, not financial advice)."""
    return [
        LoanOfferInput(lender="Bank A", apr_nominal=8.5, establishment_fee=950, term_fee=45),
        LoanOfferInput(lender="Bank B", apr_nominal=9.2, establishment_fee=0, term_fee=70),
        LoanOfferInput(lender="Bank C", apr_nominal=7.9, establishment_fee=1490, term_fee=30),
    ]


@app.post("/api/compare-loans", response_model=List[LoanOfferResult])
def compare_loans(payload: LoanComparisonRequest):
    results: List[LoanOfferResult] = []
    P = float(payload.loan_amount)
    n = int(payload.term_months)

    for offer in payload.offers:
        monthly_payment = _compute_annuity_monthly(P, offer.apr_nominal, n)
        total_interest_val = _total_interest(P, offer.apr_nominal, n, monthly_payment)
        total_fees_val = float(offer.establishment_fee) + float(offer.term_fee) * n
        monthly_with_fees = monthly_payment + float(offer.term_fee)
        total_cost_val = total_interest_val + total_fees_val

        # Effective rate including fees
        principal_plus_start = P + float(offer.establishment_fee)
        eff_rate_annual = _solve_effective_rate_annual(principal_plus_start, monthly_with_fees, n)

        results.append(
            LoanOfferResult(
                lender=offer.lender,
                monthly_payment=round(monthly_payment, 2),
                monthly_cost_with_fees=round(monthly_with_fees, 2),
                total_interest=round(total_interest_val, 2),
                total_fees=round(total_fees_val, 2),
                total_cost=round(total_cost_val, 2),
                effective_rate_annual=round(eff_rate_annual * 100, 2),
            )
        )

    # Sort by total cost ascending
    results.sort(key=lambda x: x.total_cost)
    return results


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

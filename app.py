import os
import re
import json
import logging
import datetime
import aiohttp
import requests
import openai
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import pathlib
from fastapi.staticfiles import StaticFiles


from azure.cosmos import CosmosClient
import azure.cosmos.exceptions as cosmos_exceptions


BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "dist"

print(f"Looking for static files in: {STATIC_DIR}")


app = FastAPI(
    title="Enhanced Investment AI Advisor",
    version="1.0.0",
    description="Investment advisor with real-time pricing, news integration, and natural language trading"
)

load_dotenv()


logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'backend.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


OPENAI_API_KEY = "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"  # Replace with actual key
NEWSAPI_KEY = "c79bb964cee44d689e041bd0bc9186f6"     # Replace with actual key
ALPHAVANTAGE_API_KEY = "GKBIM7SNL6YTX48E"  # Replace with actual key
RISK_ENDPOINT_URL = "https://riskapi-202501091928.uksouth.inference.ml.azure.com/score"  # Replace with actual URL
RISK_ENDPOINT_KEY = "nrV6d2ICSejapSeAFxe4WKltv9bfPtaB"  # Replace with actual key

BUY_FUNCTION_URL = "https://buyshares.azurewebsites.net/api/buyshares"
SELL_FUNCTION_URL = "https://buyshares.azurewebsites.net/api/sellshares"

COSMOS_ENDPOINT = "https://fastlmdb.documents.azure.com:443/"  # Replace with actual endpoint
COSMOS_KEY = "S3LP4JXhAAJuO9CF4mGk6dlLPK4mHxjVg29jekVtIujKGdcCFCXt0Y7TDUAZPbcifXqC3VkGOGryACDbMrfd7A=="  # Replace with actual key
COSMOS_DB_NAME = "users"
COSMOS_CONTAINER_NAME = "users"

openai.api_key = OPENAI_API_KEY



try:
    if STATIC_DIR.exists():
        print(f"Found dist folder at {STATIC_DIR}")
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:
        print(f"Dist folder not found at {STATIC_DIR}")
        @app.get("/")
        def no_build():
            return {"message": "dist folder not found"}
except Exception as e:
    print(f"Error mounting static files: {str(e)}")


except Exception as e:
    print(f"Error mounting static files: {str(e)}")

# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(500)
async def internal_error(request, exc):
    print(f"500 error: {str(exc)}")  # This will show in Azure logs
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)}
    )

@app.get("/debug")
async def debug_paths():
    try:
        import os
        current_dir = os.getcwd()
        files = os.listdir(current_dir)
        static_exists = STATIC_DIR.exists()
        static_files = os.listdir(STATIC_DIR) if static_exists else []
        
        return {
            "current_dir": str(current_dir),
            "files_in_current": files,
            "static_dir": str(STATIC_DIR),
            "static_exists": static_exists,
            "static_files": static_files,
            "base_dir": str(BASE_DIR)
        }
    except Exception as e:
        return {
            "error": str(e),
            "current_dir": os.getcwd(),
            "files": os.listdir(os.getcwd())
        }


def get_cosmos_container():
    try:
        client = CosmosClient(
            url=COSMOS_ENDPOINT,
            credential=COSMOS_KEY,
            connection_verify=True  # Enable SSL verification
        )
        
        # Add logging to debug connection
        logging.info(f"Connecting to database: {COSMOS_DB_NAME}")
        database = client.get_database_client(COSMOS_DB_NAME)
        
        logging.info(f"Getting container: {COSMOS_CONTAINER_NAME}")
        container = database.get_container_client(COSMOS_CONTAINER_NAME)
        
        # Test the connection
        container.read()
        logging.info("Successfully connected to Cosmos DB")
        
        return container
    except Exception as e:
        logging.error(f"Failed to connect to Cosmos DB: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )

@app.get("/users")  
async def get_users():
    try:
        container = get_cosmos_container()
        
        # Use parameterized query
        query = "SELECT * FROM c"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        logging.info(f"Retrieved {len(items)} users")
        return items
        
    except Exception as e:
        logging.error(f"Error in get_users: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class RiskPayload(BaseModel):
    tickers: List[str]
    weights: List[float]

class AIMessage(BaseModel):
    role: str
    content: str

class AIChatRequest(BaseModel):
    conversation: List[AIMessage]
    user_membership: str

class TradeRequest(BaseModel):
    action: str
    symbol: str
    shares: float

   
def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """Fetch user document by ID from Cosmos DB."""
    try:
        container = get_cosmos_container()
        user = container.read_item(item=user_id, partition_key=user_id)
        return dict(user)
    except cosmos_exceptions.CosmosResourceNotFoundError:
        return {}
    except Exception as e:
        logging.error(f"Error reading user from Cosmos: {str(e)}")
        return {}

def update_user_portfolio(user_id: str, symbol: str, shares: float, action: str) -> Dict[str, Any]:
    """Update user portfolio after buy/sell operations."""
    try:
        container = get_cosmos_container()
        user = container.read_item(item=user_id, partition_key=user_id)
        
        portfolio = user.get("portfolio", [])
        symbol = symbol.upper()
        
        # Find existing position
        position = next((p for p in portfolio if p["symbol"] == symbol), None)
        
        if action == "buy":
            if position:
                position["shares"] += shares
            else:
                portfolio.append({"symbol": symbol, "shares": shares})
        elif action == "sell":
            if not position:
                raise HTTPException(400, f"No position found for {symbol}")
            if position["shares"] < shares:
                raise HTTPException(400, f"Insufficient shares for {symbol}")
            position["shares"] -= shares
            if position["shares"] <= 0:
                portfolio.remove(position)
                
        user["portfolio"] = portfolio
        container.upsert_item(user)
        return user
        
    except cosmos_exceptions.CosmosHttpResponseError as e:
        logging.error(f"Cosmos DB operation failed: {str(e)}")
        raise HTTPException(500, "Database operation failed")


def fetch_realtime_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch real-time prices using yfinance."""
    price_map: Dict[str, float] = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1d")
            if hist.empty:
                logging.warning(f"No data for {sym}, using default price of 100")
                price_map[sym.upper()] = 100.0
            else:
                price_map[sym.upper()] = float(hist["Close"].iloc[-1])
        except Exception as e:
            logging.error(f"YFinance error for {sym}: {str(e)}")
            price_map[sym.upper()] = 100.0
    return price_map


def parse_buy_sell_command(text: str) -> Dict[str, Any]:
    """Parse natural language trading commands."""
    text = text.lower().strip()
    
    patterns = [
        r"(buy|sell)\s+(\d+\.?\d*)\s+shares?\s+(?:of\s+)?([a-zA-Z]+)",
        r"(buy|sell)\s+([a-zA-Z]+)\s+(\d+\.?\d*)\s+shares?"
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            action, *rest = match.groups()
            if rest[0].isalpha():
                symbol, shares = rest[0], float(rest[1])
            else:
                shares, symbol = float(rest[0]), rest[1]
            
            return {
                "action": action,
                "shares": shares,
                "symbol": symbol.upper()
            }
    
    return {}


async def fetch_news_alphavantage(symbol: str) -> List[Dict[str, Any]]:
    """Fetch news from AlphaVantage API."""
    if not ALPHAVANTAGE_API_KEY:
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "key": ALPHAVANTAGE__KEY,
        "sort": "LATEST",
        "limit": 10
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as resp:
                if resp.status != 200:
                    logging.error(f"AlphaVantage error: {resp.status}")
                    return []

                data = await resp.json()
                if "Information" in data:  # API limit reached
                    logging.warning(f"AlphaVantage limit: {data['Information']}")
                    return []

                feed = data.get("feed", [])
                return [{
                    "title": item.get("title", "(No title)"),
                    "url": item.get("url", ""),
                    "source_api": "AlphaVantage",
                    "sentiment_score": item.get("overall_sentiment_score"),
                    "timestamp": item.get("time_published")
                } for item in feed]

    except Exception as e:
        logging.error(f"AlphaVantage fetch error: {str(e)}")
        return []

async def fetch_news_newsapi(symbol: str) -> List[Dict[str, Any]]:
    """Fetch news from NewsAPI."""
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "language": "en",
        "pageSize": 10,
        "apiKey": NEWSAPI_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logging.error(f"NewsAPI error: {resp.status}")
                    return []

                data = await resp.json()
                articles = data.get("articles", [])
                return [{
                    "title": art.get("title", "(No title)"),
                    "url": art.get("url", ""),
                    "source_api": "NewsAPI",
                    "source_name": art.get("source", {}).get("name"),
                    "published_at": art.get("publishedAt")
                } for art in articles if art.get("title") and "[Removed]" not in art["title"]]

    except Exception as e:
        logging.error(f"NewsAPI fetch error: {str(e)}")
        return []

async def fetch_combined_news(symbol: str = None, topic: str = None) -> List[dict]:
    """Fetch news from all configured sources."""
    query = symbol or topic or "business"
    results = []

    # Fetch from AlphaVantage
    av_results = await fetch_news_alphavantage(query)
    results.extend(av_results)

    # Fetch from NewsAPI
    na_results = await fetch_news_newsapi(query)
    results.extend(na_results)

    # Sort by timestamp if available
    return sorted(
        results,
        key=lambda x: x.get("timestamp") or x.get("published_at") or "",
        reverse=True
    )


def interpret_risk_metrics(risk_data: Dict[str, Any]) -> Dict[str, str]:
    """Interpret risk metrics into human-readable format."""
    defaults = {
        "risk_score": 0.0,
        "portfolio_volatility": 0.0,
        "sharpe_ratio": 0.0
    }
    merged = {**defaults, **risk_data}
    
    risk_score = merged["risk_score"]
    vol = merged["portfolio_volatility"]
    sharpe = merged["sharpe_ratio"]

    # Risk level thresholds
    if risk_score < 0.03:
        risk_level = "Low"
    elif risk_score < 0.06:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Volatility interpretation
    if vol < 0.01:
        vol_level = "Low"
    elif vol < 0.02:
        vol_level = "Medium"
    else:
        vol_level = "High"

    # Performance rating based on Sharpe ratio
    if sharpe < 1:
        perf = "Poor"
    elif sharpe < 2:
        perf = "Good"
    else:
        perf = "Excellent"

    summary = (
        f"Your portfolio shows {risk_level.lower()} risk with {perf.lower()} "
        f"risk-adjusted returns. The volatility is {vol_level.lower()}, "
        f"indicating {vol_level.lower()} price fluctuations."
    )

    return {
        "risk_level": risk_level,
        "volatility_level": vol_level,
        "performance_rating": perf,
        "summary": summary
    }

def compute_user_portfolio_risk(user: Dict[str, Any]) -> Dict[str, Any]:
    """Compute risk metrics for user portfolio."""
    portfolio = user.get("portfolio", [])
    if not portfolio:
        return {}

    # Summarize shares by symbol
    symbol_shares: Dict[str, float] = {}
    for item in portfolio:
        sym = item["symbol"].upper()
        shares = float(item["shares"])
        symbol_shares[sym] = symbol_shares.get(sym, 0.0) + shares

    # Fetch real-time prices
    unique_symbols = list(symbol_shares.keys())
    price_map = fetch_realtime_prices(unique_symbols)

    # Calculate total value and weights
    total_value = 0.0
    for sym in unique_symbols:
        total_value += price_map[sym] * symbol_shares[sym]

    if total_value <= 0:
        return {}

    # Prepare risk payload
    tickers = []
    weights = []
    for sym in unique_symbols:
        val = price_map[sym] * symbol_shares[sym]
        weight = val / total_value
        tickers.append(sym)
        weights.append(weight)

    # Call risk endpoint if configured
    if not RISK_ENDPOINT_URL or not RISK_ENDPOINT_KEY:
        logging.warning("Risk endpoint not configured")
        return {}

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RISK_ENDPOINT_KEY}"
        }
        payload = {"tickers": tickers, "weights": weights}
        
        resp = requests.post(
            RISK_ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=10
        )

        if resp.status_code != 200:
            logging.error(f"Risk endpoint error: {resp.status_code}, {resp.text}")
            return {}

        risk_data = resp.json()
        interpretation = interpret_risk_metrics(risk_data)
        
        return {
            **risk_data,
            **interpretation,
            "portfolio_value": total_value,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Risk calculation error: {str(e)}")
        return {}

################################################
# Risk Endpoint
################################################
@app.post("/portfolio_risk")
async def get_portfolio_risk(payload: RiskPayload):
    """Calculate portfolio risk metrics."""
    if not RISK_ENDPOINT_URL or not RISK_ENDPOINT_KEY:
        raise HTTPException(500, "Risk endpoint not configured")

    # Limit number of positions for API efficiency
    if len(payload.tickers) > 12:
        payload.tickers = payload.tickers[:12]
        payload.weights = payload.weights[:12]

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RISK_ENDPOINT_KEY}"
        }
        
        resp = requests.post(
            RISK_ENDPOINT_URL,
            headers=headers,
            json=payload.dict(),
            timeout=10
        )

        if resp.status_code != 200:
            raise HTTPException(500, "Risk calculation failed")

        risk_data = resp.json()
        interpretation = interpret_risk_metrics(risk_data)
        
        return {
            **risk_data,
            **interpretation,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Portfolio risk endpoint error: {str(e)}")
        raise HTTPException(500, str(e))

################################################
# News Endpoints
################################################
@app.get("/symbol_news/{symbol}")
async def get_symbol_news(symbol: str, limit: int = 5):
    """Get news articles for a specific symbol."""
    articles = await fetch_combined_news(symbol=symbol)
    
    if not articles:
        return {
            "symbol": symbol,
            "articles": [],
            "message": "No articles found or API error"
        }

    return {
        "symbol": symbol,
        "articles": articles[:limit],
        "sources_used": list(set(art["source_api"] for art in articles[:limit]))
    }

@app.get("/users/{user_id}/news")
async def get_user_news(user_id: str, limit: int = 5):
    """Get news relevant to user's portfolio and preferences."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    topic = user.get("preferred_news_topic", "business")
    articles = await fetch_combined_news(topic=topic)
    
    return {
        "user_id": user_id,
        "articles": articles[:limit],
        "sources_used": list(set(art["source_api"] for art in articles[:limit]))
    }

################################################
# Trading Execution
################################################
async def execute_trade(user_id: str, action: str, symbol: str, shares: float) -> Dict[str, Any]:
    """Execute a buy/sell trade through Azure Functions."""
    if action not in ["buy", "sell"]:
        raise HTTPException(400, "Invalid trade action")

    function_url = BUY_FUNCTION_URL if action == "buy" else SELL_FUNCTION_URL
    if not function_url:
        raise HTTPException(500, f"{action.upper()}_FUNCTION_URL not configured")

    try:
        # Call Azure Function
        params = {
            "user_id": user_id,
            "symbol": symbol.upper(),
            "shares": shares
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(function_url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logging.error(f"Trade execution failed: {error_text}")
                    raise HTTPException(
                        status_code=resp.status,
                        detail=f"Trade execution failed: {error_text}"
                    )
                
                result = await resp.json()
                
                # Update user portfolio in Cosmos DB
                updated_user = update_user_portfolio(
                    user_id=user_id,
                    symbol=symbol,
                    shares=shares,
                    action=action
                )
                
                return {
                    "status": "success",
                    "action": action,
                    "symbol": symbol,
                    "shares": shares,
                    "execution_details": result,
                    "portfolio": updated_user.get("portfolio", [])
                }

    except aiohttp.ClientError as e:
        logging.error(f"Network error during trade: {str(e)}")
        raise HTTPException(500, "Failed to execute trade due to network error")
    except Exception as e:
        logging.error(f"Trade execution error: {str(e)}")
        raise HTTPException(500, str(e))

################################################
# Trading Endpoints
################################################
@app.post("/users/{user_id}/trade")
async def execute_user_trade(user_id: str, trade: TradeRequest):
    """Execute a trade for a user."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    result = await execute_trade(
        user_id=user_id,
        action=trade.action,
        symbol=trade.symbol,
        shares=trade.shares
    )
    
    return result

################################################
# AI Chat Helper Functions
################################################
def prepare_portfolio_summary(user: Dict[str, Any], risk_info: Dict[str, Any]) -> str:
    """Create a portfolio summary for AI context."""
    portfolio = user.get("portfolio", [])
    if not portfolio:
        return "No current positions in portfolio."

    # Get real-time values
    symbols = [p["symbol"] for p in portfolio]
    prices = fetch_realtime_prices(symbols)
    
    summary = ["Current Portfolio:"]
    total_value = 0
    
    for pos in portfolio:
        symbol = pos["symbol"]
        shares = pos["shares"]
        price = prices.get(symbol, 100.0)
        value = shares * price
        total_value += value
        
        summary.append(
            f"- {symbol}: {shares:.2f} shares @ ${price:.2f} = ${value:.2f}"
        )
    
    summary.append(f"\nTotal Portfolio Value: ${total_value:.2f}")
    
    if risk_info:
        summary.append(f"\nRisk Assessment: {risk_info.get('summary', 'Not available')}")
    
    return "\n".join(summary)

async def prepare_news_summary(portfolio_symbols: List[str], limit: int = 2) -> str:
    """Create a news summary for AI context."""
    if not portfolio_symbols:
        return "No symbols to fetch news for."

    all_news = []
    for symbol in portfolio_symbols:
        articles = await fetch_combined_news(symbol=symbol)
        if articles:
            all_news.append(f"\nNews for {symbol}:")
            for article in articles[:limit]:
                title = article.get("title", "No title")
                source = article.get("source_api", "Unknown source")
                all_news.append(f"- {title} (via {source})")
    
    return "\n".join(all_news) if all_news else "No recent news available."

################################################
# AI Chat Endpoint
################################################
@app.post("/users/{user_id}/ai_chat")
async def ai_chat(user_id: str, payload: AIChatRequest):
    """
    Handle AI chat interactions with trading capabilities.
    Processes natural language trading commands and provides portfolio insights.
    """
    # Validate requirements
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OpenAI API key not configured")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    try:
        # Check for trading command in last message
        if payload.conversation:
            last_msg = payload.conversation[-1].content
            command = parse_buy_sell_command(last_msg)
            
            if command:
                # Execute trade if command found
                trade_result = await execute_trade(
                    user_id=user_id,
                    action=command["action"],
                    symbol=command["symbol"],
                    shares=command["shares"]
                )
                
                # Refresh user data after trade
                user = get_user_by_id(user_id)

        # Prepare context information
        risk_info = compute_user_portfolio_risk(user)
        portfolio_summary = prepare_portfolio_summary(user, risk_info)
        
        portfolio_symbols = [p["symbol"] for p in user.get("portfolio", [])]
        news_summary = await prepare_news_summary(portfolio_symbols)

        # Build system message for AI
        system_content = f"""You are an advanced Investment AI Advisor.

User Profile:
- Membership: {payload.user_membership}
- ID: {user_id}

{portfolio_summary}

Recent Market News:
{news_summary}

Guidelines:
1. You can execute trades when users ask (e.g., "buy 10 shares of AAPL")
2. Provide clear, concise advice considering their portfolio and risk
3. Reference relevant news when making recommendations
4. Use plain text (no markdown)
5. Always consider their current positions when advising
"""

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_content}
        ] + [m.dict() for m in payload.conversation]

        # Call OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1200
        )

        reply = response.choices[0].message.content.strip()
        
        return {
            "reply": reply,
            "context": {
                "portfolio_value": risk_info.get("portfolio_value"),
                "risk_level": risk_info.get("risk_level"),
                "trade_executed": bool(command)
            }
        }

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(500, "AI service temporarily unavailable")
    except Exception as e:
        logging.error(f"AI chat error: {str(e)}")
        raise HTTPException(500, str(e))

    
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "message": str(exc.detail),
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled error: {str(exc)}")
    return {
        "error": True,
        "message": "An unexpected error occurred",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

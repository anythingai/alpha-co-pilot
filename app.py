import os
import json
import re
import logging
from datetime import datetime
from typing import Optional, Any, List, Dict, cast

import requests
from dotenv import load_dotenv  # type: ignore[import]
from pytrends.request import TrendReq  # type: ignore[import]
from flask import Flask, request, jsonify, render_template  # type: ignore[import]
from flask_cors import CORS  # type: ignore[import]
from langchain_openai import AzureChatOpenAI  # type: ignore[import]
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore[import]
from pydantic.v1 import SecretStr  # type: ignore[import]
import tiktoken  # type: ignore[import]
try:
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

# Load environment variables
load_dotenv()
# Wrapper for Google Trends data using free-tier pytrends
class GoogleTrendsAPI:
    """Wrapper for Google Trends data"""
    @staticmethod
    def get_interest_over_time(query: str, timeframe: str = 'now 7-d') -> Dict[str, int]:
        try:
            pytrends = TrendReq()
            pytrends.build_payload([query], timeframe=timeframe)
            data = pytrends.interest_over_time()
            if getattr(data, "empty", True):
                return {}
            series: Any = data[query]  # type: ignore[index]
            to_dict_fn = getattr(series, "to_dict", None)
            if callable(to_dict_fn):
                result = to_dict_fn()
                if isinstance(result, dict):
                    converted: Dict[str, int] = {}
                    for k, v in result.items():
                        key = k.isoformat() if hasattr(k, 'isoformat') else str(k)
                        try:
                            converted[key] = int(v) if v is not None else 0
                        except Exception:
                            try:
                                converted[key] = int(float(v))
                            except Exception:
                                converted[key] = 0
                    return converted
                return {}
            # Fallback conversion
            try:
                raw_map = dict(series)  # type: ignore[arg-type]
                converted2: Dict[str, int] = {}
                for k, v in raw_map.items():
                    key = k.isoformat() if hasattr(k, 'isoformat') else str(k)
                    try:
                        converted2[key] = int(v) if v is not None else 0
                    except Exception:
                        try:
                            converted2[key] = int(float(v))
                        except Exception:
                            converted2[key] = 0
                return converted2
            except Exception:
                return {}
        except Exception as e:
            logger.error(f"❌ Error fetching Google Trends for {query}: {e}")
            return {}

# Configure logging with more detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Azure OpenAI configuration using LangChain
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'o4-mini')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Utility to coerce LangChain message content to string
def _coerce_content_to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get('text')
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return ''.join(parts)
    try:
        return str(value)
    except Exception:
        return ''

# Removed old filtering helper functions - now using generalized approach
class QueryParser:
    """Generalized query parser for broad crypto analysis."""

    @staticmethod
    def parse(query: str) -> Dict[str, Any]:
        """Parse query for general analysis without strict filtering"""
        ql = query.lower()
        result: Dict[str, Any] = {
            'num': 50,  # Increased default to get more comprehensive results
            'query_text': query,  # Store original query for context
        }

        # Extract number if specified, but allow much higher limits
        m = re.search(r'\btop\s+(\d+)\b', ql)
        if m:
            try:
                result['num'] = max(1, min(200, int(m.group(1))))  # Increased max limit
            except Exception:
                pass

        # Extract symbols for targeted analysis if specified
        sym_match = re.search(r'\b(?:analy[s|z]e|compare|vs|symbols?)\b\s*([a-z0-9\s,\-]+)', ql)
        if sym_match:
            raw = sym_match.group(1)
            parts = re.split(r'[\s,]+', raw)
            symbols = [p.strip().upper() for p in parts if p.strip() and len(p.strip()) <= 10]
            if symbols:
                result['symbols'] = symbols

        return result

# Initialize LangChain AzureChatOpenAI with DEBUG ENHANCEMENTS
try:
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=1.0,  # ✅ o1/o3/o4 models require temperature=1.0
        model_kwargs={
            "max_completion_tokens": 4096,
            "reasoning_effort": "high",
            "response_format": {"type": "text"}
        },
        timeout=60,  # 🔧 INCREASED timeout for reasoning models
        max_retries=3
    )
    logger.info(f"✅ Successfully initialized AzureChatOpenAI with deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    logger.info(f"📍 Endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"🔧 API Version: {AZURE_OPENAI_API_VERSION}")
    logger.info("🎯 Using max_completion_tokens: 4096 for reasoning model (o1/o3/o4 series)")
    logger.info("🌡️ Temperature set to 1.0 (required for reasoning models)")
    logger.info("🧠 Reasoning effort: high")
    
    # Test connection with detailed response logging
    test_response = llm.invoke([HumanMessage(content="Hello, this is a connection test. Please respond with 'Connection successful'.")])
    logger.info("🔗 Connection test successful - LangChain AzureChatOpenAI is ready!")
    _test_content = _coerce_content_to_str(getattr(test_response, 'content', ''))
    logger.info(f"🧪 Test response content: '{_test_content[:100]}...'")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize AzureChatOpenAI: {e}")
    logger.error("Please check your environment variables:")
    logger.error(f"  - AZURE_OPENAI_API_KEY: {'✅ Set' if AZURE_OPENAI_API_KEY else '❌ Missing'}")
    logger.error(f"  - AZURE_OPENAI_ENDPOINT: {'✅ Set' if AZURE_OPENAI_ENDPOINT else '❌ Missing'}")
    logger.error(f"  - AZURE_OPENAI_DEPLOYMENT_NAME: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    logger.error("  - For reasoning models (o1/o3/o4), ensure you're using the correct API version and parameters")
    llm = None

# Initialize Gemini client (optional)
gemini_client = None
gemini_config = None
if genai and types and GEMINI_API_KEY:
    try:
        # Initialize the new Gemini client
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Setup grounding tool for Google Search
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # Configure generation settings with grounding
        gemini_config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=1.0  # Recommended for grounding
        )
        
        logger.info(f"✅ Gemini initialized with model: {GEMINI_MODEL} (Google Search grounding enabled)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {e}")
        gemini_client = None
        gemini_config = None

class CoinGeckoAPI:
    """Wrapper for CoinGecko API calls"""
    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def get_trending_coins(limit: int = 50) -> list:
        try:
            logger.info(f"🚀 Fetching {limit} trending coins from CoinGecko...")
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/search/trending")
            coins = resp.json().get('coins', [])[:limit]
            result = [c['item']['id'] for c in coins]
            logger.info(f"✅ Successfully fetched trending coins: {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Error get_trending_coins: {e}")
            return []

    @staticmethod
    def search_coins(query: str, limit: int = 25) -> list:
        import time
        try:
            logger.info(f"🔍 Searching for coins with query: '{query}', limit: {limit}")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/search", params={'query': query})
            
            if resp.status_code == 429:
                logger.warning("⚠️ Rate limited, waiting 2 seconds and retrying...")
                time.sleep(2)
                resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/search", params={'query': query})
            
            if resp.status_code != 200:
                logger.error(f"❌ CoinGecko search API returned {resp.status_code}: {resp.text[:100]}")
                return []
            
            try:
                data = resp.json()
            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON response from CoinGecko search: {resp.text[:100]}")
                return []
            
            coins = data.get('coins', [])
            
            # Prioritize exact symbol matches for better accuracy
            exact_matches = []
            general_matches = []
            
            for coin in coins:
                if not coin.get('id'):
                    continue
                    
                symbol = coin.get('symbol', '').upper()
                name = coin.get('name', '').upper()
                query_upper = query.upper()
                
                # Exact symbol match gets priority
                if symbol == query_upper:
                    exact_matches.append(coin['id'])
                # Partial matches
                elif query_upper in symbol or query_upper in name:
                    general_matches.append(coin['id'])
            
            # Return exact matches first, then general matches
            result = (exact_matches + general_matches)[:limit]
            logger.info(f"✅ Found coins: {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Error search_coins: {e}")
            return []

    @staticmethod
    def get_coin_data(coin_id: str) -> dict:
        """Fetch only essential CoinGecko data for prompt context"""
        try:
            logger.info(f"📊 Fetching data for coin: {coin_id}")
            resp = requests.get(
                f"{CoinGeckoAPI.BASE_URL}/coins/{coin_id}",
                params={'localization':'false','tickers':'false','market_data':'true','community_data':'true','developer_data':'true'}
            )
            raw = resp.json()
            platforms = raw.get("platforms") or {}
            chain_list = [str(k).lower() for k in platforms.keys()] if isinstance(platforms, dict) else []
            categories_list = raw.get("categories") or []
            # Safe extraction of market data
            market_data = raw.get("market_data", {})
            community_data = raw.get("community_data", {})
            developer_data = raw.get("developer_data", {})
            
            result = {
                "id": coin_id,
                "symbol": raw.get("symbol"),
                "price_usd": market_data.get("current_price", {}).get("usd") if isinstance(market_data.get("current_price"), dict) else None,
                "market_cap_usd": market_data.get("market_cap", {}).get("usd") if isinstance(market_data.get("market_cap"), dict) else None,
                "volume_24h_usd": market_data.get("total_volume", {}).get("usd") if isinstance(market_data.get("total_volume"), dict) else None,
                "change_24h_pct": market_data.get("price_change_percentage_24h"),
                "community_score": community_data.get("twitter_followers") if isinstance(community_data, dict) else None,
                "developer_forks": developer_data.get("forks") if isinstance(developer_data, dict) else None,
                "chains": chain_list,
                "categories": categories_list,
            }
            logger.info(f"✅ Extracted essential data for {coin_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Error get_coin_data for {coin_id}: {e}")
            return {}

    @staticmethod
    def get_categories_list() -> list:
        try:
            logger.info("📋 Fetching categories list from CoinGecko...")
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/coins/categories/list")
            categories = resp.json()
            logger.info(f"✅ Successfully fetched {len(categories)} categories")
            return categories
        except Exception as e:
            logger.error(f"❌ Error get_categories_list: {e}")
            return []

    @staticmethod
    def get_top_coins_by_category_slug(category_slug: str, limit: int = 50) -> list:
        """Fetch top coins for a CoinGecko category slug using markets endpoint"""
        try:
            logger.info(f"🏷️ Fetching top {limit} coins for category slug: {category_slug}")
            params = {
                'vs_currency': 'usd',
                'category': category_slug,
                'order': 'market_cap_desc',
                'per_page': max(1, min(limit, 50)),
                'page': 1,
                'price_change_percentage': '24h'
            }
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/coins/markets", params=params)
            data = resp.json() or []
            if not isinstance(data, list):
                if isinstance(data, dict) and 'error' in data:
                    logger.error(f"❌ CoinGecko API error for category {category_slug}: {data.get('error')}")
                else:
                    logger.error(f"❌ Error get_top_coins_by_category_slug: Expected list, got {type(data)}")
                return []
            data = data[:limit]
            simplified = []
            for c in data:
                simplified.append({
                    "id": c.get("id"),
                    "symbol": c.get("symbol"),
                    "price_usd": c.get("current_price"),
                    "market_cap_usd": c.get("market_cap"),
                    "volume_24h_usd": c.get("total_volume"),
                    "change_24h_pct": c.get("price_change_percentage_24h_in_currency")
                })
            logger.info(f"✅ Retrieved {len(simplified)} coins for category {category_slug}")
            return simplified
        except Exception as e:
            logger.error(f"❌ Error get_top_coins_by_category_slug: {e}")
            return []

    @staticmethod
    def find_coins_by_category_and_chain(category_slug: str, chain_slug: str, target_count: int = 50, pages: int = 10, per_page: int = 100) -> list:
        """Find coins in a CoinGecko category that are deployed on a specific chain.
        Fetches paginated markets data, then confirms chain via per-coin detail (platforms).
        """
        logger.info(f"🔍 Starting chain-aware search: category='{category_slug}', chain='{chain_slug}', target={target_count}")
        found: List[dict] = []
        try:
            for page in range(1, pages + 1):
                params = {
                    'vs_currency': 'usd',
                    'category': category_slug,
                    'order': 'market_cap_desc',
                    'per_page': max(1, min(per_page, 250)),
                    'page': page,
                    'price_change_percentage': '24h'
                }
                resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/coins/markets", params=params)
                data = resp.json() or []
                if not isinstance(data, list):
                    continue
                for c in data:
                    coin_id = c.get('id')
                    if not coin_id:
                        continue
                    detail = CoinGeckoAPI.get_coin_data(coin_id)
                    if not isinstance(detail, dict) or not detail:
                        continue
                    chains = detail.get('chains', [])
                    if not isinstance(chains, list):
                        chains = []
                    # Use flexible chain matching
                    def matches_chain_search(coin_chains, target_chain):
                        target_lower = target_chain.lower()
                        chain_aliases = {
                            'solana': ['solana', 'sol'],
                            'ethereum': ['ethereum', 'eth', 'erc-20'],
                            'binance': ['binance-smart-chain', 'bsc', 'bnb'],
                            'polygon': ['polygon-pos', 'polygon', 'matic'],
                            'avalanche': ['avalanche', 'avax'],
                            'cardano': ['cardano', 'ada']
                        }
                        aliases = chain_aliases.get(target_lower, [target_lower])
                        for chain in coin_chains:
                            chain_str = str(chain or '').lower()
                            if any(alias in chain_str for alias in aliases):
                                return True
                        return False
                    
                    if matches_chain_search(chains, chain_slug):
                        logger.info(f"✅ Found matching coin: {coin_id} with chains: {chains}")
                        # Merge detail with market fields for consistency
                        merged = {
                            'id': detail.get('id') or coin_id,
                            'symbol': detail.get('symbol') or c.get('symbol'),
                            'price_usd': detail.get('price_usd') or c.get('current_price'),
                            'market_cap_usd': detail.get('market_cap_usd') or c.get('market_cap'),
                            'volume_24h_usd': detail.get('volume_24h_usd') or c.get('total_volume'),
                            'change_24h_pct': detail.get('change_24h_pct') or c.get('price_change_percentage_24h_in_currency'),
                            'chains': chains,
                            'categories': detail.get('categories', []),
                        }
                        found.append(merged)
                        if len(found) >= target_count:
                            logger.info(f"🎯 Found {len(found)} target coins, returning early")
                            return found
                    else:
                        logger.debug(f"❌ Coin {coin_id} chains {chains} don't match {chain_slug}")
                if len(data) < per_page:
                    break
            return found
        except Exception as e:
            logger.error(f"❌ Error find_coins_by_category_and_chain: {e}")
            return found

    @staticmethod
    def get_trending_searches(limit: int = 50) -> list:
        """CoinGecko trending search list (lightweight)."""
        try:
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/search/trending")
            coins = resp.json().get('coins', [])[:limit]
            return [
                {
                    'id': c.get('item', {}).get('id'),
                    'symbol': c.get('item', {}).get('symbol'),
                }
                for c in coins
            ]
        except Exception:
            return []

    @staticmethod
    def get_top_market_cap(limit: int = 100) -> list:
        """Fetch top coins by market cap (simplified fields)."""
        try:
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': max(1, min(limit, 50)),
                'page': 1,
                'price_change_percentage': '24h'
            }
            resp = requests.get(f"{CoinGeckoAPI.BASE_URL}/coins/markets", params=params)
            data = resp.json()[:limit]
            simplified = []
            for c in data:
                simplified.append({
                    "id": c.get("id"),
                    "symbol": c.get("symbol"),
                    "price_usd": c.get("current_price"),
                    "market_cap_usd": c.get("market_cap"),
                    "volume_24h_usd": c.get("total_volume"),
                    "change_24h_pct": c.get("price_change_percentage_24h_in_currency")
                })
            return simplified
        except Exception:
            return []

class CoinMarketCapAPI:
    """Wrapper for CoinMarketCap API calls"""
    BASE_URL = os.getenv('CMC_BASE_URL', 'https://pro-api.coinmarketcap.com/v1')
    API_KEY = os.getenv('CMC_API_KEY')

    @staticmethod
    def get_listings_latest(limit: int = 100, convert: str = 'USD') -> list:
        try:
            logger.info(f"📋 Fetching top {limit} listings from CoinMarketCap...")
            headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CoinMarketCapAPI.API_KEY}
            params = {'start': '1', 'limit': limit, 'convert': convert}
            resp = requests.get(f"{CoinMarketCapAPI.BASE_URL}/cryptocurrency/listings/latest", headers=headers, params=params)
            data = resp.json().get('data', [])[:limit]
            logger.info(f"✅ Got {len(data)} listings from CMC")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_listings_latest: {e}")
            return []

    @staticmethod
    def get_global_metrics() -> dict:
        try:
            logger.info("📈 Fetching global market metrics from CoinMarketCap...")
            headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CoinMarketCapAPI.API_KEY}
            resp = requests.get(f"{CoinMarketCapAPI.BASE_URL}/global-metrics/quotes/latest", headers=headers)
            data = resp.json().get('data', {})
            logger.info("✅ Got global metrics from CMC")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_global_metrics: {e}")
            return {}

    @staticmethod
    def get_fear_and_greed() -> dict:
        try:
            logger.info("😱 Fetching Fear and Greed Index from CoinMarketCap...")
            headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CoinMarketCapAPI.API_KEY}
            resp = requests.get(f"{CoinMarketCapAPI.BASE_URL}/fear-and-greed/latest", headers=headers)
            data = resp.json().get('data', {})
            logger.info("✅ Got Fear and Greed Index from CMC")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_fear_and_greed: {e}")
            return {}

    @staticmethod
    def get_listings_by_category(category_id: str, limit: int = 100, convert: str = 'USD') -> list:
        """Fetch coins by CMC category"""
        try:
            logger.info(f"🏷️ Fetching {limit} coins from CMC category: {category_id}")
            headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CoinMarketCapAPI.API_KEY}
            params = {'id': category_id, 'start': '1', 'limit': limit, 'convert': convert}
            resp = requests.get(f"{CoinMarketCapAPI.BASE_URL}/cryptocurrency/category", headers=headers, params=params)
            data = resp.json().get('data', {}).get('coins', [])[:limit]
            logger.info(f"✅ Got {len(data)} coins from CMC category {category_id}")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_listings_by_category: {e}")
            return []

    @staticmethod
    def get_quotes_latest(symbols: list, convert: str = 'USD') -> dict:
        """Fetch only essential CMC quote fields"""
        try:
            logger.info(f"📑 Fetching CMC quotes for symbols: {symbols}")
            headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CoinMarketCapAPI.API_KEY}
            params = {'symbol': ','.join(symbols), 'convert': convert}
            resp = requests.get(f"{CoinMarketCapAPI.BASE_URL}/cryptocurrency/quotes/latest", headers=headers, params=params)
            raw = resp.json().get('data', {})
            result = {
                sym: {
                    "price": info["quote"][convert].get("price"),
                    "market_cap": info["quote"][convert].get("market_cap"),
                    "change_24h_pct": info["quote"][convert].get("percent_change_24h")
                }
                for sym, info in raw.items()
            }
            logger.info("✅ Extracted essential CMC quotes")
            return result
        except Exception as e:
            logger.error(f"❌ Error get_quotes_latest: {e}")
            return {}

class DefiLlamaAPI:
    """Wrapper for DefiLlama free-tier API calls"""
    BASE_URL = 'https://api.llama.fi'

    @staticmethod
    def get_protocols() -> list:
        try:
            logger.info("📊 Fetching DefiLlama protocols TVL data...")
            resp = requests.get(f"{DefiLlamaAPI.BASE_URL}/protocols")
            data = resp.json()
            logger.info(f"✅ Retrieved {len(data)} DefiLlama protocols")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_protocols: {e}")
            return []

    @staticmethod
    def get_chains() -> list:
        try:
            logger.info("🌐 Fetching DefiLlama chains TVL data...")
            resp = requests.get(f"{DefiLlamaAPI.BASE_URL}/v2/chains")
            data = resp.json()
            logger.info(f"✅ Retrieved {len(data)} DefiLlama chains")
            return data
        except Exception as e:
            logger.error(f"❌ Error get_chains: {e}")
            return []

    @staticmethod
    def get_protocol_tvl(protocol_slug: str) -> float:
        """Fetch simplified DefiLlama TVL value"""
        try:
            logger.info(f"📊 Fetching DefiLlama TVL for protocol: {protocol_slug}")
            resp = requests.get(f"{DefiLlamaAPI.BASE_URL}/tvl/{protocol_slug}")
            raw = resp.json()
            tvl_value = raw if isinstance(raw, (int, float)) else 0.0
            logger.info(f"✅ Simplified TVL for {protocol_slug}: {tvl_value}")
            return tvl_value
        except Exception as e:
            logger.error(f"❌ Error get_protocol_tvl for {protocol_slug}: {e}")
            return 0.0
    

class AlphaGenerator:
    """Core AI logic for generating alpha content using LangChain with ENHANCED DEBUGGING"""

    @staticmethod
    def create_crypto_analysis_prompt(coin_data: list, user_query: str, cmc_data: dict = {}, llama_data: dict = {}, web_context: Optional[list] = None, google_trends: dict = {}) -> str:
        """Create a structured prompt for crypto analysis"""
        prompt = f"""
You are an expert crypto analyst creating high-quality alpha content for a premium Discord community.

USER QUERY: {user_query}
    
COIN DATA: {json.dumps(coin_data, indent=2)}

# Additional CoinMarketCap API Data
CMC DATA: {json.dumps(cmc_data, indent=2) if cmc_data else {}}

# DefiLlama TVL Data
LLAMA DATA: {json.dumps(llama_data, indent=2) if llama_data else {}}

# Web Context from initial search crawl
WEB CONTEXT: {json.dumps(web_context, indent=2) if web_context else []}

# Google Trends Data
GOOGLE TRENDS: {json.dumps(google_trends, indent=2) if google_trends else {}}

Create a professional analysis that includes:

1. Executive Summary (2-3 sentences)
2. Key Metrics & Performance (price, market cap, volume)
3. Market Analysis
4. Risk Assessment
5. Community & Developer Metrics
6. Actionable Insights

Format the response in markdown with headings (##) and bullet lists for clarity. Use emojis sparingly and professionally. Keep it concise but comprehensive (300-500 words).

Ensure to include specific price data, percentage changes, market cap information, community engagement metrics, and developer activity from the provided data.

IMPORTANT: You MUST provide a complete response. Do not leave the content empty.
"""
        logger.info("📝 Created crypto analysis prompt")
        logger.info(f"📏 Prompt length: {len(prompt)} characters")
        return prompt

    @staticmethod
    def generate_content(prompt: str) -> tuple[str, Dict[str, Any]]:
        """Generate content using LangChain AzureChatOpenAI with DETAILED DEBUGGING
        Returns: (content, llm_debug)
        """
        try:
            logger.info("🤖 Generating content using LangChain AzureChatOpenAI...")
            if not llm:
                logger.error("❌ LangChain LLM not initialized. Cannot generate content.")
                return ("Error: AI service not available. Please check configuration.", {'provider': 'none'})
            if not llm:
                return ("Error: AI service not available. Please check configuration.", {'provider': 'none'})
            logger.info("🔧 Model Configuration - Temperature: 1.0, Max Completion Tokens: 4096, Reasoning Effort: high, Context: 200k")
            logger.info(f"📏 Input prompt length: {len(prompt)} characters")
            
            # Trim prompt to fit within ~200k token context window for o4-mini
            try:
                encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            system_text = "You are a professional crypto analyst who creates premium alpha content for paid communities. Always provide complete, detailed responses. At the end, include a section titled 'FINAL ANSWER' summarizing the key insights."
            system_tokens = len(encoding.encode(system_text))
            prompt_tokens = len(encoding.encode(prompt))
            total_tokens_est = system_tokens + prompt_tokens
            max_context_tokens = 200000
            reserved_output_tokens = 4096
            prompt_trimmed = False
            trimmed_prompt_tokens = 0
            if total_tokens_est > (max_context_tokens - reserved_output_tokens):
                allowed = max_context_tokens - reserved_output_tokens - system_tokens
                if allowed > 0:
                    trimmed = encoding.encode(prompt)[:allowed]
                    prompt = encoding.decode(trimmed)
                    prompt_trimmed = True
                    trimmed_prompt_tokens = prompt_tokens - len(trimmed)
                    logger.info(f"✂️ Trimmed prompt to fit context window. Allowed tokens: {allowed}")

            # Create messages using LangChain message types
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=prompt)
            ]
            
            logger.info(f"📨 Total messages: {len(messages)}")
            logger.info(f"📝 System message length: {len(messages[0].content)} characters")
            logger.info(f"📝 User message length: {len(messages[1].content)} characters")
            
            # Generate response using LangChain's invoke method
            logger.info("🚀 Invoking LLM...")
            response = llm.invoke(messages)
            
            # DETAILED RESPONSE DEBUGGING
            logger.info("🔍 RESPONSE DEBUGGING:")
            logger.info(f"  - Response type: {type(response)}")
            logger.info(f"  - Response object: {response}")
            
            # Check if response has content attribute
            if hasattr(response, 'content'):
                content = _coerce_content_to_str(response.content)
                logger.info(f"  - Content type: {type(content)}")
                logger.info(f"  - Content length: {len(content) if content else 0} characters")
                logger.info(f"  - Content preview: {repr(content[:200])}")
            else:
                logger.error("  - ❌ Response object has no 'content' attribute")
                content = ""
            
            # Check if response has usage_metadata for token information
            token_usage: Dict[str, Any] = {}
            model_name: Optional[str] = None
            finish_reason: Optional[str] = None
            response_metadata: Dict[str, Any] = {}
            if hasattr(response, 'usage_metadata'):
                usage = getattr(response, 'usage_metadata')  # type: ignore[assignment]
                logger.info(f"  - Token usage: {usage}")
                token_usage = dict(usage) if isinstance(usage, dict) else token_usage
            if hasattr(response, 'response_metadata'):
                response_metadata = getattr(response, 'response_metadata')  # type: ignore[assignment]
                logger.info(f"  - Response metadata: {response_metadata}")
                model_name = response_metadata.get('model_name') or model_name
                finish_reason = response_metadata.get('finish_reason') or finish_reason
                if 'token_usage' in response_metadata:
                    token_usage = response_metadata['token_usage']
                    logger.info(f"  - Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
                    logger.info(f"  - Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
                    logger.info(f"  - Total tokens: {token_usage.get('total_tokens', 'N/A')}")
            
            # Handle empty content
            llm_debug: Dict[str, Any] = {
                'model_name': model_name,
                'finish_reason': finish_reason,
                'token_usage': token_usage,
                'prompt_trimmed': prompt_trimmed,
                'trimmed_prompt_tokens': trimmed_prompt_tokens,
                'max_completion_tokens': 4096,
                'reasoning_effort': 'high',
            }

            if not content or len(content.strip()) == 0:
                logger.error("❌ EMPTY RESPONSE DETECTED!")
                logger.error("💡 This usually happens when:")
                logger.error("   1. All completion tokens are used for reasoning")
                logger.error("   2. Response was filtered by content policy")
                logger.error("   3. Model hit token limit during reasoning")
                logger.error("🔧 Recommended fixes:")
                logger.error("   1. Increase max_completion_tokens")
                logger.error("   2. Reduce prompt complexity")
                logger.error("   3. Check content filtering policies")

                # Retry once with higher output token budget and lower reasoning effort
                try:
                    logger.info("♻️ Retrying LLM with increased output tokens and lower reasoning effort...")
                    llm_retry = AzureChatOpenAI(
                        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                        api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
                        api_version=AZURE_OPENAI_API_VERSION,
                        temperature=1.0,
                        model_kwargs={
                            "max_completion_tokens": 8192,
                            "reasoning_effort": "high",
                            "response_format": {"type": "text"}
                        },
                        timeout=60,
                        max_retries=3
                    )
                    response2 = llm_retry.invoke(messages)
                    content2 = _coerce_content_to_str(getattr(response2, 'content', ''))
                    if content2 and len(content2.strip()) > 0:
                        content = content2
                        llm_debug['retry_used'] = True
                        llm_debug['retry_reason'] = 'empty_content_first_call'
                    else:
                        # Second retry: drastically simplified prompt
                        logger.info("♻️ Second retry with simplified prompt")
                        simple_prompt = (
                            "Provide a concise market analysis with: Executive Summary, Key Metrics, Risks, and Actionable Insights. "
                            "Focus on symbols and categories mentioned by the user."
                        )
                        response3 = llm_retry.invoke([
                            SystemMessage(content=(
                                "You are a professional crypto analyst. Provide a direct, concise answer with headings. "
                                "Include a 'FINAL ANSWER' section."
                            )),
                            HumanMessage(content=simple_prompt)
                        ])
                        content = _coerce_content_to_str(getattr(response3, 'content', ''))
                        if not content:
                            content = "Analysis is currently unavailable. Please try again shortly."
                        llm_debug['retry_used'] = True
                        llm_debug['retry_reason'] = 'empty_content_second_call'
                except Exception as retry_e:
                    logger.error(f"❌ Retry failed: {retry_e}")
                    content = "Analysis is currently unavailable. Please try again shortly."
                    llm_debug['retry_used'] = True
                    llm_debug['retry_reason'] = f'exception:{type(retry_e).__name__}'
            
            logger.info(f"✅ Successfully processed content (final length: {len(content)} characters)")
            
            # Log the actual content being returned to console
            logger.info("📄 GENERATED CONTENT (Console Output):")
            logger.info("=" * 80)
            logger.info(content)
            logger.info("=" * 80)
            
            return (content, llm_debug)

        except Exception as e:
            logger.error(f"❌ Error generating content with LangChain: {e}")
            logger.error(f"💡 Exception type: {type(e)}")
            logger.error(f"💡 Exception args: {e.args}")
            return (f"Error generating analysis: {str(e)}. Please try again with a simpler query.", {'error': str(e)})

    @staticmethod
    def gemini_grounded_sources(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if gemini_client is None or gemini_config is None:
            return results
        try:
            prompt = f"Find recent information about: {query}. Focus on crypto/blockchain context if relevant. Provide specific sources and URLs."
            
            # Use the new Gemini API with Google Search grounding
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=gemini_config
            )
            
            # Extract grounding metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        grounding = candidate.grounding_metadata
                        # Extract web results from grounding metadata
                        if hasattr(grounding, 'web_search_queries') and grounding.web_search_queries:
                            for query_result in grounding.web_search_queries[:max_results]:
                                if hasattr(query_result, 'search_results') and getattr(query_result, 'search_results', None):
                                    search_results = cast(Any, getattr(query_result, 'search_results', []))
                                    for search_result in search_results[:max_results]:
                                        if hasattr(search_result, 'title') and hasattr(search_result, 'url'):
                                            results.append({
                                                'title': search_result.title,
                                                'url': search_result.url
                                            })
                        # Alternative: extract from grounding chunks
                        elif hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                            for chunk in grounding.grounding_chunks[:max_results]:
                                if hasattr(chunk, 'web') and chunk.web:
                                    results.append({
                                        'title': getattr(chunk.web, 'title', 'Web Source'),
                                        'url': getattr(chunk.web, 'uri', '')
                                    })
            
            # Fallback: extract URLs from response text
            if not results and hasattr(response, 'text') and response.text:
                for m in re.findall(r'https?://\S+', str(response.text))[:max_results]:
                    results.append({'title': 'Source', 'url': m})
                        
        except Exception as e:
            logger.error(f"❌ Gemini grounded source extraction failed: {e}")
            # Fallback to regular Gemini call without grounding
            try:
                fallback_response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"List relevant sources about: {query}"
                )
                if hasattr(fallback_response, 'text') and fallback_response.text:
                    for m in re.findall(r'https?://\S+', str(fallback_response.text))[:max_results]:
                        results.append({'title': 'Source', 'url': m})
            except Exception as fallback_error:
                logger.error(f"❌ Gemini fallback also failed: {fallback_error}")
        
        return results

    @staticmethod
    def gemini_suggest_tokens(query: str, chain_hint: Optional[str], category_hint: Optional[str], max_items: int = 8) -> List[str]:
        suggestions: List[str] = []
        if gemini_client is None:
            return suggestions
        try:
            if chain_hint and category_hint:
                # More specific prompt for chain+category combos
                prompt = f"List the top {max_items} {category_hint} tokens on {chain_hint} blockchain. Return only token symbols separated by commas."
            else:
                instruction = (
                    "Identify crypto tokens matching the user's constraints. "
                    f"Return STRICT JSON with key 'tokens' as a list of up to {max_items} token names or symbols. No extra text."
                )
                parts = [f"User query: {query}"]
                if chain_hint:
                    parts.append(f"Chain constraint: {chain_hint}")
                if category_hint:
                    parts.append(f"Category constraint: {category_hint}")
                prompt = instruction + "\n" + "\n".join(parts)
            
            # Use the new Gemini API
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            
            text = response.text if hasattr(response, 'text') else None
            if not text and hasattr(response, 'candidates') and response.candidates:
                text = '\n'.join([cast(Any, p).text for p in response.candidates if getattr(p, 'text', None)])
            if not text:
                return suggestions
            
            # Try to extract JSON from the response
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract tokens from text using different patterns
                logger.warning(f"⚠️ JSON parsing failed, trying text extraction from: {text[:200]}...")
                
                # Try comma-separated format first
                if ',' in text:
                    token_matches = [t.strip().upper() for t in text.split(',') if t.strip()]
                else:
                    # Try regex for token symbols (2-10 characters, mostly uppercase)
                    token_matches = re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b', text)
                    if not token_matches:
                        # Try to find any token-like words (case insensitive)
                        token_matches = re.findall(r'\b[A-Za-z][A-Za-z0-9]{1,9}\b', text)
                
                for t in token_matches[:max_items]:
                    if isinstance(t, str) and 2 <= len(t) <= 10:
                        suggestions.append(t.upper())
                return suggestions
            for t in data.get('tokens', [])[:max_items]:
                if isinstance(t, str) and 1 <= len(t) <= 40:
                    suggestions.append(t)
        except Exception as e:
            logger.error(f"❌ Gemini token suggestion failed: {e}")
        return suggestions

    @staticmethod
    def generate_analysis(query: str) -> tuple[str, Dict[str, Any]]:
        """Fetch comprehensive crypto data and generate generalized analysis. Returns (content, debug)."""
        logger.info(f"🎯 Starting generalized analysis generation for query: '{query}'")
        # Initialize Google Trends data
        google_trends_data: dict = {}
        analysis_debug: Dict[str, Any] = {}
        
        # Parse intent from query for basic parameters only
        intent = QueryParser.parse(query)
        analysis_debug['intent'] = intent
        logger.info(f"🎯 Parsed intent (generalized): {intent}")

        # Perform grounded search for additional context using Gemini
        try:
            logger.info("🌐 Performing expanded grounded search for comprehensive context (Gemini)")
            sources = AlphaGenerator.gemini_grounded_sources(query, max_results=20) if gemini_client else []  # Increased results
            web_context = sources
            # Fetch free-tier Google Trends data
            google_trends_data = GoogleTrendsAPI.get_interest_over_time(query)
            logger.info(f"📈 Retrieved Google Trends data points: {len(google_trends_data)}")
            logger.debug(f"🔍 Google Trends raw data: {google_trends_data}")
        except Exception as e:
            logger.error(f"❌ Grounded search failed: {e}")
            web_context = []
            google_trends_data = {}
        
        coin_data = []
        analysis_debug['strategy'] = 'comprehensive_gathering'
        num = intent.get('num', 50)  # Default to broader results
        
        logger.info(f"🎆 Starting comprehensive crypto data gathering (target: {num} coins)")
        
        # Strategy 1: If user provided explicit symbols, prioritize those
        symbols = intent.get('symbols') or []
        if symbols:
            logger.info(f"🎯 Processing explicit symbols: {symbols}")
            for sym in symbols:
                found_ids = CoinGeckoAPI.search_coins(sym, limit=5)  # Get multiple matches per symbol
                for coin_id in found_ids:
                    data = CoinGeckoAPI.get_coin_data(coin_id)
                    if data:
                        coin_data.append(data)
        
        # Strategy 2: Use Gemini for intelligent token suggestions based on query
        if gemini_client:
            logger.info(f"🤖 Getting Gemini suggestions for: '{query}'")
            suggested = AlphaGenerator.gemini_suggest_tokens(query, None, None, max_items=min(30, num))  # No constraints
            for i, token_name in enumerate(suggested):
                if i > 0 and i % 10 == 0:  # Occasional delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
                found_ids = CoinGeckoAPI.search_coins(token_name, limit=2)
                for coin_id in found_ids:
                    data = CoinGeckoAPI.get_coin_data(coin_id)
                    if data and not any(existing.get('id') == data.get('id') for existing in coin_data):
                        coin_data.append(data)
        
        # Strategy 3: Direct search on the query itself
        logger.info(f"🔍 Performing direct search on query: '{query}'")
        direct_search_ids = CoinGeckoAPI.search_coins(query, limit=20)
        for coin_id in direct_search_ids:
            data = CoinGeckoAPI.get_coin_data(coin_id)
            if data and not any(existing.get('id') == data.get('id') for existing in coin_data):
                coin_data.append(data)
        
        # Strategy 4: Get trending coins to ensure we have popular tokens
        logger.info("📈 Adding trending coins for market context")
        trending_ids = CoinGeckoAPI.get_trending_coins(limit=min(20, num//2))
        for coin_id in trending_ids:
            data = CoinGeckoAPI.get_coin_data(coin_id)
            if data and not any(existing.get('id') == data.get('id') for existing in coin_data):
                coin_data.append(data)
        
        # Strategy 5: Get top market cap coins for market leaders
        if len(coin_data) < num:
            logger.info("📉 Adding top market cap coins for comprehensive coverage")
            top_mc_coins = CoinGeckoAPI.get_top_market_cap(limit=min(50, num - len(coin_data)))
            for coin in top_mc_coins:
                if not any(existing.get('id') == coin.get('id') for existing in coin_data):
                    coin_data.append(coin)

        # Final safety fallback to ensure we always have data to analyze
        if not coin_data:
            logger.info("🧭 Final safety fallback: BTC, ETH, and top market cap coins")
            analysis_debug['fallback_used'] = True
            # Get the major coins as baseline
            for major_coin in ["bitcoin", "ethereum", "binancecoin", "solana", "cardano"]:
                data = CoinGeckoAPI.get_coin_data(major_coin)
                if data:
                    coin_data.append(data)
            
            # Add some top market cap coins
            top_mc = CoinGeckoAPI.get_top_market_cap(limit=20)
            for coin in top_mc:
                if not any(existing.get('id') == coin.get('id') for existing in coin_data):
                    coin_data.append(coin)

        if not coin_data:
            logger.warning("⚠️ Unable to gather any crypto data")
            analysis_debug['error'] = 'no_data_available'
            return ("Unable to gather cryptocurrency data at this time. Please try again later.", analysis_debug)

        logger.info(f"📊 Found {len(coin_data)} coins for analysis")
        logger.info(f"📊 Coin data size: {len(str(coin_data))} characters")
        
        # Trim to desired number if we have too many results
        if len(coin_data) > num:
            # Sort by market cap (if available) to prioritize larger coins
            coin_data.sort(key=lambda x: x.get('market_cap_usd', 0) if x.get('market_cap_usd') else 0, reverse=True)
            coin_data = coin_data[:num]
        
        logger.info(f"📅 Gathered {len(coin_data)} coins for comprehensive analysis")
        logger.info(f"📅 Total data size: {len(str(coin_data))} characters")
        
        analysis_debug['coin_data_count'] = len(coin_data)
        analysis_debug['web_context_count'] = len(web_context) if 'web_context' in locals() and isinstance(web_context, list) else 0
        analysis_debug['google_trends_points'] = len(google_trends_data)

        # Fetch additional market data from multiple sources for comprehensive analysis
        symbols = [c.get("symbol") for c in coin_data if c.get("symbol")]
        logger.info(f"💰 Fetching additional market data for {len(symbols)} symbols")
        cmc_quotes = CoinMarketCapAPI.get_quotes_latest(symbols)
        cmc_data = {"quotes": cmc_quotes}

        # Fetch DefiLlama TVL data for DeFi protocols
        slugs = [c.get("id") for c in coin_data if c.get("id")]
        logger.info(f"📊 Fetching TVL data for {len(slugs)} protocols")
        llama_tvl = {slug: DefiLlamaAPI.get_protocol_tvl(slug) for slug in slugs[:20]}  # Limit to avoid too many calls
        llama_data = {"tvl": llama_tvl}

        prompt = AlphaGenerator.create_crypto_analysis_prompt(coin_data, query, cmc_data, llama_data, web_context, google_trends_data)
        analysis_result, llm_debug = AlphaGenerator.generate_content(prompt)
        analysis_debug['llm'] = llm_debug
        
        logger.info(f"🎯 FINAL ANALYSIS RESULT LENGTH: {len(analysis_result)} characters")
        return (analysis_result, analysis_debug)

@app.route('/')
def index():
    """Main dashboard"""
    logger.info("🏠 Serving dashboard page")
    return render_template('dashboard.html')

@app.route('/api/generate', methods=['POST'])
def generate_alpha():
    """Core endpoint to generate alpha content with ENHANCED DEBUGGING"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        logger.info(f"📥 Received alpha generation request: '{query}'")
        logger.info(f"📏 Query length: {len(query)} characters")
        
        if not query:
            logger.warning("⚠️ Empty query received")
            return jsonify({'error': 'Query is required'}), 400

        if not llm:
            logger.error("❌ LangChain LLM not available")
            return jsonify({'error': 'AI service not available. Check reasoning model configuration.'}), 503

        # Generate analysis with detailed logging
        logger.info("🔄 Starting analysis generation...")
        analysis, analysis_debug = AlphaGenerator.generate_analysis(query)
        logger.info(f"🔄 Analysis generation completed. Result length: {len(analysis)} characters")
        
        # LOG THE RESPONSE TO CONSOLE
        logger.info("📋 FINAL API RESPONSE CONTENT:")
        logger.info("-" * 60)
        logger.info(analysis)
        logger.info("-" * 60)
        
        response_data = {
            'success': True,
            'analysis': analysis,
            'query': query,
            'generated_at': datetime.utcnow().isoformat(),
            'debug_info': {
                'response_length': len(analysis),
                'query_length': len(query),
                'model_deployment': AZURE_OPENAI_DEPLOYMENT_NAME,
                'api_version': AZURE_OPENAI_API_VERSION,
                'parameter_type': 'max_completion_tokens',
                'reasoning_model': True,
                'analysis': analysis_debug,
                'provider': 'azure',
                'gemini_model': GEMINI_MODEL if gemini_client else None,
            }
        }
        
        logger.info("✅ Successfully generated alpha content response")
        logger.info(f"📤 Sending response with {len(analysis)} characters to frontend")
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error in generate_alpha endpoint: {e}")
        logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"📚 Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/share', methods=['POST'])
def share_to_discord():
    """Share generated content to Discord"""
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        title = data.get('title', '🚀 Fresh Alpha Alert')
        
        logger.info(f"📤 Sharing content to Discord: '{title}'")
        logger.info(f"📏 Content length: {len(content)} characters")

        if not content:
            logger.warning("⚠️ Empty content for Discord share")
            return jsonify({'error': 'Content is required'}), 400

        embed = {
            "title": title,
            "description": content[:4096],  # Discord embed limit
            "color": 0x00ff00,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Generated by Alpha Co-Pilot (LangChain + Azure OpenAI)"}
        }

        payload = {"embeds": [embed], "username": "Alpha Co-Pilot"}
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        if not webhook_url:
            logger.error("❌ Discord webhook URL not configured")
            return jsonify({'error': 'Discord webhook not configured'}), 500

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 204:
            logger.info("✅ Successfully posted to Discord")
            return jsonify({'success': True, 'message': 'Posted to Discord successfully!'})
        
        logger.error(f"❌ Discord API returned status {response.status_code}")
        return jsonify({'error': 'Failed to post to Discord'}), 500

    except Exception as e:
        logger.error(f"❌ Error in share_to_discord: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint with detailed diagnostics"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'langchain_llm': 'available' if llm else 'unavailable',
        'gemini': 'available' if gemini_client and gemini_config else 'unavailable',
        'azure_openai_endpoint': AZURE_OPENAI_ENDPOINT is not None,
        'azure_openai_deployment': AZURE_OPENAI_DEPLOYMENT_NAME,
        'model_type': 'reasoning_model' if any(x in AZURE_OPENAI_DEPLOYMENT_NAME.lower() for x in ['o1', 'o3', 'o4']) else 'standard_model',
        'parameter_compatibility': 'max_completion_tokens',
        'api_version': AZURE_OPENAI_API_VERSION,
        'max_tokens_config': 2000,
        'reasoning_effort': 'medium'
    }
    
    logger.info(f"🔍 Health check: {health_status}")
    return jsonify(health_status)

if __name__ == '__main__':
    logger.info("🚀 Starting Flask application with ENHANCED DEBUGGING...")
    logger.info("🔧 Environment check:")
    logger.info(f"  - Azure OpenAI Endpoint: {'✅ Set' if AZURE_OPENAI_ENDPOINT else '❌ Missing'}")
    logger.info(f"  - Azure OpenAI API Key: {'✅ Set' if AZURE_OPENAI_API_KEY else '❌ Missing'}")
    logger.info(f"  - Deployment Name: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    logger.info(f"  - API Version: {AZURE_OPENAI_API_VERSION}")
    logger.info(f"  - Model Type: {'Reasoning Model (o1/o3/o4)' if any(x in AZURE_OPENAI_DEPLOYMENT_NAME.lower() for x in ['o1', 'o3', 'o4']) else 'Standard Model'}")
    logger.info(f"  - LangChain LLM: {'✅ Ready' if llm else '❌ Not Available'}")
    logger.info("  - Max Completion Tokens: 2000")
    logger.info("  - Reasoning Effort: medium")
    logger.info("🔍 Debug mode: ENABLED - All responses will be logged to console")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
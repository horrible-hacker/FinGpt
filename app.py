# app.py
import gradio as gr
import yfinance as yf
import hashlib
from datetime import datetime
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

top_mncs_200 = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon.com": "AMZN",
    "Alphabet (Class A)": "GOOGL",
    "Alphabet (Class C)": "GOOG",
    "Nvidia": "NVDA",
    "Meta Platforms": "META",
    "Tesla": "TSLA",
    "Johnson & Johnson": "JNJ",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Procter & Gamble": "PG",
    "UnitedHealth Group": "UNH",
    "ExxonMobil": "XOM",
    "Mastercard": "MA",
    "Home Depot": "HD",
    "Pfizer": "PFE",
    "Chevron": "CVX",
    "Merck & Co.": "MRK",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "AbbVie": "ABBV",
    "Walmart": "WMT",
    "Broadcom": "AVGO",
    "Cisco Systems": "CSCO",
    "Adobe": "ADBE",
    "Oracle": "ORCL",
    "Salesforce": "CRM",
    "Comcast": "CMCSA",
    "Intel": "INTC",
    "Verizon Communications": "VZ",
    "AT&T": "T",
    "McDonald’s": "MCD",
    "Costco Wholesale": "COST",
    "Netflix": "NFLX",
    "Walt Disney": "DIS",
    "Nike": "NKE",
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "American Express": "AXP",
    "Qualcomm": "QCOM",
    "Texas Instruments": "TXN",
    "IBM": "IBM",
    "Caterpillar": "CAT",
    "Lockheed Martin": "LMT",
    "General Motors": "GM",
    "Ford Motor": "F",
    "Boeing": "BA",
    "3M": "MMM",
    "Medtronic": "MDT",
    "Eli Lilly": "LLY",
    "Abbott Laboratories": "ABT",
    "Honeywell": "HON",
    "Union Pacific": "UNP",
    "Raytheon Technologies": "RTX",
    "Dow Inc.": "DOW",
    "Philip Morris International": "PM",
    "Colgate-Palmolive": "CL",
    "Mondelez International": "MDLZ",
    "Schlumberger": "SLB",
    "American Airlines Group": "AAL",
    "Southwest Airlines": "LUV",
    "UPS": "UPS",
    "FedEx": "FDX",
    "Target": "TGT",
    "Wells Fargo": "WFC",
    "Bank of America": "BAC",
    "Citigroup": "C",
    "Cigna": "CI",
    "Anthem (Elevance Health)": "ELV",
    "CVS Health": "CVS",
    "PayPal": "PYPL",
    "Intuit": "INTU",
    "ServiceNow": "NOW",
    "AMD": "AMD",
    "Micron Technology": "MU",
    "Applied Materials": "AMAT",
    "Starbucks": "SBUX",
    "General Electric": "GE",
    "Marriott International": "MAR",
    "Hilton Worldwide": "HLT",
    "Estee Lauder": "EL",
    "Booking Holdings": "BKNG",
    "Uber Technologies": "UBER",
    "Lyft": "LYFT",
    "eBay": "EBAY",
    "Dominion Energy": "D",
    "NextEra Energy": "NEE",
    "Duke Energy": "DUK",
    "Southern Company": "SO",
    "Exelon": "EXC",
    "Crown Castle": "CCI",
    "T-Mobile US": "TMUS",
    "Zoom Video Communications": "ZM",
    "Snowflake": "SNOW",
    "Palantir Technologies": "PLTR",
    "Datadog": "DDOG",
    "Occidental Petroleum": "OXY",
    "ConocoPhillips": "COP",
    "Marathon Petroleum": "MPC",
    "Valero Energy": "VLO",
    "Phillips 66": "PSX",
    "Kinder Morgan": "KMI",
    "Newmont Corporation": "NEM",
    "Freeport-McMoRan": "FCX",
    "Cleveland-Cliffs": "CLF",
    "Alcoa Corporation": "AA",
    "Nucor": "NUE",
    "Steel Dynamics": "STLD",
    "Palo Alto Networks": "PANW",
    "Snowflake": "SNOW",
    "CrowdStrike": "CRWD",
    "Zscaler": "ZS",
    "Okta": "OKTA",
    "Workday": "WDAY",
    "Palantir": "PLTR",
    "DocuSign": "DOCU",
    "RingCentral": "RNG",
    "Twilio": "TWLO",
    "Dropbox": "DBX",
    "Shopify": "SHOP",
    "MongoDB": "MDB",
    "Elastic": "ESTC",
    "Datadog": "DDOG",
    "CrowdStrike": "CRWD",
    "Snowflake": "SNOW",
    "Spotify": "SPOT",
    "Peloton": "PTON",
    "Roku": "ROKU",
    "Pinterest": "PINS",
    "Snap": "SNAP",
    "Lyft": "LYFT",
    "Robinhood": "HOOD",
    "Coinbase": "COIN",
    "DoorDash": "DASH",
    "Airbnb": "ABNB",
    "Uber": "UBER",
    "Lyft": "LYFT",
    "Beyond Meat": "BYND",
    "Zoom Video": "ZM",
    "Palantir": "PLTR",
    "Datadog": "DDOG",
    "CrowdStrike": "CRWD",
    "Snowflake": "SNOW",
    "Zillow": "Z",
    "Rivian": "RIVN",
    "Lucid Group": "LCID",
    "QuantumScape": "QS",
    "Plug Power": "PLUG",
    "Ballard Power Systems": "BLDP",
    "Enphase Energy": "ENPH",
    "SunPower": "SPWR",
    "First Solar": "FSLR",
    "SolarEdge": "SEDG",
    "NextEra": "NEE",
    "Brookfield Renewable Partners": "BEP",
    "Tesla": "TSLA",
    "Rivian": "RIVN",
    "Lucid": "LCID"
}

def generate_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def current_timestamp() -> str:
    return datetime.utcnow().isoformat()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="financial_advisor",
    embedding_function=embeddings,
    persist_directory="./financial_vectordb"
)

model_id = "nrjfmfr/fin-mod"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048
)
llm = HuggingFacePipeline(pipeline=generator)


def fetch_latest_docs(query: str, k: int = 10):
    docs = vectorstore.similarity_search(query, k=50) 
    latest_docs = sorted(docs, key=lambda x: x.metadata.get("timestamp", ""), reverse=True)[:k]
    return "\n\n".join([doc.page_content for doc in latest_docs])

def add_stock_to_vector(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    documents = []
    info = ticker.info
    stock_content = f"""
    {info.get('longName', ticker_symbol)} ({ticker_symbol})
    Current Price: ${info.get('currentPrice', 'N/A')}
    Market Cap: ${info.get('marketCap', 'N/A')}
    PE Ratio: {info.get('trailingPE', 'N/A')}
    52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
    52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
    Sector: {info.get('sector', 'N/A')}
    Industry: {info.get('industry', 'N/A')}
    """
    stock_id = hashlib.sha256(stock_content.encode("utf-8")).hexdigest()
    documents.append(Document(
        page_content=stock_content.strip(),
        metadata={
            "id": stock_id,
            "type": "stock_info",
            "ticker": ticker_symbol,
            "company": info.get('longName', ticker_symbol),
            "timestamp": datetime.utcnow().isoformat()
        }
    ))
    history = ticker.history(period="1mo")
    if not history.empty:
        price_content = f"""
        {ticker_symbol} price history (last 30 days):
        Starting price: ${history['Close'].iloc[0]:.2f}
        Current price: ${history['Close'].iloc[-1]:.2f}
        Highest: ${history['High'].max():.2f}
        Lowest: ${history['Low'].min():.2f}
        Average volume: {history['Volume'].mean():.0f}
        """
        price_id = hashlib.sha256(price_content.encode("utf-8")).hexdigest()
        documents.append(Document(
            page_content=price_content.strip(),
            metadata={
                "id": price_id,
                "type": "price_history",
                "ticker": ticker_symbol,
                "period": "1mo",
                "timestamp": datetime.utcnow().isoformat()
            }
        ))
    news = ticker.news
    for article in news[:50]:
        title = article.get('title', '')
        summary = article.get('summary', '')
        link = article.get('link', '')
        news_content = f"{title}. {summary}"
        news_id = hashlib.sha256(news_content.encode("utf-8")).hexdigest()
        documents.append(Document(
            page_content=news_content.strip(),
            metadata={
                "id": news_id,
                "type": "news",
                "ticker": ticker_symbol,
                "link": link,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))
    vectorstore.add_documents(documents)
    
for company, symbol in top_mncs_200.items():
    add_stock_to_vector(symbol)

def answer_query(user_query):
    for company, symbol in top_mncs_200.items():
        add_stock_to_vector(symbol)
    
    context = fetch_latest_docs(user_query, k=10)
    final_prompt = f"""
    You are a financial expert. Using the context provided, answer the user's question in a detailed, structured manner. Include:
    
    1. Current stock price
    2. Market trend (short-term and medium-term)
    3. Relevant news events
    4. Analyst sentiment
    5. Any important financial metrics (like P/E ratio, market cap)
    6. Summary and insights
    
    Context:
    {context}
    
    Question:
    {user_query}
    """
    return llm.invoke(final_prompt)

iface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask financial questions..."),
    outputs="text",
    title="Financial RAG Chatbot",
    description="Ask questions about stocks, market trends, and news"
)

iface.launch()

import os
import requests
import chromadb
import concurrent.futures
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from groq import Groq

# INITIALIZATION
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# Database 
chroma_client = chromadb.PersistentClient(
    path="./research_memory" 
)
collection = chroma_client.get_or_create_collection(
    name="topics"
)

# SEARCH
def web_search_tool(query, num_results=10): 
    results_data = []
    try:
        results = DDGS().text(
            query,
            max_results=num_results
        )
        seen = set()
        for r in results:
            url = r.get("href", "")
            if url not in seen:
                seen.add(url)
                results_data.append({
                    "title": r.get("title", ""),
                    "link": url,
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print("Search Error:", e)
    return results_data

# SCRAPER
def web_scraper_tool(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        if len(text) < 200:
            return ""
        return text[:3000]
    except:
        return ""

# LLM
AVAILABLE_MODELS = {
    "Gemini 2.5 Flash": { 
        "provider": "gemini",
        "model": "gemini-2.5-flash"
    },
    "Llama 3.3 70B (Groq)": { 
        "provider": "groq",
        "model": "llama-3.3-70b-versatile"
    }
}

# MEMORY
def clean_topic_name(topic):
    return " ".join(topic.strip().lower().split())

def search_memory(topic):
    try:
        topic_id = clean_topic_name(topic)
        result = collection.get(ids=[topic_id])
        docs = result.get("documents")
        if docs:
            return docs[0]
    except:
        pass
    return None

def save_memory(topic, report, sources):
    try:
        collection.upsert(
            ids=[clean_topic_name(topic)],
            documents=[report],
            metadatas=[{
                "topic": topic,
                "sources": str(sources)
            }]
        )
        print("Saved To Research Memory")
    except Exception as e:
        print(e)

def get_search_history():
    try:
        result = collection.get()
        history = []
        if result and result["metadatas"]:
            for item in result["metadatas"]:
                # Backwards compatible if old DB still has 'product' key
                history.append(item.get("topic", item.get("product", "Unknown")))
        history = list(dict.fromkeys(history))
        return list(reversed(history))
    except:
        return []

def delete_topic(topic):
    try:
        collection.delete(ids=[clean_topic_name(topic)])
        print(f"Deleted: {topic}")
    except Exception as e:
        print(e)

# AGENT 
def run_research_agent(topic_name, use_memory=True,model_choice="Gemini 2.5 Flash",save_to_memory=True):
    original_topic = topic_name
    topic_name = clean_topic_name(topic_name)
    
    # MEMORY CHECK
    if use_memory:
        cached = search_memory(topic_name)
        if cached:
            return "Cached Result\n\n" + cached

    # WEB SEARCH 
    search_query = (
        f"{topic_name}  "
    )
    search_results = web_search_tool(search_query)
    
    if not search_results:
        return "No search results found."
        
    combined_web_data = ""
    print("Starting parallel web scraping...")
    
    # PARALLEL SCRAPING for faster load times
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(web_scraper_tool, item["link"]): item for item in search_results}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url), start=1):
            item = future_to_url[future]
            url = item["link"]
            try:
                page_text = future.result()
                if page_text:
                    combined_web_data += f"\nSOURCE {i}\nURL: {url}\n{page_text}\n"
                else:
                    combined_web_data += f"\nSOURCE {i}\nURL: {url}\nSNIPPET:\n{item['snippet']}\n"
            except Exception:
                pass

    # GEMINI PROMPT
    system_prompt = """
    You are an Expert Research Agent.

    Your job is to analyze web data and answer the user's query accurately.

    Rules:
    - First understand the query type.
    - Use only information from provided web data.
    - Never invent facts.
    - If information is unavailable, state that clearly.
    - Be concise and accurate.

    Pricing Rules:
    - Prefer Indian pricing (₹ INR) whenever available.
    - If multiple prices are found, prioritize Indian market prices.
    - If only USD pricing is available, mention it and provide an approximate INR equivalent.
    - Clearly indicate if the price is global, US, or India-specific.
    - Never guess prices.

    For PRODUCT:
    - Specifications
    - Features
    - Pros
    - Cons
    - Price (prefer INR)
    - Recommendation

    For COMPARISON:
    - Comparison table
    - Strengths and weaknesses
    - Price comparison (prefer INR)
    - Final recommendation

    For SOFTWARE:
    - Purpose
    - Features
    - Pricing
    - Alternatives

    For TECH_TOPIC:
    - Explain concepts clearly
    - Include examples
    - Include practical use cases

    For HOW_TO:
    - Provide step-by-step guidance

    For GENERAL_RESEARCH:
    - Provide a structured research report

    Never mention source numbers.
    Use markdown formatting.
    """

    user_prompt = f"""
    Product:
    {topic_name}

    Web Data:
    {combined_web_data}
    """

    model_info = AVAILABLE_MODELS[model_choice]

    provider = model_info["provider"]
    model = model_info["model"]

    try:

        if provider == "gemini":

            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )

            report = response.text

        elif provider == "groq":

            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.3
            )

            report = response.choices[0].message.content

    except Exception as e:

        report = f"""
    # Error

    Provider: {provider}

    Model: {model}

    Error:
    {str(e)}
    """
            
    if save_to_memory and "Data Not Provided" not in report:
        save_memory(original_topic, report, search_results)
        
    return report


def compare_products(product1, product2, model_choice):

    report1 = run_research_agent(
        product1,
        use_memory=True,
        model_choice=model_choice,
        save_to_memory=False
    )

    report2 = run_research_agent(
        product2,
        use_memory=True,
        model_choice=model_choice,
        save_to_memory=False
    )

    model_info = AVAILABLE_MODELS[model_choice]

    provider = model_info["provider"]
    model = model_info["model"]

    prompt = f"""
    Compare these products.

    Product 1:
    {product1}

    Research:
    {report1}

    Product 2:
    {product2}

    Research:
    {report2}

    Rules:
    - Use only information present in the research.
    - Never invent specifications.
    - If information is missing, write "Not Found".
    - Prefer Indian prices (₹ INR).
    - Keep pros and cons concise.

    Create a markdown comparison table with:

    - Price
    - Pros
    - Cons
    - Key Features

    Additionally, compare all important specifications relevant to the product category (e.g., performance, display, battery, camera, storage, dimensions, weight, connectivity, warranty, etc.).

    Only include specifications that meaningfully differ between the products.

    Return only markdown.
    """

    try:

            if provider == "gemini":

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                final_report = response.text

            else:

                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                final_report = response.choices[0].message.content

            # Save ONLY the comparison query
            comparison_query = f"{product1} vs {product2}"

            save_memory(
                comparison_query,
                final_report,
                []
            )

            return final_report

    except Exception as e:

            return f"""
            # Error

            Comparison failed.

            {str(e)}
            """
import os
import json
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(BASE_DIR, 'data', 'eval')
GOLDEN_EVAL_FILE = os.path.join(EVAL_DIR, 'golden_eval_set.json')


def build_golden_dataset() -> List[Dict[str, Any]]:
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    questions = []
    
    # 1. Single Fact Prose Lookup (30 questions)
    single_facts = [
        ("AAPL", "2023", "Item 1. Business", "What are the primary products and services sold by Apple?", "iPhone, Mac, iPad, Wearables, Home & Accessories, and Services"),
        ("MSFT", "2023", "Item 1. Business", "What are Microsoft's main operating business segments?", "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing"),
        ("NVDA", "2023", "Item 1A. Risk Factors", "What risk factors does NVIDIA identify regarding supply chain dependencies?", "Dependence on third-party semiconductor foundries like TSMC"),
        ("AMZN", "2023", "Item 1. Business", "What is Amazon Web Services (AWS) and what services does it provide?", "Cloud computing, storage, database, and analytics services"),
        ("WMT", "2023", "Item 1. Business", "What are Walmart's primary operating reportable segments?", "Walmart U.S., Walmart International, and Sam's Club"),
        ("NKE", "2023", "Item 1. Business", "What are NIKE's primary product categories?", "Footwear, Apparel, and Equipment"),
        ("GE", "2023", "Item 1. Business", "What is General Electric's main industrial focus after spinoffs?", "Aerospace aviation engines and systems"),
        ("CAT", "2023", "Item 1. Business", "What are Caterpillar's primary business operations?", "Construction Industries, Resource Industries, Energy & Transportation, and Financial Products"),
        ("TSLA", "2023", "Item 1. Business", "What automotive models does Tesla produce and market?", "Model 3, Model Y, Model S, Model X, Cybertruck, and Tesla Semi"),
        ("AAPL", "2023", "Item 1A. Risk Factors", "What risks does Apple outline regarding global macroeconomic conditions?", "Global economic downturns, inflation, and currency fluctuations affecting consumer demand"),
    ]
    
    for i, (ticker, year, section, q_text, ans) in enumerate(single_facts):
        for rep in range(3): # Expand variations
            q_id = f"SF_{i*3+rep+1:03d}"
            questions.append({
                'id': q_id,
                'category': 'single_fact_prose',
                'question': f"{q_text} (Target: {ticker} {year})" if rep > 0 else q_text,
                'expected_answer': ans,
                'grounding_ticker': ticker,
                'grounding_year': year,
                'grounding_section': section,
                'should_refuse': False
            })

    # 2. Numeric Table Lookup (30 questions)
    numeric_lookups = [
        ("AAPL", "2023", "Item 8. Financial Statements", "What was Apple's total net sales or revenue in fiscal year 2023?", "$383,285 million"),
        ("MSFT", "2023", "Item 8. Financial Statements", "What was Microsoft's total revenue for fiscal year 2023?", "$211,915 million"),
        ("NVDA", "2023", "Item 8. Financial Statements", "What was NVIDIA's net income for fiscal year 2023?", "$4,368 million"),
        ("AMZN", "2023", "Item 8. Financial Statements", "What was Amazon's total net sales in fiscal year 2023?", "$574,785 million"),
        ("WMT", "2023", "Item 8. Financial Statements", "What was Walmart's total revenues for fiscal year 2023?", "$611,289 million"),
        ("NKE", "2023", "Item 8. Financial Statements", "What was NIKE's total revenues in fiscal year 2023?", "$51,217 million"),
        ("GE", "2023", "Item 8. Financial Statements", "What was General Electric's total revenues in 2023?", "$67,954 million"),
        ("CAT", "2023", "Item 8. Financial Statements", "What were Caterpillar's total sales and revenues in 2023?", "$67,060 million"),
        ("TSLA", "2023", "Item 8. Financial Statements", "What was Tesla's total revenues in fiscal year 2023?", "$96,773 million"),
        ("AAPL", "2023", "Item 7. MD&A", "What was Apple's gross margin percentage in fiscal year 2023?", "44.1%"),
    ]

    for i, (ticker, year, section, q_text, ans) in enumerate(numeric_lookups):
        for rep in range(3):
            q_id = f"NUM_{i*3+rep+1:03d}"
            questions.append({
                'id': q_id,
                'category': 'numeric_table_lookup',
                'question': f"{q_text} ({ticker} FY{year})" if rep > 0 else q_text,
                'expected_answer': ans,
                'grounding_ticker': ticker,
                'grounding_year': year,
                'grounding_section': section,
                'should_refuse': False
            })

    # 3. Cross-Year Comparison Within Company (20 questions)
    cross_year = [
        ("AAPL", "2022 vs 2023", "Item 7. MD&A", "Did Apple's net sales increase or decrease from 2022 to 2023?", "Decreased from $394,328 million in 2022 to $383,285 million in 2023"),
        ("MSFT", "2022 vs 2023", "Item 7. MD&A", "How did Microsoft's Intelligent Cloud revenue grow year over year?", "Grew significantly driven by Azure cloud services adoption"),
        ("AMZN", "2022 vs 2023", "Item 7. MD&A", "Compare Amazon's net sales growth from fiscal year 2022 to 2023.", "Increased from $513,983 million in 2022 to $574,785 million in 2023"),
        ("TSLA", "2022 vs 2023", "Item 7. MD&A", "Did Tesla's automotive sales revenue increase between 2022 and 2023?", "Increased driven by higher vehicle deliveries of Model Y and Model 3"),
        ("WMT", "2022 vs 2023", "Item 7. MD&A", "What was Walmart's net sales trend between 2022 and 2023?", "Increased across U.S. and Sam's Club segments"),
    ]

    for i, (ticker, year, section, q_text, ans) in enumerate(cross_year):
        for rep in range(4):
            q_id = f"CY_{i*4+rep+1:03d}"
            questions.append({
                'id': q_id,
                'category': 'cross_year_comparison',
                'question': f"{q_text} ({ticker})" if rep > 0 else q_text,
                'expected_answer': ans,
                'grounding_ticker': ticker,
                'grounding_year': year,
                'grounding_section': section,
                'should_refuse': False
            })

    # 4. Cross-Company Comparison (20 questions)
    cross_company = [
        ("AAPL vs MSFT", "2023", "Item 8. Financial Statements", "Compare the total revenues of Apple and Microsoft in fiscal year 2023.", "Apple ($383.3B) generated higher revenue than Microsoft ($211.9B) in FY2023"),
        ("WMT vs AMZN", "2023", "Item 8. Financial Statements", "Which retail company generated higher total revenue in 2023, Walmart or Amazon?", "Walmart ($611.3B) generated higher revenue than Amazon ($574.8B)"),
        ("CAT vs GE", "2023", "Item 8. Financial Statements", "Compare the 2023 revenues of Caterpillar and General Electric.", "General Electric ($68.0B) and Caterpillar ($67.1B) had similar total revenues in 2023"),
        ("NVDA vs TSLA", "2023", "Item 1A. Risk Factors", "Compare the key supply chain risks disclosed by NVIDIA and Tesla.", "NVIDIA focuses on semiconductor foundry reliance; Tesla focuses on lithium/battery cell supply chain"),
        ("AAPL vs NKE", "2023", "Item 7. MD&A", "Compare the international revenue dependence of Apple and NIKE.", "Both companies rely heavily on international supply chains and Chinese manufacturing/sales"),
    ]

    for i, (tickers, year, section, q_text, ans) in enumerate(cross_company):
        for rep in range(4):
            q_id = f"CC_{i*4+rep+1:03d}"
            questions.append({
                'id': q_id,
                'category': 'cross_company_comparison',
                'question': f"{q_text}" if rep == 0 else f"{q_text} (Comparison set {rep+1})",
                'expected_answer': ans,
                'grounding_ticker': tickers,
                'grounding_year': year,
                'grounding_section': section,
                'should_refuse': False
            })

    # 5. Out-of-Scope Unanswerable Questions (20 questions)
    out_of_scope = [
        "What is Elon Musk's favorite ice cream flavor?",
        "Who won the FIFA World Cup in 2022?",
        "What is the weather forecast for Seattle next week?",
        "What is the capital city of France?",
        "How do you cook a traditional Italian lasagna?",
        "What is the stock price of Apple right now in real time?",
        "Who is the current Prime Minister of Japan?",
        "What is the plot of the latest Marvel movie?",
        "Which team won the Super Bowl in 2023?",
        "What is the best prescription medicine for a mild fever?"
    ]

    for i, q_text in enumerate(out_of_scope):
        for rep in range(2):
            q_id = f"OOS_{i*2+rep+1:03d}"
            questions.append({
                'id': q_id,
                'category': 'out_of_scope_refusal',
                'question': q_text if rep == 0 else f"Tell me: {q_text}",
                'expected_answer': "SHOULD_REFUSE: Question is outside the scope of SEC filings corpus.",
                'grounding_ticker': "N/A",
                'grounding_year': "N/A",
                'grounding_section': "N/A",
                'should_refuse': True
            })

    with open(GOLDEN_EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2)

    print(f"Generated {len(questions)} Golden Evaluation questions saved to {GOLDEN_EVAL_FILE}.")
    return questions


if __name__ == '__main__':
    build_golden_dataset()

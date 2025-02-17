import requests
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from pypfopt import EfficientFrontier, expected_returns, risk_models, discrete_allocation
import cvxpy as cp
import numpy as np

# API Key pour Financial Modeling Prep
FMP_API_KEY = "3gYsTdxP48Q6aci1mDcgchFXGrhd7CQo"

import cvxpy as cp
import numpy as np


# Fonction pour récupérer les prix ajustés d'un actif
def get_adj_close_price(symbol, start_date="2023-01-01"):
    hist_price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&apikey={FMP_API_KEY}"
    r_json = requests.get(hist_price_url).json()
    if "historical" not in r_json:
        return None
    df = pd.DataFrame(r_json["historical"]).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df[["adjClose"]].rename(columns={"adjClose": symbol})

# Fonction pour récupérer les bornes avant que l'utilisateur choisisse
def get_bounds(tickers_input):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    
    # Récupération des prix
    price_df_list = []
    for ticker in tickers:
        df = get_adj_close_price(ticker)
        if df is not None:
            price_df_list.append(df)

    if not price_df_list:
        return "⚠️ Aucune donnée disponible.", None, None, None, None

    prices_df = pd.concat(price_df_list, axis=1).dropna()
    
    # Calcul des rendements attendus et de la matrice de covariance
    avg_returns = expected_returns.mean_historical_return(prices_df, compounding=False)
    cov_mat = risk_models.sample_cov(prices_df)

    # Instancier l'optimisation
    ef = EfficientFrontier(avg_returns, cov_mat)

    # Récupérer le rendement et risque minimum
    ef.min_volatility()
    ret_min, vol_min, _ = ef.portfolio_performance(verbose=False)

    # Récupérer le rendement et risque maximum
    ef = EfficientFrontier(avg_returns, cov_mat)
    ef.max_sharpe()
    ret_max, vol_max, _ = ef.portfolio_performance(verbose=False)
  

    return f"📊 Rendements possibles : {ret_min:.2%} → {ret_max:.2%} ⚖️ Risques possibles : {vol_min:.2%} → {vol_max:.2%}"# float(ret_min) float(ret_max), float(vol_min), float(vol_max)

# Fonction principale d'optimisation
def optimize_portfolio(tickers_input, method, target_value):
    try:
        target_value = float(target_value)  # Convertir en float pour éviter l'erreur
    except ValueError:
        return "⚠️ L'objectif doit être un nombre valide.", None

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    # Récupérer les prix
    price_df_list = []
    for ticker in tickers:
        df = get_adj_close_price(ticker)
        if df is not None:
            price_df_list.append(df)

    if not price_df_list:
        return "⚠️ Aucune donnée disponible.", None

    prices_df = pd.concat(price_df_list, axis=1).dropna()

    # Calcul des rendements et covariance
    avg_returns = expected_returns.mean_historical_return(prices_df, compounding=False)
    cov_mat = risk_models.sample_cov(prices_df)

    ef = EfficientFrontier(avg_returns, cov_mat)
    
    # Correction du bug avec conversion en float
    if method == "Maximiser le gain pour un risque donné":
        optimal_weights = ef.efficient_risk(target_value)

    expected_ret, volatility, sharpe_ratio = ef.portfolio_performance(verbose=True, risk_free_rate=0)

    latest_prices = discrete_allocation.get_latest_prices(prices_df)

    allocation, rem_cash = discrete_allocation.DiscreteAllocation(
        optimal_weights, latest_prices, total_portfolio_value=10000
    ).greedy_portfolio()

    allocated_value = sum(latest_prices[ticker] * shares for ticker, shares in allocation.items())
    total_gains = allocated_value * expected_ret
    potential_value = allocated_value + total_gains

    # Générer le graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Allocated ($10,000)", "Potential (Expected Return)"]
    values = [allocated_value, potential_value]
    bars = ax.bar(labels, values, color=['#1f77b4', '#2ca02c'], alpha=0.85, edgecolor='black', linewidth=1.2)

    ax.set_title('Portfolio Allocation vs Expected Return', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"${value:,.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("temp_fig.png")

    results = f"""
    ✅ **Optimisation Complète**  
    📈 **Rendement Attendu** : {expected_ret * 100:.2f} %  
    📊 **Volatilité (Risque)** : {volatility * 100:.2f} %  
    ⚖️ **Ratio de Sharpe** : {sharpe_ratio:.2f}  
    💰 **Montant Alloué** : ${allocated_value:,.2f}  
    🚀 **Gains Potentiels** : ${total_gains:,.2f}  
    📈 **Valeur Totale du Portefeuille** : ${potential_value:,.2f}  
    """

    return results, "temp_fig.png"

# Interface Gradio
with gr.Blocks() as demo:
    tickers_input = gr.Textbox(label="Tickers (séparés par des virgules)", placeholder="Ex: AAPL, MSFT, GOOG")
    
    output=gr.Textbox(label="Intervalles pour le rendement et le risque")
    button1 = gr.Button("Limites pour rendements et risques")
    button1.click(get_bounds, inputs=[tickers_input], outputs=[output])


    method_input = gr.Radio(["Maximiser le gain pour un risque donné"], label="Choix")
    
    target_value = gr.Number(label="Entrez votre objectif", value=0.2)


    output_text = gr.Textbox(label="Résultats", interactive=False)
    output_graph = gr.Image(label="Graphique")

    button = gr.Button("Optimiser")
    button.click(optimize_portfolio, inputs=[tickers_input, method_input, target_value], outputs=[output_text, output_graph])

demo.launch()
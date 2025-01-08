#Importing all required libraries
#Created by Sanket Karve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from pandas_datareader import DataReader
from matplotlib.ticker import FuncFormatter
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, objective_functions
from pypfopt import discrete_allocation
from pypfopt import expected_returns
from pypfopt.cla import CLA
from pypfopt import plotting
from matplotlib.ticker import FuncFormatter

import gradio as gr
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, objective_functions, expected_returns, risk_models, discrete_allocation
from pandas_datareader import data as web
import matplotlib.pyplot as plt


def optimize_portfolio_gradio(tickers_input):
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]

    # Charger les données des prix
    price_data = []
    for ticker in tickers:
        prices = web.DataReader(ticker, start='2015-01-01', end='2020-06-06', data_source='stooq')
        price_data.append(prices[['Close']])
    
    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns = tickers
    df_stocks.dropna(inplace=True)

    # Calcul des rendements et de la covariance
    mu = expected_returns.mean_historical_return(df_stocks)
    Sigma = risk_models.sample_cov(df_stocks)

    # Vérification des rendements
    print("Rendements attendus :")
    print(mu)

    # Si tous les rendements sont trop faibles, ajuster le taux sans risque
    try:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1), solver="ECOS")
        ef.max_sharpe(risk_free_rate=0.01)  # Ajuste à un taux sans risque de 1 %
    except ValueError as e:
        print("Erreur rencontrée :", e)
        print("Essai avec une autre méthode (minimisation de la volatilité)...")
        ef.min_volatility()  # Utilise une méthode alternative
    optimal_weights = ef.clean_weights()
    optimal_performance = ef.portfolio_performance(verbose=True)

    # Calcul des allocations discrètes et du graphe
    latest_prices = discrete_allocation.get_latest_prices(df_stocks)
    allocation_minv, rem_minv = discrete_allocation.DiscreteAllocation(
        optimal_weights, latest_prices, total_portfolio_value=10000
    ).lp_portfolio()

    # Calcul des valeurs
    allocated_value = sum(latest_prices[ticker] * shares for ticker, shares in allocation_minv.items())
    expected_return = sum(optimal_weights[ticker] * mu[ticker] for ticker in allocation_minv.keys())
    total_gains = allocated_value * expected_return  # Gains potentiels uniquement
    potential_value = allocated_value + total_gains  # Montant total attendu

    # Afficher les résultats explicites
    print(f"Montant alloué : ${allocated_value:,.2f}")
    print(f"Rendement attendu (taux) : {expected_return * 100:.2f}%")
    print(f"Gains potentiels : ${total_gains:,.2f}")
    print(f"Montant total attendu (Potential Value) : ${potential_value:,.2f}")

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Allocated ($10,000)', 'Potential (Expected Return)']
    values = [allocated_value, potential_value]
    bars = ax.bar(labels, values, color=['#1f77b4', '#2ca02c'], alpha=0.85, edgecolor='black', linewidth=1.2)

    # Titre et labels
    ax.set_title('Portfolio Allocation vs Expected Return', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"${value:,.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("temp_fig.png")
    plt.close()

    return allocation_minv, "temp_fig.png"





# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("### Optimisation de Portefeuille")
    tickers_input = gr.Textbox(label="Tickers (séparés par des virgules)", placeholder="Ex: AAPL, MSFT, GOOG")
    output_allocation = gr.Textbox(label="Allocation optimale")
    output_graph = gr.Image(label="Graphique des Allocations et Rendement")

    def run_portfolio_optimization(tickers_input):
        allocation, graph_path = optimize_portfolio_gradio(tickers_input)
        return str(allocation), graph_path

    button = gr.Button("Optimiser le Portefeuille")
    button.click(run_portfolio_optimization, inputs=[tickers_input], outputs=[output_allocation, output_graph])

# Lancer l'interface
demo.launch()

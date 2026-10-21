"""
Build a payment transaction dataset using cardsim
"""

from importlib import resources
from cardsim import Cardsim

dcpc_path = resources.files("cardsim.dcpc")

simulator = Cardsim(seed=212, dcpc_folder=dcpc_path)

df = simulator.simulate(
    n_payers=1000,
    n_days=180,
    start_date="2023-01-01"
)

simulator.export_transaction_data(df, folder='data')

params = simulator.export_run_parameters(
    df, folder='data', file_name='cardsim_runs'
)

print("\nRows and columns:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nFraud counts:")
print(df["fraud"].value_counts())
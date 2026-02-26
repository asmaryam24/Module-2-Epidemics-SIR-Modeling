import os
from pathlib import Path
HERE = Path(__file__).resolve().parent

# Because your CSV is now in the SAME folder as this .py file
csv_path = HERE / "mystery_virus_daily_active_counts_RELEASE#1.csv"

print("Using CSV at:", csv_path)


from pathlib import Path
HERE = Path(__file__).resolve().parent
sir_csv = HERE / "in_class_SIR_data.csv"   # <-- OR whatever the real name is

df = pd.read_csv(sir_csv)



print("CWD:", os.getcwd())
print("\nHere are files/folders in the current directory:\n")
for p in Path(".").iterdir():    
    print("-", p)
print("\nCSV files I can see, recursively:\n")
for p in Path(".").rglob("*.csv"):    
    print("-", p)






import pandas as pd
import matplotlib.pyplot as plt

# -Load your S, I, R data -

# Replace with the correct filename from your repo

df = pd.read_csv("ANALYSIS DATA RELEASE #1 data/in_class_SIR_data.csv")
days = df["day"]
S = df["S"]
I = df["I"]
R = df["R"]

# - Plot -
plt.figure(figsize=(8,5))
plt.plot(days, S, label="Susceptible (S)")
plt.plot(days, I, label="Infected (I)")
plt.plot(days, R, label="Recovered (R)")

plt.xlabel("Day")
plt.ylabel("Population size")
plt.title("SIR Compartments Over Time")
plt.legend()
plt.grid(True)
plt.show()
# filename: rivers_table.py

# Define the data
rivers = [("Nile", 6650), ("Congo", 4700), ("Niger", 4180), ("Zambezi", 2574), ("Orange", 2092)]

# Print the table in markdown format
print("| River        | Length (km) |\n")
print("|--------------|---------------|\n")
for river, length in rivers:
    print(f"| {river}          | {length}         |\n")